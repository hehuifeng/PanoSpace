import os
import numpy as np
import scanpy as sc
import anndata as ad
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

from tqdm import tqdm
from itertools import product

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl

from .utils import *

class vgg_neighboor(pl.LightningModule):
    def __init__(self, num_classes=9, class_weights=None, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, device='cuda')

        # Load pre-trained VGG16 model and modify the last FC layer
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.softmax = nn.Softmax(dim=1)

        self.net = vgg16
        self.fc_1 = nn.Linear(2000, 512)
        self.fc_2 = nn.Linear(512, num_classes)
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.fc_1.parameters():
            param.requires_grad = True
        for param in self.fc_2.parameters():
            param.requires_grad = True        


    def forward(self, crop, crop_neighbor):
        x1 = self.net(crop)
        x2 = self.net(crop_neighbor)
        x = torch.cat((x1.squeeze(0),x2.squeeze(0)),1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        crop, crop_neighbor, label = batch
        pred = self(crop, crop_neighbor)
        loss = F.kl_div(pred.log(), label, reduction='none')
    
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqeeze(0) 
            
        loss = torch.mean(torch.sum(loss, dim=1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = list(self.fc_1.parameters()) + list(self.fc_2.parameters())
        optimizer = Adam(params, lr=self.learning_rate)
        return optimizer


class vgg_neighboor_dataset(torch.utils.data.Dataset):
    def __init__(self, centers, img_dir, label_frame=None, train=True, transform=None, radius=129, neighb=3, param=''):
        self.centers = centers
        self.label_frame = label_frame
        self.train = train
        self.radius = radius
        self.neighb=neighb
        self.transform = transform
        self.image_crop = [np.nan] * self.centers.shape[0]
        self.image_crop_neighboor = [np.nan] * self.centers.shape[0]

        self.phase = 'train' if train else 'valid'
        self.param=param
        self.image = Image.open(img_dir)
        self.image.load()
    
    def __getitem__(self, index):
        i = index
        if self.train:
            temp_image_crop = self.image.crop((self.centers[i,0]-self.radius, self.centers[i,1]-self.radius,
                                               self.centers[i,0]+self.radius, self.centers[i,1]+self.radius))
            
            temp_image_crop_neighboor = self.image.crop((self.centers[i,0]-self.radius*self.neighb,
                                                         self.centers[i,1]-self.radius*self.neighb,
                                                         self.centers[i,0]+self.radius*self.neighb,
                                                         self.centers[i,1]+self.radius*self.neighb))
            
            crop = self.transform(temp_image_crop, self.phase, self.param)
            crop_neighboor = self.transform(temp_image_crop_neighboor, self.phase, self.param)
            label = self.label_frame.iloc[index,:].values

            return crop, crop_neighboor, label.astype(np.float32)
        else:
            temp_image_crop = self.image.crop((self.centers[i,0]-self.radius, self.centers[i,1]-self.radius,
                                               self.centers[i,0]+self.radius, self.centers[i,1]+self.radius))
            
            temp_image_crop_neighboor = self.image.crop((self.centers[i,0]-self.radius*self.neighb,
                                                         self.centers[i,1]-self.radius*self.neighb,
                                                         self.centers[i,0]+self.radius*self.neighb,
                                                         self.centers[i,1]+self.radius*self.neighb))
            
            crop = self.transform(temp_image_crop, self.phase, self.param)
            crop_neighboor = self.transform(temp_image_crop_neighboor, self.phase, self.param)

            return crop, crop_neighboor

    def __len__(self):
        return self.centers.shape[0]

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'init': transforms.Compose([
                transforms.Resize((resize, resize))
            ]),
            'end': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'flip': transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]),
            'noise': transforms.Compose([
                transforms.GaussianBlur(kernel_size=3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ]),
            'blur': transforms.Compose([
                transforms.GaussianBlur(kernel_size=3),
            ]),
            'dist': transforms.Compose([
                transforms.RandomAffine(degrees=30, shear=10),
                transforms.RandomPerspective(),
            ]),
            'contrast': transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            ]),
            'color': transforms.Compose([
                transforms.ColorJitter(hue=0.5),
            ]),
            'crop': transforms.Compose([
                transforms.RandomResizedCrop(size=(resize, resize), scale=(0.5, 1.0)),
            ]),
            'random': transforms.Compose([
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=30, shear=10),
                    transforms.RandomPerspective(),
                ], p=0.5),
            ]),
            'valid': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        }

    def __call__(self, img, phase='train', param=''):
        if phase == 'train':
            img = self.data_transform['init'](img)

            if param != 'none':
                param = param.split(',')
                
                for para in param:
                    img = self.data_transform[para](img)

            img = self.data_transform['end'](img)
        elif phase == 'valid':
            img = self.data_transform['valid'](img)
        
        return img


class superres_deconv(object):
    def __init__(self,deconv_adata,segment_adata,img_dir,Experimental_path,radius=129,neighb=2,num_classes=9):
        self.img_dir = img_dir
        self.path = Experimental_path
        if not os.path.exists(self.path+'/sr'):
            os.mkdir(self.path+'/sr')

        self.deconv_adata = deconv_adata
        self.segment_adata = segment_adata
        self.cell_type_name = list(self.deconv_adata.uns['celltype'])

        self.num_classes = num_classes
        self.radius=radius
        self.neighb = neighb
        self.augmentation = 'flip,crop,color,random'
        self.size = 224
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if os.path.exists(os.path.join(self.path,"sr/superres_model.ckpt")):
            print('the checkpoint exists, loading the checkpoint...')
            print('if use the checkpoint, do not execute run_train method')
            self.model = vgg_neighboor.load_from_checkpoint(os.path.join(self.path,"sr/superres_model.ckpt"),num_classes=num_classes)
        else:
            self.model = vgg_neighboor(num_classes=num_classes)
        print('model loaded...')
        print('loading super res data')
        if not os.path.exists(os.path.join(self.path,'adata/sr_adata.h5ad')):
            self.sr_adata = self.make_sr_datalist()
            self.sr_adata.write(os.path.join(self.path,'adata/sr_adata.h5ad'))
        else:
            self.sr_adata = sc.read(os.path.join(self.path,'adata/sr_adata.h5ad'))

    def make_sr_datalist(self):
        r = self.deconv_adata.uns['radius']
        spot_centers = self.deconv_adata.obsm['spatial']
        axis_x = range(spot_centers[:,0].min().astype(int),spot_centers[:,0].max().astype(int),r)
        axis_y = range(spot_centers[:,1].min().astype(int),spot_centers[:,1].max().astype(int),r)
        subspot_centers = np.array([*product([*axis_x],[*axis_y])])

        Q = if_contain_batch(subspot_centers, self.segment_adata.obsm['spatial'], r=r, norm = False) #Remove subspots that are too far away
        subspot_centers = subspot_centers[np.sum(Q,0) != 0]

        I = cv2.imread(self.img_dir)
        for i in range(subspot_centers.shape[0]):
            pos_x1 = subspot_centers[i,0]-r
            pos_x2 = subspot_centers[i,0]+r
            pos_y1 = subspot_centers[i,1]-r
            pos_y2 = subspot_centers[i,1]+r
            I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (0, 0, 0), 2)
            I = cv2.putText(I, str(i), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
        cv2.imwrite(self.path+"/sr/sr.png", I) 
        I = cv2.imread(self.img_dir)
        spot_centers = spot_centers.astype(int)
        for i in range(spot_centers.shape[0]):
            pos_x1 = spot_centers[i,0]-r
            pos_x2 = spot_centers[i,0]+r
            pos_y1 = spot_centers[i,1]-r
            pos_y2 = spot_centers[i,1]+r
            I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (255, 0, 0), 2)
            I = cv2.putText(I, str(i), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA, thickness=2)
        cv2.imwrite(self.path+"/sr/patch_totrain.png", I) 
        sr_adata = ad.AnnData(np.zeros((subspot_centers.shape[0],1)))
        sr_adata.obsm['spatial'] = subspot_centers

        return sr_adata
    
    def pred(self,dataloader,device=torch.device('cuda')):
        self.model.eval()
        model = self.model.to(device)
        pred = np.zeros((len(dataloader.dataset), self.num_classes))

        current_index = 0
        for img, img_neighboor in tqdm(dataloader):
            img = img.to(device)
            img_neighboor = img_neighboor.to(device)
    
            outputs = model(img, img_neighboor)
            batch_size = outputs.shape[0]
            pred[current_index:current_index+batch_size] = outputs.cpu().detach().numpy()
            current_index += batch_size
        return pred
    
    def run_train(self, epoch=50):
        deconv = self.deconv_adata.obs[self.cell_type_name]
        deconv[deconv < 0] = 0
        deconv = (deconv.T/deconv.sum(1)).T

        dataset = vgg_neighboor_dataset(centers=self.deconv_adata.obsm['spatial'],
                                        img_dir=self.img_dir,
                                        label_frame=deconv,
                                        transform=ImageTransform(self.size,self.mean,self.std),
                                        radius=self.radius,
                                        neighb=self.neighb,
                                        param=self.augmentation)

        dataloader = DataLoader(dataset, batch_size=256, num_workers=4)
        trainer = pl.Trainer(max_epochs=epoch, accelerator='gpu', devices=1)
        trainer.fit(self.model, dataloader)

        trainer.save_checkpoint(os.path.join(self.path,"sr/superres_model.ckpt"))

    def run_superres(self):
        dataset = vgg_neighboor_dataset(centers=self.sr_adata.obsm['spatial'],
                                        img_dir=self.img_dir,
                                        label_frame=None,
                                        train=False,
                                        transform=ImageTransform(self.size,self.mean,self.std),
                                        radius=self.radius, 
                                        neighb=self.neighb,
                                        param=self.augmentation)

        dataloader = DataLoader(dataset, batch_size=256, num_workers=4)
        predict = self.pred(dataloader)
        self.sr_adata.obs[self.cell_type_name] = predict

        self.sr_adata.write(os.path.join(self.path,'adata/sr_adata.h5ad'))

if __name__ == "__main__":

    sample = 'Xenium'
    num_classes=9
    Experimental_path = os.path.join('dataset',sample)
    img_dir = os.path.join(Experimental_path, 'Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image_registered.png')

    adata_dir = os.path.join('dataset',sample,'adata')
    deconv_adata = sc.read(os.path.join(adata_dir,'EnDecon_adata.h5ad'))
    segment_adata = sc.read(os.path.join(adata_dir,'img_adata_sc.h5ad'))

    sr_inferencer=superres_deconv(deconv_adata,
                                  segment_adata,
                                  img_dir,
                                  Experimental_path,
                                  neighb=2,
                                  num_classes=num_classes)
    
    sr_inferencer.run_train()
    sr_inferencer.run_superres()

