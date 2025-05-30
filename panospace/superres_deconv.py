import os
import cv2
import numpy as np
import scanpy as sc
import anndata as ad
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from utils import if_contain_batch

from typing import Literal, Union, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import AutoModel, AutoImageProcessor
from torchvision.transforms import ToTensor
from torchvision import transforms

class DINOv2NeighborDataset(Dataset):
    def __init__(self, centers, img_path, label_frame=None, train=True,
                 radius=129, neighb=3, path=None):
        self.centers = centers
        self.label_frame = label_frame
        self.train = train
        self.radius = radius
        self.neighb = neighb

        self.image = Image.open(img_path).convert("RGB")
        self.image.load()

        self.transform = ImageTransform(resize=518, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        x, y = self.centers[index]
        r, n = self.radius, self.neighb

        crop = self.image.crop((x - r, y - r, x + r, y + r))
        crop_neighbor = self.image.crop((x - r * n, y - r * n, x + r * n, y + r * n))

        if self.train:
            crop = self.transform(img=crop, phase="valid")
            crop_neighbor = self.transform(img=crop_neighbor, phase="valid")
            label = self.label_frame.iloc[index, :].values.astype(np.float32)
            return crop, crop_neighbor, label
        else:
            crop = self.transform(img=crop, phase="valid")
            crop_neighbor = self.transform(img=crop_neighbor, phase="valid")
            return crop, crop_neighbor

    def __len__(self):
        return len(self.centers)
    

class DINOv2NeighborClassifier(pl.LightningModule):
    def __init__(self, num_classes=9, class_weights=None, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.vit = AutoModel.from_pretrained('/mnt/Fold/hehf/dinov2-base', local_files_only=True)

        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, crop, crop_neighbor):
        x1 = self._extract_feature(crop)
        x2 = self._extract_feature(crop_neighbor)
        x = torch.cat((x1, x2), dim=1)  # [B, 1536]
        x = self.classifier(x)
        x = self.softmax(x)
        return x

    def _extract_feature(self, images):
        with torch.no_grad():
            outputs = self.vit(images)
        cls_embedding = outputs.pooler_output  # [B, 768]
        return cls_embedding

    def training_step(self, batch, batch_idx):
        crop, crop_neighbor, label = batch
        pred = self(crop, crop_neighbor)  # [B, num_classes]
        loss = F.kl_div(pred.log(), label, reduction='none')

        if self.class_weights is not None:
            weights = self.class_weights.unsqueeze(0).to(self.device)
            loss = loss * weights

        loss = torch.mean(torch.sum(loss, dim=1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(list(self.classifier.parameters()), lr=self.learning_rate)
        return optimizer
    
    
class DINOv2_superres_deconv(object):
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

        if os.path.exists(os.path.join(self.path,"sr/superres_model.ckpt")):
            print('the checkpoint exists, loading the checkpoint...')
            print('if use the checkpoint, do not execute run_train method')
            self.model = DINOv2NeighborClassifier.load_from_checkpoint(os.path.join(self.path,"sr/superres_model.ckpt"),num_classes=num_classes)
        else:
            self.model = DINOv2NeighborClassifier(num_classes=num_classes)
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

        I = Image.open(self.img_dir).convert("RGB")
        I = np.array(I)[:, :, ::-1].copy()
        for i in tqdm(range(subspot_centers.shape[0])):
            pos_x1 = subspot_centers[i,0]-r
            pos_x2 = subspot_centers[i,0]+r
            pos_y1 = subspot_centers[i,1]-r
            pos_y2 = subspot_centers[i,1]+r
            I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (0, 0, 0), 2)
            I = cv2.putText(I, str(i), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
        cv2.imwrite(self.path+"/sr/sr.png", I) 
        I = Image.open(self.img_dir).convert("RGB")
        I = np.array(I)[:, :, ::-1].copy()

        to_tensor = ToTensor()
        valid_spot_indices = []
        spot_centers = spot_centers.astype(int)
        for i in tqdm(range(spot_centers.shape[0])):
            pos_x1 = spot_centers[i,0]-r
            pos_x2 = spot_centers[i,0]+r
            pos_y1 = spot_centers[i,1]-r
            pos_y2 = spot_centers[i,1]+r

            I = cv2.rectangle(I, (pos_x1,pos_y1), (pos_x2,pos_y2), (255, 0, 0), 2)
            I = cv2.putText(I, str(i), (pos_x1,pos_y1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA, thickness=2)

            valid_spot_indices.append(i)
        cv2.imwrite(self.path+"/sr/patch_totrain.png", I) 
        sr_adata = ad.AnnData(np.zeros((subspot_centers.shape[0],1)))
        sr_adata.obsm['spatial'] = subspot_centers

        self.deconv_adata = self.deconv_adata[valid_spot_indices,:]
        return sr_adata
    
    def pred(self,dataloader,device=torch.device('cuda')):
        self.model.eval()
        model = self.model.to(device)

        pred = np.zeros((len(dataloader.dataset), self.num_classes))

        current_index = 0
        for crop, crop_neighbor in tqdm(dataloader):
            crop = crop.to('cuda')
            crop_neighbor = crop_neighbor.to('cuda')
            outputs = model(crop, crop_neighbor)
            batch_size = outputs.shape[0]
            pred[current_index:current_index+batch_size] = outputs.cpu().detach().squeeze(1).numpy()
            current_index += batch_size
        print(pred)
        return pred
    
    def run_train(self, epoch=50, batch_size=256, num_workers=4, accelerator='gpu'):
        deconv = self.deconv_adata.obs[self.cell_type_name]
        deconv[deconv < 0] = 0
        deconv = (deconv.T/deconv.sum(1)).T

        dataset = DINOv2NeighborDataset(centers=self.deconv_adata.obsm['spatial'],
                                        img_path=self.img_dir,
                                        label_frame=deconv,
                                        radius=self.radius,
                                        neighb=self.neighb,
                                        path=self.path)

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        trainer = pl.Trainer(max_epochs=epoch, accelerator=accelerator, devices=1)
        trainer.fit(self.model, dataloader)

        trainer.save_checkpoint(os.path.join(self.path,"sr/superres_model.ckpt"))

    def run_superres(self):
        dataset = DINOv2NeighborDataset(centers=self.sr_adata.obsm['spatial'],
                                        img_path=self.img_dir,
                                        label_frame=None,
                                        train=False,
                                        radius=self.radius, 
                                        neighb=self.neighb,
                                        path=self.path)

        dataloader = DataLoader(dataset, batch_size=256, num_workers=4)
        predict = self.pred(dataloader)
        print(predict)
        self.sr_adata.obs[self.cell_type_name] = predict

        self.sr_adata.write(os.path.join(self.path,'adata/sr_adata.h5ad'))


class ImageTransform:
    def __init__(self, resize: int, mean: list[float], std: list[float]):
        self.resize = resize

        self.base_resize = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resize)
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.augmentations = {
            'flip': transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]),
            'noise': transforms.Compose([
                transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            ]),
            'blur': transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
            'dist': transforms.Compose([
                transforms.RandomAffine(degrees=30, shear=10),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            ]),
            'contrast': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            'color': transforms.ColorJitter(hue=0.1),
            'crop': transforms.RandomResizedCrop(size=resize, scale=(0.5, 1.0)),
            'random': transforms.Compose([
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=30, shear=10),
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
                ], p=0.5),
            ])
        }

        self.valid_transform = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img: Image.Image, phase: Literal['train', 'valid'] = 'train', param: str = 'none') -> torch.Tensor:
        if phase == 'train':
            img = self.base_resize(img)

            if param != 'none':
                for para in param.split(','):
                    if para in self.augmentations:
                        img = self.augmentations[para](img)
                    else:
                        raise ValueError(f"Unknown augmentation parameter: {para}")

            img = self.to_tensor(img)

        elif phase == 'valid':
            img = self.valid_transform(img)

        else:
            raise ValueError("phase must be 'train' or 'valid'")

        return img
    
    def transform_batch(self, imgs: Union[List[Image.Image], Image.Image], phase="train", param='none') -> torch.Tensor:
        if isinstance(imgs, Image.Image):
            imgs = [imgs]

        results = [self.__call__(img, phase=phase, param=param) for img in imgs]
        return torch.stack(results)  # shape: [B, C, H, W]
    

if __name__ == "__main__":

    sample = 'Xenium'
    num_classes=9
    Experimental_path = os.path.join('dataset',sample)
    img_dir = os.path.join(Experimental_path, 'Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image_registered.png')

    adata_dir = os.path.join('dataset',sample,'adata')
    deconv_adata = sc.read(os.path.join(adata_dir,'EnDecon_adata.h5ad'))
    segment_adata = sc.read(os.path.join(adata_dir,'img_adata_sc.h5ad'))

    sr_inferencer=DINOv2_superres_deconv(deconv_adata,
                                           segment_adata,
                                           img_dir,
                                           Experimental_path,
                                           neighb=3,
                                           radius=129,
                                           num_classes=num_classes)
    
    sr_inferencer.run_train()
    sr_inferencer.run_superres()

