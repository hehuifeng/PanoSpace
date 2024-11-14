import os
import re
import glob
import shutil
import pathlib
import warnings
import subprocess

import numpy as np
import scipy.io as sio

import cv2

class train_hovernet(object):
    def __init__(self, pannuke_dir, focus='all',):
        self.focus = focus
        if os.path.exists(pannuke_dir):
            self.pannuke_dir = pannuke_dir
        else:
            raise FileNotFoundError(f"The path is not exist: {pannuke_dir}")
        
        self.save_root = os.path.join(self.pannuke_dir, 'Input')
        self.win_size = [256, 256]

    def download_pannuke(self):
        fold_urls = {
            'fold_1': 'https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip',
            'fold_2': 'https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip',
            'fold_3': 'https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip'
        }

        fold_dir = os.path.join(self.pannuke_dir, 'Fold')
        os.makedirs(fold_dir, exist_ok=True)

        for fold_name, url in fold_urls.items():
            zip_path = os.path.join(fold_dir, f'{fold_name}.zip')
            subprocess.run(['wget', '-O', zip_path, url], check=True)
            subprocess.run(['unzip', zip_path, '-d', fold_dir], check=True)
    
    def split_pannuke(self):

        def map_inst(inst):
            seg_indexes = np.unique(inst)
            new_indexes = np.array(range(0, len(seg_indexes)))
            dict = {}
            for seg_index, new_index in zip(seg_indexes, new_indexes):
                dict[seg_index] = new_index

            flat_for(inst, lambda x: dict[x])

        def flat_for(a, f):
            a = a.reshape(-1)
            for i, v in enumerate(a):
                a[i] = f(v)

        def myoverlay(image, msk):

            from skimage import segmentation   
            color_pannuke = {
                #"0" : ["nolabe", [0  ,   0,   0]], 
                "0" : ["neopla", [255,   0,   0]], 
                "1" : ["inflam", [0  , 255,   0]], 
                "2" : ["connec", [0  ,   0, 255]], 
                #"4" : ["necros", [255, 255,   0]], 
                "4" : ["no-neo", [255, 165,   0]] 
            }
            ct = [0,1,2,4]

            img = image.copy()
            for i in  ct:  
                overlay = msk[:,:,i]
                img = segmentation.mark_boundaries(img,overlay,color=color_pannuke[str(i)][1], mode='thin')

            return img
        
        def load(f, select):
            bdir = os.path.join(self.pannuke_dir, 'Fold', f'Fold {f}')
            ifile = os.path.join(bdir, f'images/fold{f}/images.npy')
            tfile = os.path.join(bdir, f'images/fold{f}/types.npy')
            mfile = os.path.join(bdir, f'masks/fold{f}/masks.npy')

            T = np.load(tfile)
            # print(np.unique(T))
            M = np.load(mfile)
            I = np.load(ifile)
            if select:
                M = M[T==self.focus]
                I = I[T==self.focus]
            return M, I
        
        def processI(I, M, stage):

            os.makedirs(f'{self.pannuke_dir}/split/PanNuke_{self.focus}/{stage}/Images/')
            os.makedirs(f'{self.pannuke_dir}/split/PanNuke_{self.focus}/{stage}/Labels/')
            os.makedirs(f'{self.pannuke_dir}/split/PanNuke_{self.focus}/{stage}/Overlay/')

            for i in range(I.shape[0]):
                img = I[i,:,:]
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                train_I = f'train_{i}.png'
                cv2.imwrite(f'{self.pannuke_dir}/split/PanNuke_{self.focus}/{stage}/Images/'+train_I, img)
                # convert inst and type format for mask
                msk = M[i]
                inst = np.zeros((256,256))
                ann = np.zeros((256,256))
                for j in range(5):
                    #copy value from new array if value is not equal 0
                    inst = np.where(msk[:,:,j] != 0, msk[:,:,j], inst)
                    ann[np.where(msk[:,:,j] != 0)] = j+1
                    
                map_inst(inst)
                
                train_M = f'train_{i}.mat'
                label = sio.savemat(f'{self.pannuke_dir}/split/PanNuke_{self.focus}/{stage}/Labels/'+train_M, {'inst_map': inst, 'type_map': ann})

                overlay_img = myoverlay(img,msk)
                cv2.imwrite(f'{self.pannuke_dir}/split/PanNuke_{self.focus}/{stage}/Overlay/'+train_I, overlay_img)

        select = True if not self.focus == 'all' else False
        M1, I1 = load(f=1, select=select)
        M2, I2 = load(f=2, select=select)
        I = np.concatenate((I1, I2), axis=0)        
        M = np.concatenate((M1, M2), axis=0)
        processI(I, M, stage='Train')
        M, I = load(f=3, select=select)
        processI(I, M, stage='Validation')


    def prepare_input(self):

        def load_img(path):
            return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        def load_ann(path, with_type=True):
            # assumes that ann is HxW
            ann_inst = sio.loadmat(path)["inst_map"]
            if with_type:
                ann_type = sio.loadmat(path)["type_map"]

                ann = np.dstack([ann_inst, ann_type])
                ann = ann.astype("int32")
            else:
                ann = np.expand_dims(ann_inst, -1)
                ann = ann.astype("int32")
            return ann
        
        def rm_n_mkdir(dir_path):
            """Remove and make directory."""
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)


        dataset_info = {
            "train": {
                "img": (".png", f'{self.pannuke_dir}/split/PanNuke_{self.focus}/Train/Images/'),
                "ann": (".mat", f'{self.pannuke_dir}/split/PanNuke_{self.focus}/Train/Labels/'),
            },
            "valid": {
                "img": (".png", f'{self.pannuke_dir}/split/PanNuke_{self.focus}/Validation/Images/'),
                "ann": (".mat", f'{self.pannuke_dir}/split/PanNuke_{self.focus}/Validation/Labels/'),
            },
        }

        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        for split_name, split_desc in dataset_info.items():
            img_ext, img_dir = split_desc["img"]
            ann_ext, ann_dir = split_desc["ann"]

            out_dir = "%s/%s/%s/%dx%d" % (
                self.save_root,
                self.focus,
                split_name,
                self.win_size[0],
                self.win_size[1],
                # step_size[0],
                # step_size[1],
            )
            file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
            file_list.sort()  # ensure same ordering across platform

            rm_n_mkdir(out_dir)

            for file_path in file_list:
                base_name = pathlib.Path(file_path).stem

                img = load_img("%s/%s%s" % (img_dir, base_name, img_ext))
                ann = load_ann("%s/%s%s" % (ann_dir, base_name, ann_ext))

                img = np.concatenate([img, ann], axis=-1)
                np.save("{0}/{1}.npy".format(out_dir, base_name), img)

    def control_opt(self, hover_net_dir = 'hover_net',):
        pre_weight_dir = "./pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar"
        opt_dir = os.path.join(hover_net_dir, "models/hovernet/opt.py")
        with open(opt_dir, 'r') as file:
            content = file.readlines()
        content[54] = f'                        "pretrained": \"{pre_weight_dir}\",\n'
        with open(opt_dir, 'w') as file:
            file.writelines(content)

    def control_config(self, hover_net_dir = 'hover_net',):   
        config_dir = os.path.join(hover_net_dir, "config.py")

        train_dir = "%s/%s/%s/%dx%d" % (
            self.save_root,
            self.focus,
            'train',
            self.win_size[0],
            self.win_size[1],
        )
        valid_dir = "%s/%s/%s/%dx%d" % (
            self.save_root,
            self.focus,
            'valid',
            self.win_size[0],
            self.win_size[1],
        )
        with open(config_dir, 'r') as file:
            content = file.readlines()
        content[21] = '        model_mode = "fast" # choose either `original` or `fast`\n'
        content[26] = '        nr_type = 6 # number of nuclear types (including background)\n'
        content[36] = '        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation\n'
        content[37] = '        out_shape = [164, 164] # patch shape at output of network\n'
        content[47] = f'        self.log_dir = "logs/{self.focus}" # where checkpoints will be saved\n'
        content[51] = f'            \"../{train_dir}\"\n'
        content[54] = f'            \"../{valid_dir}\"\n'
        with open(config_dir, 'w') as file:
            file.writelines(content)

    def run_train(self):
        warnings.warn('Before training, User shoud check the config in config.py and replace the dir of pretrain weights in models/hovernet/opt.py', UserWarning)
        warnings.warn('This step will take several hours, if you want to watch the training process in real time, check out the hover-net github page.', UserWarning)
        conda_env = 'hovernet'  
        python_command = 'python'  

        command_arguments = ['run_train.py']
        working_directory = os.path.join(os.getcwd(), 'hover_net')

        cwd = os.getcwd()
        full_command = [f'conda', 'run', '-n', conda_env, python_command] + command_arguments
        
        subprocess.run(full_command, cwd=working_directory)


if __name__ == "__main__":

    # ['Adrenal_gland','Bile-duct','Bladder','Breast','Cervix','Colon','Esophagus','HeadNeck',
    # 'Kidney','Liver','Lung','Ovarian','Pancreatic','Prostate','Skin','Stomach','Testis','Thyroid','Uterus']
    
    trainer = train_hovernet(pannuke_dir = 'PanNuke', focus='all')
    trainer.download_pannuke()
    trainer.split_pannuke()
    trainer.prepare_input()
    trainer.control_opt(hover_net_dir = 'hover_net')
    trainer.control_config(hover_net_dir = 'hover_net')
    trainer.run_train()
