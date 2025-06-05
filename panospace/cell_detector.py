import os
import shutil
import numpy as np
import anndata as ad

import warnings
import subprocess

import cv2
from PIL import Image

from .utils import *

Image.MAX_IMAGE_PIXELS = 1576193600
os.environ["MKL_THREADING_LAYER"] = "GNU"  # or "INTEL"
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

class celldetector(object):
    def __init__(self,
                 img_dir,
                 tissue_name,
                 small_image_size,
                 hover_net_dir = 'hover_net',
                 resize=None):
        
        self.tissue_name = tissue_name
        self.work_dir = os.path.join('dataset',self.tissue_name)
        self.infer_dir = os.path.join(self.work_dir,'imgs')
        self.pred_dir = os.path.join(self.work_dir,'pred')
        self.out_dir = os.path.join(self.pred_dir, 'out')

        self.hover_net_dir = hover_net_dir
        self.image = Image.open(img_dir)

        self.small_image_size = small_image_size
        self.resize = resize
        
    def split_img(self, cvt=True, hue=None):

        # Clean up and recreate directories
        for directory in [self.infer_dir, self.pred_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        def change_hue(image, delta_hue):
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            hue_channel = hsv_image[:, :, 0]
            hue_channel = (hue_channel + delta_hue) % 180
            hsv_image[:, :, 0] = hue_channel
            modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            return modified_image

        # Get the dimensions of the large image
        width, height = self.image.size

        # Calculate the number of small images needed
        num_small_images_x = (width + self.small_image_size[0] - 1) // self.small_image_size[0]
        num_small_images_y = (height + self.small_image_size[1] - 1) // self.small_image_size[1]
        if self.resize is not None and isinstance(self.resize, (int, float)):
            new_size = (self.small_image_size[0]*self.resize, self.small_image_size[1]*self.resize)

        # Split the large image into small images
        small_images = {}
        for i in range(num_small_images_x):
            for j in range(num_small_images_y):
                box = (i * self.small_image_size[0], j * self.small_image_size[1],
                    (i + 1) * self.small_image_size[0], (j + 1) * self.small_image_size[1])
                im = self.image.crop(box)
                if self.resize is not None and isinstance(self.resize, (int, float)):
                    im = im.resize(new_size, Image.LANCZOS)
                small_images[str(i)+'_'+str(j)] = im

        for k, v in small_images.items():
            im = np.array(v)
            if cvt:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if isinstance(hue, (int, float)):
                im = change_hue(im, hue)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            _ = cv2.imwrite(os.path.join(self.infer_dir, f'{k}.png'), im)
            
    def run_infer(self, weight_dir=None):
        conda_env = 'hovernet'

        sh_dir = os.path.join(self.hover_net_dir, 'run_tile.sh')
        model_path = weight_dir or f'logs/{self.tissue_name}/01/net_epoch=50.tar'

        with open(sh_dir, 'r') as file:
            content = file.readlines()
        content[1] = "--gpu='0' \\\n"
        content[6] = f'--model_path={model_path} \\\n'
        content[10] = f'--input_dir=../{self.infer_dir} \\\n'
        content[11] = f'--output_dir=../{self.pred_dir} \\\n'
        content[12] = '--mem_usage=0.7 \\\n'

        with open(sh_dir, 'w') as file:
            file.writelines(content)

        command_arguments = ['./run_tile.sh']
        working_directory = os.path.join(os.getcwd(), 'hover_net')
        full_command = [f'conda', 'run', '-n', conda_env] + command_arguments

        warnings.warn('Before infering, User shoud check the config in run_tile.sh', UserWarning)
        warnings.warn('This step will take a moment, if you want to watch the intering process in real time, check out the hover-net github page.', UserWarning)
        subprocess.run(['chmod', '+x', 'run_tile.sh'], cwd=working_directory)
        subprocess.run(full_command, cwd=working_directory)

    def merge_img(self):
        overlay_dir = os.path.join(self.pred_dir, 'overlay')
        json_dir = os.path.join(self.pred_dir, 'json')
        
        os.makedirs(self.out_dir, exist_ok=True)

        overlay_file = os.listdir(overlay_dir)
        overlay_file.sort()
        width, height = 0, 0
        prev_x, prev_y = [], []
        small_images = {}
        for file_name in overlay_file:
            n, _ = os.path.splitext(file_name)
            im = Image.open(os.path.join(overlay_dir, file_name))
            if self.resize is not None and isinstance(self.resize, (int, float)):
                im = im.resize((im.size[0]//self.resize,im.size[1]//self.resize))
            small_images[n] = im
            x, y = n.split('_')
            if int(x) not in prev_x:
                width += small_images[n].size[0]
                prev_x.append(int(x))
            if int(y) not in prev_y:
                height += small_images[n].size[1]
                prev_y.append(int(y))

        synthesized_image = Image.new('RGB', (width, height))
        for k, v in small_images.items():
            i, j = k.split('_')
            small_image_size = v.size
            x, y = int(i) * small_image_size[0], int(j) * small_image_size[1]
            synthesized_image.paste(v, (x, y))
        
        synthesized_image = synthesized_image.crop(self.image.getbbox())
        synthesized_image.save(self.out_dir+'/pred_overlay.png', format="PNG")

        json_file = os.listdir(json_dir)
        json_file.sort()

        centroid_dict = {}
        contour_dict = {}
        bbox_dict = {}
        type_dict = {}
        for f in json_file:
            n, _ = os.path.splitext(f)
            bbox_list, centroid_list, contour_list, type_list = process_json(json_dir+'/'+f)
            centroid_dict[n] = centroid_list
            contour_dict[n] = contour_list
            bbox_dict[n] = bbox_list
            type_dict[n] = type_list

        stacked_centers = np.array([], dtype=int).reshape(0, 2)
        for keys, values in centroid_dict.items():
            if len(values) == 0:
                continue
            x, y = keys.split('_')
            if self.resize is not None and isinstance(self.resize, (int, float)):
                centers = np.array(values, dtype=int)/self.resize
            else:
                centers = np.array(values, dtype=int)
            centers += (int(x)*small_image_size[0], int(y)*small_image_size[1])
            stacked_centers = np.vstack((stacked_centers, centers))
        centroid_list = stacked_centers.tolist()
        
        contour_list = []
        for keys, values in contour_dict.items():
            if len(values) == 0:
                continue
            x, y = keys.split('_')
            for point_contour in values:
                if self.resize is not None and isinstance(self.resize, (int, float)):
                    point_contour = np.array(point_contour, dtype=int)/self.resize
                else:
                    point_contour = np.array(point_contour, dtype=int)
                point_contour = np.array(point_contour, dtype=int)
                point_contour += (int(x)*small_image_size[0], int(y)*small_image_size[1])
                point_contour = point_contour.tolist()
                contour_list.append(point_contour)
        
        type_list = sum(type_dict.values(), [])

        new_dict = {}
        if len(stacked_centers) == len(contour_list) and len(contour_list) == len(type_list):
            for idx, (val1, val2, val3) in enumerate(zip(centroid_list, contour_list, type_list), 1):
                new_dict[str(idx)] = {
                    'centroid':val1,
                    'contour':val2,
                    'bbox':[],
                    'type':val3
                }

        whole_dict = {
            'mag':None,
            'nuc':new_dict
        }
        
        with open(os.path.join(self.out_dir, 'whole.json'), 'w') as f:
            json.dump(whole_dict, f)

    def make_nuclei_adata(self):

        _, centroid_list, contour_list, type_list = process_json(self.out_dir+'/whole.json')

        nuclei_centers = np.array(centroid_list, dtype=int)
        nuclei_type = np.array(type_list, dtype=int)
        new_contour_list = [[[round(values) for values in sublist] for sublist in original_list] for original_list in contour_list]

        img_adata = ad.AnnData(np.zeros((nuclei_centers.shape[0],1)), dtype='float32')
        img_adata.obsm['spatial'] = nuclei_centers
        img_adata.obs['img_type'] = nuclei_type

        label_dict = {
            0 : 'nolabe', 
            1 : 'Neoplastic cells', 
            2 : 'Inflammatory', 
            3 : 'Connective/Soft tissue cells', 
            4 : 'Dead Cells', 
            5 : 'Epithelial' 
        }

        remove_list = ['nolabe', 'Dead Cells']
        keep_indices = ~img_adata.obs['img_type'].isin([key for key, label in label_dict.items() if label in remove_list])

        img_adata = img_adata[keep_indices]
        new_contour_list = [contour for contour, keep in zip(new_contour_list, keep_indices) if keep]

        with open(os.path.join(self.work_dir, 'new_contour_list.pkl'), 'w') as f:
            json.dump(new_contour_list, f)

        os.makedirs(os.path.join(self.work_dir, 'adata'), exist_ok=True)
        img_adata.write(os.path.join(self.work_dir, 'adata', 'img_adata_sc.h5ad'))

        print({label_dict[key]: (grp.shape[0], grp.shape[0] / img_adata.shape[0]) for key, grp in img_adata.obs.groupby('img_type')})


def process_json_from_cellvit(json_dir):
    """
    load the json file and add the contents to corresponding lists.
    Args:
        json_dir (str): Directory containing the JSON file with nuclei instance information fetched from CellViT.

    Returns:
        Three lists containing information about each nuclei instance: centroids, contours, and types.
    """
    centroid_list = []
    contour_list = [] 
    type_list = []

    with open(json_dir) as json_f:
        data = json.load(json_f)

        for inst in data:
            offset = inst['offset_global']

            inst_centroid = np.array(inst['centroid'])
            centroid_list.append(inst_centroid)

            inst_contour = np.array(inst['contour'])
            contour_list.append(inst_contour)

            inst_type = inst['type']
            type_list.append(inst_type) 
             
    return centroid_list, contour_list, type_list



def process_json_from_hovernet(json_dir):
    """
    load the json file and add the contents to corresponding lists.
    Args:
        json_dir (str): Directory containing the JSON file with nuclei instance information fetched from hover-net.

    Returns:
        Four lists containing information about each nuclei instance: bounding boxes, centroids, contours, and types.
    """
    bbox_list = []
    centroid_list = []
    contour_list = [] 
    type_list = []

    with open(json_dir) as json_f:
        data = json.load(json_f)
        # mag_info = data['mag']
        nuc_info = data['nuc']

        for inst in nuc_info:
            inst_info = nuc_info[inst]
            inst_centroid = inst_info['centroid']
            centroid_list.append(inst_centroid)

            inst_contour = inst_info['contour']
            contour_list.append(inst_contour)

            inst_bbox = inst_info['bbox']
            bbox_list.append(inst_bbox)

            inst_type = inst_info['type']
            type_list.append(inst_type) 
             
    return bbox_list, centroid_list, contour_list, type_list


if __name__ == "__main__":
    seg = celldetector(
        img_dir='dataset/Xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image_registered.png',
        tissue_name='Xenium',
        hover_net_dir='hover_net',
        small_image_size=(5000, 5000),
        resize=None
    )

    seg.split_img(cvt=False, hue=None)
    seg.run_infer(weight_dir='logs/Breast/01/net_epoch=50.tar')
    seg.merge_img()
    seg.make_nuclei_adata()
