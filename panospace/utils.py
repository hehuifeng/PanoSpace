import numpy as np
import sys
import json
import logging
import multiprocessing
from functools import partial

def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger


def process_json(json_dir):
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


def get_contain_mat(dict, arg):
    which_spot = list((arg[0] < dict['spot_x_max']) &
                      (arg[0] > dict['spot_x_min']) &
                      (arg[1] < dict['spot_y_max']) &
                      (arg[1] > dict['spot_y_min']))
    return which_spot

def if_contain_(spot, subspot, r=56, norm = True):
    spot_x_max = spot[:, 0] + r
    spot_x_min = spot[:, 0] - r
    spot_y_max = spot[:, 1] + r
    spot_y_min = spot[:, 1] - r

    dict = {'spot_x_max':spot_x_max, 'spot_x_min':spot_x_min, 'spot_y_max':spot_y_max, 'spot_y_min':spot_y_min}
    pfunc = partial(get_contain_mat, dict)
    pool = multiprocessing.Pool(multiprocessing.cpu_count(), maxtasksperchild=1)
    contain_mat = pool.map(pfunc, [(x,y) for x,y in subspot[:, :]])
    pool.close()
    pool.join()
    contain_mat = np.array(contain_mat)
    if norm:
        contain_mat = (contain_mat.T / np.array([i if i != 0 else 1 for i in np.sum(contain_mat, axis=1)])).T

    return contain_mat


def if_contain(spot, subspot, r=56, norm=True):
    dist_mat = np.sqrt((subspot[:,0][:, np.newaxis] - spot[:,0])**2 + (subspot[:,1][:, np.newaxis] - spot[:,1])**2)
    
    contain_mat = dist_mat <= r

    if norm:
        row_sums = contain_mat.sum(axis=1)
        contain_mat = np.array([row / sum if sum != 0 else row * 0 for row, sum in zip(contain_mat, row_sums)])

    return contain_mat


def if_contain_batch(spot, subspot, r=56, norm=True, batch_size=1000):
    r_squared = r ** 2
    total_spots = spot.shape[0]
    contain_mat_list = []
    
    for i in range(0, total_spots, batch_size):
        spot_batch = spot[i:min(i + batch_size, total_spots)]
        
        dist_mat_squared = (subspot[:, 0][:, np.newaxis] - spot_batch[:, 0]) ** 2 + (subspot[:, 1][:, np.newaxis] - spot_batch[:, 1]) ** 2
        
        contain_mat_batch = dist_mat_squared <= r_squared

        if norm:
            row_sums = contain_mat_batch.sum(axis=1, keepdims=True)
            
            np.divide(contain_mat_batch, row_sums, out=contain_mat_batch, where=row_sums != 0)

        contain_mat_list.append(contain_mat_batch.T)

    contain_mat = np.concatenate(contain_mat_list, axis=0)
    
    return contain_mat.T