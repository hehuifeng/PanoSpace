import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree

import cv2
import json
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, find
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

class microenvironment_analyzer(object):
    def __init__(self, genemap, Experimental_path):
        self.genemap = genemap
        self.genemap.obs_names = [str(i) for i in range(len(self.genemap.obs_names))]

        self.Experimental_path = Experimental_path

        self.cell_type = self.genemap.obs['pred_cell_type'].cat.categories

    def umap(self):
        sc.tl.pca(self.genemap)
        sc.pp.neighbors(self.genemap, n_neighbors=40)
        sc.tl.umap(self.genemap)


    def detect_heg(self, expressed_genes, threshold=3):
        expressed_genes = 100 * (np.exp(expressed_genes) - 1)
        expressed_genes = np.log1p(expressed_genes.mean(axis=1))
        expressed_genes = expressed_genes[expressed_genes >= threshold]
        return expressed_genes.index
    
    def filter_gene(self,threshold):
        exp = pd.DataFrame(self.genemap.X, index=self.genemap.obs_names, columns=self.genemap.var_names)
        genes = self.detect_heg(exp.T, threshold=threshold)

        self.genemap = self.genemap[:, genes]

    def detect_microenvironment(self, search_radius = 94):
        coordinates = self.genemap.obsm['spatial']
        kdtree = cKDTree(coordinates)

        envirfea = []
        for i,point in enumerate(coordinates):
            indices = kdtree.query_ball_point(point, search_radius)
            indices.remove(i)
            if len(indices)==0:
                envirfea.append(np.zeros(len(self.cell_type)))
            else:
                envirfea.append(np.sum(self.genemap[indices].obs[self.cell_type].values,axis = 0))
            
        self.envirfea = np.vstack(envirfea)

    def detect_env_gene(self, sender='CAFs', receiver='Cancer Epithelial', threshold=0.2):
        envirfea_rec = self.envirfea[self.genemap.obs['pred_cell_type']==receiver,:]
        self.genexp_rec = self.genemap[self.genemap.obs['pred_cell_type']==receiver,:]

        self.genexp_rec = self.genexp_rec[:,np.sum(self.genexp_rec.X,0)!=0]

        sender_proportion = envirfea_rec[:,self.cell_type.get_loc(sender)]
        self.genexp_rec.obs['microenv_of_sender'] = sender_proportion

        exp = pd.DataFrame(self.genexp_rec.X, columns=self.genexp_rec.var_names, index=self.genexp_rec.obs_names)
        test_genes = exp.columns
        results = []
        for gene in test_genes:
            correlation, p_value = pearsonr(exp[gene].values, sender_proportion)
            results.append([correlation, p_value])
        results_df = pd.DataFrame(results, index=test_genes, columns=['correlation', 'p_value'])
        results_df['p_adjust'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        self.results_df = results_df.sort_values(by='correlation', ascending=False)

        self.detected_gene = results_df[(results_df['correlation']>threshold)&(results_df['p_adjust']<0.01)]

    def plot_rank_order(self):
        self.plot_df = self.results_df.reset_index().rename(columns={'index': 'gene'})
        self.plot_df['-log10(p_adjust)'] = -np.log10(self.plot_df['p_adjust'])
        max_non_inf = self.plot_df[self.plot_df['-log10(p_adjust)'] != np.inf]['-log10(p_adjust)'].max()
        replacement_value = max_non_inf
        self.plot_df['-log10(p_adjust)'].replace(np.inf, replacement_value, inplace=True)

    def load_contour_list(self):
        with open(f'{self.Experimental_path}/new_contour_list.pkl') as f:
            self.new_contour_list = json.load(f)

    def prepare_plot_ligrec(self,genemap,sender,receiver,ligand,receptor, plot_num=20000):
        se = genemap[genemap.obs['pred_cell_type']==sender,ligand]
        re = genemap[genemap.obs['pred_cell_type']==receiver,receptor]

        self.se_coor = se.obsm['spatial']
        self.re_coor = re.obsm['spatial']

        distances = cdist(self.se_coor, self.re_coor, metric='euclidean')
        distances = 1 / distances

        activ = np.outer(se.X, re.X)
        activ_by_dist = distances * activ
        threshold = np.partition(activ_by_dist.flatten(), -plot_num)[-plot_num]
        activ_by_dist = np.where(activ_by_dist > threshold, activ_by_dist, 0)
        activ_by_dist = csr_matrix(activ_by_dist)

        self.rows, self.cols, values = find(activ_by_dist)
        values *= (200 / values.max())
        values = np.log1p(values)
        self.values = np.round(values).astype(int)
        print((self.values>0).sum())
    
    def plot_ligrec(self,genemap,img,img_adata):
        def plot_contour(img, type_list, contours, type_info_dict):
            for tp, clr in type_info_dict.items():
                index = [i for i, t in enumerate(type_list) if t==tp]
                this_contours = [contours[i] for i in index]
                this_contours = [np.array(c) for c in this_contours]
                img = cv2.drawContours(img, this_contours, -1, clr, -1)
            return img
        img = np.full(img.shape, 255, dtype=np.uint8)

        for row, col, value in zip(self.rows, self.cols, self.values):
            if value == 0:
                continue
            else:
                _ = cv2.line(img, tuple(self.se_coor[row]), tuple(self.re_coor[col]), (0,0,0), value)

        new_type_list = genemap.obs['pred_cell_type'].to_list()

        idx = img_adata.obs_names.get_indexer(genemap.obs_names)
        pred_contour = [self.new_contour_list[i] for i in idx]
        type_list = self.cell_type
        tab10 = plt.cm.get_cmap('tab10', 10) 
        color_dict = {}
        for i, ct in enumerate(self.cell_type):
            color = (int(tab10(i)[0] * 255), int(tab10(i)[1] * 255), int(tab10(i)[2] * 255))
            color_dict[ct] = color

        img = plot_contour(img, new_type_list, pred_contour, color_dict)
        # for center in spot.obsm['spatial']:
        #     _ = cv2.circle(img, center, 94, (190,190,190), 2)
        return img