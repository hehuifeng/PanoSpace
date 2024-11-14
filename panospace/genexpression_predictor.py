import numpy as np
from tqdm import tqdm
from .utils import if_contain

import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse import diags, vstack

import anndata as ad


class genexpredictor(object):
    def __init__(self, sc_adata, spot_adata, infered_adata):
        self.sc_adata = sc_adata
        self.infered_adata = infered_adata
        self.spot_adata = spot_adata

        self.spot_adata, self.sc_adata = self.Find_common_gene(self.spot_adata, self.sc_adata)

    def Find_common_gene(self, adata1, adata2):
        common_genes = adata1.var_names.intersection(adata2.var_names)
        adata1 = adata1[:,common_genes]
        adata2 = adata2[:,common_genes]
        return adata1, adata2

    def ctspecific_spot_gene_exp(self, celltype_list, celltype_column='celltype_major'):
        Y = self.spot_adata.X.toarray()
        beta = self.spot_adata.obs[celltype_list].values.T
        self.cell_type_means = self.sc_adata.to_df().groupby(self.sc_adata.obs[celltype_column]).mean()

        mu = self.cell_type_means.values
        sum_term = beta.T @ mu
        numerator = (beta[:, :, np.newaxis] * mu[:, np.newaxis, :] * Y[np.newaxis, :, :])  # (k,i,j)
        result = numerator / sum_term[np.newaxis, :, :]
        result = np.nan_to_num(result)

        row_sum = result.sum(axis=2, keepdims=True)
        normalized_tensor = result / row_sum
        normalized_tensor *= 1e4
        normalized_tensor = np.log1p(normalized_tensor)
        self.cell_type_specific_spot_exp = normalized_tensor

    def concat(self, ada):

        ada_x = [a.X for a in ada]
        ada_x = vstack(ada_x)
        adata = ad.AnnData(ada_x)
        ada_loc = [a.obsm['spatial'] for a in ada]
        loc = np.concatenate(ada_loc)
        adata.obsm['spatial'] = loc
        ada_obsnames = [list(a.obs_names) for a in ada]
        obsn = []
        for o in ada_obsnames:
            obsn.extend(o)
        adata.obs_names = obsn
        adata.var_names = ada[0].var_names
        ada_obs = [a.obs['pred_cell_type'].values for a in ada]
        ada_obs = np.concatenate(ada_obs)
        adata.obs['pred_cell_type'] = ada_obs
        
        return adata
    
    def do_geneinfer(self, lambda_param=0.1):
        ada = []
        cell_type = list(self.cell_type_means.index)
        for ct_idx, ct in enumerate(cell_type):

            print(ct)
            ct_adata = self.infered_adata[self.infered_adata.obs['pred_cell_type'] == ct]
            affi = if_contain(self.spot_adata.obsm['spatial'], ct_adata.obsm['spatial'], self.spot_adata.uns['radius'], norm=False)
            ind = np.sum(affi, 1)
            non_zero_index = np.where(ind != 0)[0]

            cell_sort_loc = ct_adata[np.concatenate((non_zero_index, np.where(ind == 0)[0]))].obsm['spatial']
            obs_names = ct_adata[np.concatenate((non_zero_index, np.where(ind == 0)[0]))].obs_names
            tri = Delaunay(cell_sort_loc)

            edges = {(p1, p2, np.linalg.norm(cell_sort_loc[p1] - cell_sort_loc[p2]))
                    for simplex in tri.simplices for i in range(3) for j in range(i + 1, 3)
                    for p1, p2 in [(simplex[i], simplex[j])]}

            rows, cols, data = zip(*[(u, v, weight) for u, v, weight in edges] + [(v, u, weight) for u, v, weight in edges])
            n_nodes = cell_sort_loc.shape[0]


            weight_matrix = sp.csc_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
            new_data = [1/d for d in weight_matrix.data]

            gaussian_matrix = sp.csc_matrix((new_data, (weight_matrix.nonzero())), shape=weight_matrix.shape, dtype=np.float32)

            D = diags(gaussian_matrix.sum(axis=1).A.ravel())
            L = D - gaussian_matrix

            ct_spot = self.cell_type_specific_spot_exp[ct_idx,:,:]

            D_inv = D.copy()
            D_inv.data = 1.0 / D_inv.data
            D_inv.data[D_inv.data == np.inf] = 0.0

            W = D_inv.dot(gaussian_matrix)
            n = W.shape[0]  

            F_l_values = np.concatenate([ct_spot[np.where(affi[id,] != False)[0]]
                                        for id in non_zero_index])

            values = F_l_values[F_l_values != 0]
            row_indices, col_indices = np.nonzero(F_l_values)

            F_l = sp.csc_matrix((values, (row_indices, col_indices)), shape=F_l_values.shape)
            del F_l_values

            n_l = F_l.shape[0]  # Number of labeled points

            Y_l = F_l.copy()
            initYu = sp.csr_matrix(ct_spot.mean(0))
            initYu = [initYu for _ in range(W.shape[0]-n_l)]
            Y_u = sp.vstack(initYu)
            del initYu

            for iteration in tqdm(range(10), desc="Iterations Progress"):
                Y_u = W[n_l:, :n_l] @ Y_l + W[n_l:, n_l:] @ Y_u
                Y_l = (1 - lambda_param) * Y_l + lambda_param * (W[:n_l, :n_l] @ Y_l + W[:n_l, n_l:] @ Y_u)

            Y = sp.vstack([Y_l,Y_u])
            Y = sp.csc_matrix(Y)

            nuclei = ad.AnnData(Y)
            nuclei.var_names = self.sc_adata.var_names

            nuclei.obsm['spatial'] = cell_sort_loc
            nuclei.obs_names = obs_names
            nuclei.obs['pred_cell_type'] = ct
            ada.append(nuclei)

        adata = self.concat(ada)

        return adata