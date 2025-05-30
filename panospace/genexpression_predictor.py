import numpy as np
import scanpy as sc
from tqdm import tqdm
from utils import if_contain

import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, diags, vstack
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import anndata as ad
import torch

import matplotlib.pyplot as plt

class GeneExpPredictor(object):
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
        self.cell_type_means = self.cell_type_means.loc[celltype_list]

        mu = self.cell_type_means.values
        sum_term = beta.T @ mu
        numerator = (beta[:, :, np.newaxis] * mu[:, np.newaxis, :] * Y[np.newaxis, :, :])  # (k,i,j)
        result = numerator / sum_term[np.newaxis, :, :]
        result = np.nan_to_num(result)

        row_sum = result.sum(axis=2, keepdims=True) + 1e-3
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
    
    def construct_graph(self, coords, graph_mode='delaunay', weight_mode='inverse', k=10, sigma=50.0):
        if graph_mode == 'delaunay':
            tri = Delaunay(coords)
            edges = {(p1, p2, np.linalg.norm(coords[p1] - coords[p2]))
                     for simplex in tri.simplices for i in range(3) for j in range(i + 1, 3)
                     for p1, p2 in [(simplex[i], simplex[j])]}
        elif graph_mode == 'knn':
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            edges = set()
            for i in range(len(coords)):
                for j in range(1, k+1):  # skip self
                    p1, p2 = i, indices[i, j]
                    dist = distances[i, j]
                    edges.add((p1, p2, dist))
        else:
            raise ValueError("Unsupported graph_mode. Use 'delaunay' or 'knn'.")

        rows, cols, data = [], [], []
        node_edge_weights = defaultdict(list)
        for u, v, d in edges:
            if weight_mode == 'inverse':
                w = 1.0 / (d + 1e-6)
            elif weight_mode == 'gaussian':
                w = np.exp(- (d ** 2) / (2 * sigma ** 2))
            else:
                raise ValueError("Unsupported weight_mode. Use 'inverse' or 'gaussian'.")
            rows += [u, v]
            cols += [v, u]
            data += [w, w]
            node_edge_weights[u].append(w)
            node_edge_weights[v].append(w)

        n_nodes = coords.shape[0]
        for i in range(n_nodes):
            w = np.mean(node_edge_weights[i]) if node_edge_weights[i] else 1.0
            rows.append(i)
            cols.append(i)
            data.append(w) 
        W = sp.csc_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        return W

    def do_geneinfer(self, gamma=0.1, graph_mode='delaunay', weight_mode='inverse', k=10, sigma=50.0, return_w=False, iterations=10, return_ada=False):
        ada = []
        cell_type = list(self.cell_type_means.index)
        for ct_idx, ct in enumerate(cell_type):
            print(ct)
            ct_adata = self.infered_adata[self.infered_adata.obs['pred_cell_type'] == ct]
            if ct_adata.shape[0] == 0:
                print(f"[Warning] Skipping cell type '{ct}' because no cells found.")
                continue
            affi = if_contain(self.spot_adata.obsm['spatial'], ct_adata.obsm['spatial'], self.spot_adata.uns['radius'], norm=False)
            ind = np.sum(affi, 1)
            non_zero_index = np.where(ind != 0)[0]
            zero_index = np.where(ind == 0)[0]
            cell_sort_loc = ct_adata[np.concatenate((non_zero_index, zero_index))].obsm['spatial']
            obs_names = ct_adata[np.concatenate((non_zero_index, zero_index))].obs_names

            W = self.construct_graph(cell_sort_loc, graph_mode=graph_mode, weight_mode=weight_mode, k=k, sigma=sigma)

            degree_values = W.sum(axis=1).A.ravel()
            zero_degree_nodes = np.sum(degree_values == 0)
            if zero_degree_nodes > 0:
                print(f"[Warning] {zero_degree_nodes} nodes have zero degree (isolated), propagation may be inaccurate.")

            degree_values[degree_values == 0] = 1.0  # 避免除以 0
            degree_values = np.clip(degree_values, 1e-3, np.inf)

            # D = diags(degree_values)
            D_inv = diags(1.0 / degree_values)
            W = D_inv.dot(W)

            ct_spot = self.cell_type_specific_spot_exp[ct_idx,:,:]

            F_l_values = np.concatenate([ct_spot[np.where(affi[id,] != False)[0]] for id in non_zero_index])
            values = F_l_values[F_l_values != 0]
            row_indices, col_indices = np.nonzero(F_l_values)
            F_l = sp.csc_matrix((values, (row_indices, col_indices)), shape=F_l_values.shape)
            n_l = F_l.shape[0]
            Y_l = F_l.copy()
            # initYu = sp.csr_matrix(ct_spot.mean(0))
            # initYu = [initYu for _ in range(W.shape[0]-n_l)]
            # Y_u = sp.vstack(initYu)
            af = if_contain(self.spot_adata.obsm['spatial'], ct_adata[zero_index].obsm['spatial'], r=self.spot_adata.uns['radius']*4, norm=True)
            Y_u = af @ ct_spot


            gamma_param = 1/(1+gamma)

            # Early stopping
            early_stop = True
            tol = 1e-4
            patience = 5
            best_diff = np.inf
            wait = 0
            diff_list = []
            for iter in tqdm(range(iterations), desc="Iterations Progress"):
                Y_u_new = W[n_l:, :n_l] @ Y_l + W[n_l:, n_l:] @ Y_u
                Y_l_new = gamma_param * F_l + (1 - gamma_param) * (W[:n_l, :n_l] @ Y_l + W[:n_l, n_l:] @ Y_u)

                Y_u = np.asarray(Y_u)
                Y_u_new = np.asarray(Y_u_new)
                diff = ((Y_u_new - Y_u)**2).sum() / Y_u.shape[0]  

                if early_stop:
                    if diff < best_diff - tol:
                        best_diff = diff
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"[Info] Early stopped at iteration {iter}, diff={diff:.2e}")
                            break

                Y_u = Y_u_new
                Y_l = Y_l_new

                diff_list.append(diff)

            # plt.figure(figsize=(6, 4))
            # plt.plot(range(1, len(diff_list) + 1), diff_list, marker='o', linewidth=2)
            # plt.xlabel("Iteration")
            # plt.ylabel("Mean squared change in $Y_u$")
            # title = f"Convergence of Graph Propagation"
            # plt.title(title)
            # plt.grid(True)
            # plt.tight_layout()
            # plt.savefig(f'./plot/{ct}.png', dpi=300)
            # plt.close()

            Y = sp.vstack([Y_l,Y_u])
            Y = sp.csc_matrix(Y)
            nuclei = ad.AnnData(Y)
            nuclei.var_names = self.sc_adata.var_names
            nuclei.obsm['spatial'] = cell_sort_loc
            nuclei.obs_names = obs_names
            nuclei.obs['pred_cell_type'] = ct
            ada.append(nuclei)

        if return_ada:
            return ada
        adata = self.concat(ada)
        if return_w:
            return adata, W

        else:
            return adata
