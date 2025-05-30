import os
import ot
import numpy as np
import pandas as pd

import scipy

import gurobipy as gp
from gurobipy import GRB


import matplotlib.pyplot as plt
from utils import *
from superres_deconv import superres_deconv

class CellTypeAnnotator(object):
    def __init__(self, experimental_path, img_dir, num_classes, deconv_adata, sr_deconv_adata, segment_adata, priori_type_affinities=None, alpha=0.3):
        """
        Initialize the CellTypeAnnotator class.

        Parameters:
        - experimental_path: Path to the experimental data.
        - img_dir: Directory containing the images.
        - num_classes: Number of cell types/classes.
        - deconv_adata: AnnData object containing deconvolution data.
        - segment_adata: AnnData object containing segmentation data.
        - priori_type_affinities: Optional dictionary containing prior type affinities.
        - alpha: Regularization parameter alpha.
        """
        self.experimental_path = experimental_path
        self.output_dir = os.path.join(experimental_path, 'celltype_infer')
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.deconv_adata = deconv_adata
        self.sr_deconv_adata = sr_deconv_adata
        self.segment_adata = segment_adata
        self.alpha = alpha
        self.mode = 'mor' if 'img_type' in self.segment_adata.obs.columns else None

        self.priori_type_affinities = priori_type_affinities
        
        # Setup prior type affinities and cell types
        self.img_types = list(priori_type_affinities.keys()) if priori_type_affinities else None
        self.cell_types = deconv_adata.uns['celltype'].tolist()
        self.sr_celltype_ratios = self.sr_deconv_adata.obs[self.cell_types].values
        self.celltype_ratios = self._normalize_celltype_ratios()

        # Create output directories if they don't exist
        self._create_output_dirs()

    def _normalize_celltype_ratios(self):
        """
        Normalize cell type ratios in deconv_adata.

        Returns:
        - Normalized cell type ratios.
        """
        celltype_ratios = self.deconv_adata.obs[self.cell_types].values
        celltype_ratios[celltype_ratios < 0] = 0
        return (celltype_ratios.T / np.sum(celltype_ratios, axis=1)).T

    def _create_output_dirs(self):
        """Create necessary output directories."""
        os.makedirs(os.path.join(self.output_dir, 'af'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'fig'), exist_ok=True)

    def filter_segmentation(self):
        """
        Filter segmentation data based on spatial information.
        """
        affiliation_matrix = if_contain_batch(
            self.sr_deconv_adata.obsm['spatial'], 
            self.segment_adata.obsm['spatial'], 
            r=self.deconv_adata.uns['radius'], 
            norm=False
        )
        self.segment_adata = self.segment_adata[np.sum(affiliation_matrix, axis=1) != 0]
        if self.mode == 'mor':
            self._annotate_nuclei_types()

        norm_affiliation_matrix = affiliation_matrix[np.sum(affiliation_matrix, axis=1) != 0,:]
        row_sums = norm_affiliation_matrix.sum(axis=1, keepdims=True)
        norm_affiliation_matrix = scipy.sparse.csr_matrix(norm_affiliation_matrix)  
        self.norm_affiliation_matrix = norm_affiliation_matrix / row_sums

    def _annotate_nuclei_types(self):
        """
        Annotate nuclei types based on pre-defined labels.
        """
        nuclei_type = self.segment_adata.obs['img_type'].copy()
        label_dict = {
            0: 'nolabel',
            1: 'Neoplastic cells',
            2: 'Inflammatory',
            3: 'Connective/Soft tissue cells',
            4: 'Dead Cells',
            5: 'Epithelial'
        }
        remove_list = ['nolabel', 'Dead Cells']

        nuclei_type.replace(label_dict, inplace=True)
        valid_cell_types = [label for label in label_dict.values() if label not in remove_list]
        nuclei_type.replace(valid_cell_types, range(len(valid_cell_types)), inplace=True)

        num_unique_values = len(np.unique(nuclei_type))
        self.onehot_mortype = np.eye(num_unique_values)[nuclei_type.values.astype(int)]

    def calculate_cell_count(self):
        """
        Calculate the number of cells per spot.
        """
        spot_cell_affiliation = if_contain(
            self.deconv_adata.obsm['spatial'], 
            self.segment_adata.obsm['spatial'], 
            r=self.deconv_adata.uns['radius'], 
            norm=False
        )

        self.cell_counts = np.sum(spot_cell_affiliation, axis=0)
        self.deconv_adata.obs['cell_count'] = self.cell_counts
        self.spot_cell_affiliation = scipy.sparse.csr_matrix(spot_cell_affiliation)

    def calculate_imgtype_ratio(self):
        """
        Calculate the imgtype_ratio parameter for each spot.
        """
        counts_list = []
        raw_data = self.segment_adata.obs.copy()
        rows, cols, _ = scipy.sparse.find(self.spot_cell_affiliation)

        for col in range(self.spot_cell_affiliation.shape[1]):
            cell_indices = rows[cols == col]
            counts_list.append(raw_data.iloc[cell_indices, :]['img_type'].value_counts())

        simulate_prop = pd.concat(counts_list, axis=1).fillna(0)
        self.mortype_in_spot = simulate_prop.T[[1, 2, 3, 5]].T
        self.imgtype_ratio = self.mortype_in_spot.sum(axis=1).values / np.sum(self.mortype_in_spot.values)

    def calculate_celltype_ratio(self):
        """
        Calculate the celltype_ratio parameter for cell type proportions.
        """
        int_cell_type_ratios = np.zeros(self.celltype_ratios.shape)
        for spot in range(self.cell_counts.shape[0]):
            int_cell_type_ratios[spot, :] = self._solve_integer_vector(self.celltype_ratios[spot, :], self.cell_counts[spot])

        if not (np.sum(int_cell_type_ratios, axis=1).astype(int) == self.cell_counts).all():
            raise ValueError("Inconsistent cell counts")

        self.cell_count = self.cell_counts[self.cell_counts != 0]
        self.int_cell_type_ratios = int_cell_type_ratios[self.cell_counts != 0, :]
        self.spot_cell_affiliation = self.spot_cell_affiliation[:, self.cell_counts != 0]

        self.celltype_ratio = np.sum(self.int_cell_type_ratios, axis=0) / np.sum(self.int_cell_type_ratios)
        self.N = np.array(self._solve_integer_vector(self.celltype_ratio, self.segment_adata.shape[0])).astype(int)

    def calculate_type_transfer_matrix(self, factor=2):
        """
        Calculate the type transfer matrix using optimal transport.

        Parameters:
        - factor: Adjustment factor for the cost matrix.
        """
        self.cost_matrix = ot.dist(self.int_cell_type_ratios.T, self.mortype_in_spot.values[:, self.cell_counts != 0], metric='cosine')

        if self.priori_type_affinities:
            self.adjusted_cost_matrix = self.cost_matrix.copy()
            for i, img_type in enumerate(self.img_types):
                for cell_type in self.priori_type_affinities[img_type]:
                    idx = self.cell_types.index(cell_type)
                    self.adjusted_cost_matrix[idx, i] /= factor
            self.type_transfer_ot = ot.emd(self.celltype_ratio, self.imgtype_ratio, self.adjusted_cost_matrix, numItermax=1000)
        else:
            self.type_transfer_ot = ot.emd(self.celltype_ratio, self.imgtype_ratio, self.cost_matrix, numItermax=1000)

        self.type_transfer_prop = self.type_transfer_ot / self.type_transfer_ot.sum(axis=0)

    def infer_cell_types(self):
        """
        Infer cell types using integer programming.

        Returns:
        - Annotated segment AnnData object.
        """
        if self.mode == 'mor':
            to_solve = ((1 - self.alpha) * self.norm_affiliation_matrix @ self.sr_celltype_ratios
                        + self.alpha * self.onehot_mortype @ self.type_transfer_prop.T)
        else:
            to_solve = self.norm_affiliation_matrix @ self.sr_celltype_ratios
        
        optimal_solution = self._integer_programming_solver(to_solve, self.N, self.spot_cell_affiliation, self.int_cell_type_ratios)
        max_indexes = np.argmax(optimal_solution, axis=1)
        
        self.segment_cp = self.segment_adata.copy()
        self.segment_cp.obs['pred_cell_type'] = [self.cell_types[i] for i in max_indexes]
        self.segment_cp.obs[self.cell_types] = np.where(optimal_solution != 0, 1, 0)
        
        output_path = os.path.join(self.output_dir, 'results', f'adata_{self.alpha}.h5ad')
        self.segment_cp.write(output_path)

        return self.segment_cp

    def _solve_integer_vector(self, vector, n_k):
        """
        Solve an integer programming problem to match a vector to integer counts.

        Parameters:
        - vector: Array of proportions.
        - n_k: Target count.

        Returns:
        - Optimized integer vector.
        """
        vector = vector / np.sum(vector) * n_k
        n = len(vector)

        model = gp.Model("Integer_Vector_Problem")
        model.setParam("OutputFlag", 0)

        integer_vars = model.addVars(n, vtype=GRB.INTEGER, name="x")
        model.addConstr(gp.quicksum(integer_vars[i] for i in range(n)) == round(np.sum(vector)), "sum_constraint")
        objective = gp.quicksum((integer_vars[i] - vector[i])**2 for i in range(n))

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        if model.status in [GRB.OPTIMAL]:
            return [int(integer_vars[i].X) for i in range(n)]
        else:
            raise ValueError("Integer programming problem could not be solved")

    def _integer_programming_solver(self, matrix, N, spa_Q, V):
        """
        Solve a binary integer programming problem for cell type inference.

        Parameters:
        - matrix: Cost matrix.
        - N: Array of integers representing cell counts.
        - spa_Q: Sparse matrix representing spatial affiliations.
        - V: Array representing integer cell type ratios.

        Returns:
        - Optimized matrix.
        """
        n, m = matrix.shape

        try:
            model = gp.Model("0-1_Programming_Problem")
            model.setParam("OutputFlag", 0)

            X = {(i, k): model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{k}") for i in range(n) for k in range(m)}
            model.setObjective(gp.quicksum(matrix[i, k] * X[(i, k)] for i in range(n) for k in range(m)), GRB.MAXIMIZE)

            for i in range(n):
                model.addConstr(gp.quicksum(X[(i, k)] for k in range(m)) == 1)

            for k in range(m):
                model.addConstr(gp.quicksum(X[(i, k)] for i in range(n)) == int(N[k]))

            rows, cols, _ = scipy.sparse.find(spa_Q)
            for col in range(spa_Q.shape[1]):
                cell_indices = rows[cols == col]
                for k in range(m):
                    model.addConstr(gp.quicksum(X[(i, k)] for i in cell_indices) == V[col, k])

            model.optimize()

            return np.array([[X[(i, k)].X for k in range(m)] for i in range(n)])

        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return None
        except AttributeError as e:
            print(f"Attribute Error: {e}")
            return None
