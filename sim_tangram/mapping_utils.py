"""
    Mapping helpers
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import logging
from . import mapping_optimizer_CCA as mo

# from . import utils as ut

# from torch.nn.functional import cosine_similarity

logging.getLogger().setLevel(logging.INFO)


def map_cells_to_space(
    single_cell,
    spatial,
    image,
    device="cpu",
    learning_rate=0.1,
    num_epochs=1000,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    print_each=100,
    random_state=None,
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_sp`).

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cv_train_genes (list): Optional. Training gene list. Default is None.
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for `mode=clusters`.
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'cell'.
        device (string or torch.device): Optional. Default is 'cpu'.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        scale (bool): Optional. Whether weight input single cell data by the number of cells in each cluster, only valid when cluster_label is not None. Default is True.
        lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 0.
        lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
        lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is True.
        density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If None, the density term is ignored. Default value is 'rna_count_based'.

    Returns:
        a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.
    """

    S = np.array(
        single_cell.cpu().detach().numpy(),
        dtype="float32",
    )
    G = np.array(spatial.cpu().detach().numpy(), dtype="float32")
    F = np.array(image.cpu().detach().numpy(), dtype="float32")

    # Choose device
    device = torch.device(device)  # for gpu

    """
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
    """

    hyperparameters = {
        "lambda_d": lambda_d,  # KL (ie density) term
        "lambda_g1": lambda_g1,  # gene-voxel cos sim
        "lambda_g2": lambda_g2,  # voxel-gene cos sim
        "lambda_r": lambda_r,  # regularizer: penalize entropy
    }

    mapper = mo.Mapper(
        S=S,
        G=G,
        image=F,
        device=device,
        random_state=random_state,
        **hyperparameters,
    )

    # TODO `train` should return the loss function

    mapping_matrix = mapper.train(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        print_each=print_each,
        batch_size=1289
    )

    return mapping_matrix
