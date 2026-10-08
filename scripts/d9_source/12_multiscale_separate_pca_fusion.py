import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# =========================================================
# Configuration
# =========================================================

ADATA_PATH = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D8 Squidpy/outputs/visium_hne_spatial_graph.h5ad"
)

LOWRES_EMBEDDING_PATH = Path(
    "outputs/dinov2_vits14_image_embeddings.npy"
)

FULLRES_EMBEDDING_PATH = Path(
    "outputs/dinov2_vits14_fullres_image_embeddings.npy"
)

MANIFEST_PATH = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

FULLRES_MANIFEST_PATH = Path(
    "outputs/fullres_spot_patch_manifest.csv"
)

SPLIT_PATH = Path(
    "outputs/prediction_split_manifest.csv"
)

OUTDIR = Path("outputs")
FIGDIR = Path("figures")

OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)


TARGET_GENES = [
    "Camk2n1",
    "Slc17a7",
    "Plp1",
    "Cldn11",
]


HIDDEN_DIM = 64
DROPOUT = 0.20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 500
PATIENCE = 60

RIDGE_ALPHA = 10.0


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =========================================================
# Metrics
# =========================================================

def evaluate_gene(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )


    if (
        np.std(y_true) > 0
        and
        np.std(y_pred) > 0
    ):

        pearson = pearsonr(
            y_true,
            y_pred,
        ).statistic


        spearman = spearmanr(
            y_true,
            y_pred,
        ).statistic

    else:

        pearson = np.nan
        spearman = np.nan


    r2 = r2_score(
        y_true,
        y_pred,
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


    return {

        "pearson_r":
            float(pearson),

        "spearman_rho":
            float(spearman),

        "r2":
            float(r2),

        "rmse":
            float(rmse),
    }


# =========================================================
# Neural models
# =========================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    ):

        super().__init__()


        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim,
        )


        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )


        self.head = nn.Linear(
            hidden_dim,
            output_dim,
        )


        self.dropout = dropout


    def forward(
        self,
        x,
        edge_index=None,
    ):

        x = self.fc1(
            x
        )

        x = F.relu(
            x
        )


        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )


        x = self.fc2(
            x
        )

        x = F.relu(
            x
        )


        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )


        return self.head(
            x
        )


class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    ):

        super().__init__()


        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim,
        )


        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim,
        )


        self.head = nn.Linear(
            hidden_dim,
            output_dim,
        )


        self.dropout = dropout


    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(
            x
        )


        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )


        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(
            x
        )


        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )


        return self.head(
            x
        )


def make_model(
    model_name,
    input_dim,
):

    if model_name == "MLP":

        return MLP(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=len(
                TARGET_GENES
            ),
            dropout=DROPOUT,
        )


    if model_name == "GraphSAGE":

        return GraphSAGE(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=len(
                TARGET_GENES
            ),
            dropout=DROPOUT,
        )


    raise ValueError(
        model_name
    )


# =========================================================
# 1. Load embeddings
# =========================================================

print(
    "\nLoading DINOv2 embeddings..."
)


X_low_raw = np.load(
    LOWRES_EMBEDDING_PATH
).astype(np.float32)


X_full_raw = np.load(
    FULLRES_EMBEDDING_PATH
).astype(np.float32)


print(
    "Low-res raw:",
    X_low_raw.shape
)


print(
    "Full-res raw:",
    X_full_raw.shape
)


assert (
    X_low_raw.shape
    ==
    X_full_raw.shape
)


assert (
    X_low_raw.shape[1]
    ==
    384
)


assert (
    np.isnan(
        X_low_raw
    ).sum()
    ==
    0
)


assert (
    np.isnan(
        X_full_raw
    ).sum()
    ==
    0
)


X_joint_raw = np.concatenate(
    [
        X_low_raw,
        X_full_raw,
    ],
    axis=1,
).astype(np.float32)


print(
    "Joint raw:",
    X_joint_raw.shape
)


# =========================================================
# 2. Barcode alignment
# =========================================================

manifest = pd.read_csv(
    MANIFEST_PATH
)


full_manifest = pd.read_csv(
    FULLRES_MANIFEST_PATH
)


barcodes = (
    manifest[
        "barcode"
    ]
    .astype(str)
    .tolist()
)


full_barcodes = (
    full_manifest[
        "barcode"
    ]
    .astype(str)
    .tolist()
)


if (
    barcodes
    !=
    full_barcodes
):

    raise ValueError(
        "Low/full barcode ordering mismatch."
    )


assert (
    len(barcodes)
    ==
    X_low_raw.shape[0]
)


print(
    "\nBarcode alignment successful:",
    len(barcodes),
    "spots"
)


# =========================================================
# 3. Load AnnData
# =========================================================

print(
    "\nLoading AnnData..."
)


adata = sc.read_h5ad(
    ADATA_PATH
)


missing = [
    b
    for b in barcodes
    if b not in adata.obs_names
]


if missing:

    raise ValueError(
        f"{len(missing)} barcodes "
        "missing from AnnData."
    )


adata = adata[
    barcodes
].copy()


assert (
    list(
        adata.obs_names
    )
    ==
    barcodes
)


# =========================================================
# 4. Load fixed split
# =========================================================

split_df = pd.read_csv(
    SPLIT_PATH
)


split_df[
    "barcode"
] = (
    split_df[
        "barcode"
    ]
    .astype(str)
)


split_df = (
    split_df
    .set_index(
        "barcode"
    )
    .loc[
        barcodes
    ]
    .reset_index()
)


assert (
    list(
        split_df[
            "barcode"
        ]
    )
    ==
    barcodes
)


# =========================================================
# 5. Expression targets
# =========================================================

Y = np.zeros(
    (
        len(barcodes),
        len(TARGET_GENES),
    ),
    dtype=np.float32,
)


print(
    "\nExpression targets:"
)


for j, gene in enumerate(
    TARGET_GENES
):

    if gene not in adata.var_names:

        raise KeyError(
            f"{gene} not found."
        )


    values = adata[
        :,
        gene
    ].X


    if sparse.issparse(
        values
    ):

        values = (
            values
            .toarray()
        )


    values = (
        np.asarray(
            values
        )
        .ravel()
        .astype(np.float32)
    )


    Y[
        :,
        j
    ] = values


    print(
        f"{gene}: "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}"
    )


# =========================================================
# 6. Spatial graph
# =========================================================

if (
    "spatial_connectivities"
    not in adata.obsp
):

    raise KeyError(
        "spatial_connectivities "
        "not found."
    )


A = (
    adata.obsp[
        "spatial_connectivities"
    ]
    .tocsr()
    .astype(np.float32)
    .copy()
)


A.setdiag(
    0
)

A.eliminate_zeros()


print(
    "\nSpatial graph:"
)

print(
    "Shape:",
    A.shape
)

print(
    "Edge entries:",
    A.nnz
)


# =========================================================
# 7. Strict spatial split
# =========================================================

train_pool_idx = np.where(
    split_df[
        "spatial_split"
    ].values
    ==
    "train"
)[0]


test_idx = np.where(
    split_df[
        "spatial_split"
    ].values
    ==
    "test"
)[0]


print(
    "\nStrict spatial split:"
)

print(
    "Train pool:",
    len(train_pool_idx)
)

print(
    "Test:",
    len(test_idx)
)


# =========================================================
# 8. Train-only validation split
# =========================================================

train_groups = (
    split_df.iloc[
        train_pool_idx
    ][
        "spatial_block"
    ]
    .values
)


gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)


fit_rel, val_rel = next(
    gss.split(
        train_pool_idx,
        groups=train_groups,
    )
)


fit_idx = (
    train_pool_idx[
        fit_rel
    ]
)


val_idx = (
    train_pool_idx[
        val_rel
    ]
)


print(
    "Fit:",
    len(fit_idx)
)

print(
    "Validation:",
    len(val_idx)
)


# =========================================================
# 9. Strict graph helper
# =========================================================

def get_subgraph(
    adjacency,
    indices,
):

    sub = (
        adjacency[
            indices
        ][:, indices]
        .tocoo()
    )


    edge_index = np.vstack(
        [
            sub.row,
            sub.col,
        ]
    ).astype(
        np.int64
    )


    return torch.from_numpy(
        edge_index
    )


edge_fit = get_subgraph(
    A,
    fit_idx,
)


edge_val = get_subgraph(
    A,
    val_idx,
)


edge_train = get_subgraph(
    A,
    train_pool_idx,
)


edge_test = get_subgraph(
    A,
    test_idx,
)


print(
    "\nStrict graph edges:"
)

print(
    "Fit:",
    edge_fit.shape[1]
)

print(
    "Validation:",
    edge_val.shape[1]
)

print(
    "Train:",
    edge_train.shape[1]
)

print(
    "Test:",
    edge_test.shape[1]
)


# =========================================================
# 10. Feature construction
#
# Selection features:
# PCA fitted ONLY on fit region.
#
# Final features:
# PCA fitted ONLY on full train region.
# =========================================================

def build_feature_set(
    feature_name,
    fit_indices,
    train_indices,
):

    print(
        "\n--------------------------------------"
    )

    print(
        "Building:",
        feature_name
    )

    print(
        "--------------------------------------"
    )


    metadata = {
        "feature_set":
            feature_name,
    }


    # =====================================================
    # LowRes PCA50
    # =====================================================

    if feature_name == "LowRes_PCA50":

        pca_select = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        pca_select.fit(
            X_low_raw[
                fit_indices
            ]
        )


        X_select = (
            pca_select.transform(
                X_low_raw
            )
            .astype(np.float32)
        )


        pca_final = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        pca_final.fit(
            X_low_raw[
                train_indices
            ]
        )


        X_final = (
            pca_final.transform(
                X_low_raw
            )
            .astype(np.float32)
        )


        metadata.update(
            {
                "feature_dim":
                    50,

                "low_pca_dim":
                    50,

                "full_pca_dim":
                    0,

                "joint_pca_dim":
                    0,

                "variance_low":
                    float(
                        pca_final
                        .explained_variance_ratio_
                        .sum()
                    ),

                "variance_full":
                    np.nan,

                "variance_joint":
                    np.nan,
            }
        )


        return (
            X_select,
            X_final,
            metadata,
        )


    # =====================================================
    # FullRes PCA50
    # =====================================================

    if feature_name == "FullRes_PCA50":

        pca_select = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        pca_select.fit(
            X_full_raw[
                fit_indices
            ]
        )


        X_select = (
            pca_select.transform(
                X_full_raw
            )
            .astype(np.float32)
        )


        pca_final = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        pca_final.fit(
            X_full_raw[
                train_indices
            ]
        )


        X_final = (
            pca_final.transform(
                X_full_raw
            )
            .astype(np.float32)
        )


        metadata.update(
            {
                "feature_dim":
                    50,

                "low_pca_dim":
                    0,

                "full_pca_dim":
                    50,

                "joint_pca_dim":
                    0,

                "variance_low":
                    np.nan,

                "variance_full":
                    float(
                        pca_final
                        .explained_variance_ratio_
                        .sum()
                    ),

                "variance_joint":
                    np.nan,
            }
        )


        return (
            X_select,
            X_final,
            metadata,
        )


    # =====================================================
    # Joint PCA50
    # =====================================================

    if feature_name == "JointPCA50":

        pca_select = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        pca_select.fit(
            X_joint_raw[
                fit_indices
            ]
        )


        X_select = (
            pca_select.transform(
                X_joint_raw
            )
            .astype(np.float32)
        )


        pca_final = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        pca_final.fit(
            X_joint_raw[
                train_indices
            ]
        )


        X_final = (
            pca_final.transform(
                X_joint_raw
            )
            .astype(np.float32)
        )


        metadata.update(
            {
                "feature_dim":
                    50,

                "low_pca_dim":
                    0,

                "full_pca_dim":
                    0,

                "joint_pca_dim":
                    50,

                "variance_low":
                    np.nan,

                "variance_full":
                    np.nan,

                "variance_joint":
                    float(
                        pca_final
                        .explained_variance_ratio_
                        .sum()
                    ),
            }
        )


        return (
            X_select,
            X_final,
            metadata,
        )


    # =====================================================
    # Separate PCA25 + PCA25 = 50-D
    # =====================================================

    if feature_name == "SeparatePCA25x2":

        low_select = PCA(
            n_components=25,
            svd_solver="randomized",
            random_state=42,
        )


        full_select = PCA(
            n_components=25,
            svd_solver="randomized",
            random_state=42,
        )


        low_select.fit(
            X_low_raw[
                fit_indices
            ]
        )


        full_select.fit(
            X_full_raw[
                fit_indices
            ]
        )


        X_low_select = (
            low_select.transform(
                X_low_raw
            )
            .astype(np.float32)
        )


        X_full_select = (
            full_select.transform(
                X_full_raw
            )
            .astype(np.float32)
        )


        X_select = np.concatenate(
            [
                X_low_select,
                X_full_select,
            ],
            axis=1,
        ).astype(np.float32)


        low_final = PCA(
            n_components=25,
            svd_solver="randomized",
            random_state=42,
        )


        full_final = PCA(
            n_components=25,
            svd_solver="randomized",
            random_state=42,
        )


        low_final.fit(
            X_low_raw[
                train_indices
            ]
        )


        full_final.fit(
            X_full_raw[
                train_indices
            ]
        )


        X_low_final = (
            low_final.transform(
                X_low_raw
            )
            .astype(np.float32)
        )


        X_full_final = (
            full_final.transform(
                X_full_raw
            )
            .astype(np.float32)
        )


        X_final = np.concatenate(
            [
                X_low_final,
                X_full_final,
            ],
            axis=1,
        ).astype(np.float32)


        metadata.update(
            {
                "feature_dim":
                    50,

                "low_pca_dim":
                    25,

                "full_pca_dim":
                    25,

                "joint_pca_dim":
                    0,

                "variance_low":
                    float(
                        low_final
                        .explained_variance_ratio_
                        .sum()
                    ),

                "variance_full":
                    float(
                        full_final
                        .explained_variance_ratio_
                        .sum()
                    ),

                "variance_joint":
                    np.nan,
            }
        )


        return (
            X_select,
            X_final,
            metadata,
        )


    # =====================================================
    # Joint PCA100
    # =====================================================

    if feature_name == "JointPCA100":

        pca_select = PCA(
            n_components=100,
            svd_solver="randomized",
            random_state=42,
        )


        pca_select.fit(
            X_joint_raw[
                fit_indices
            ]
        )


        X_select = (
            pca_select.transform(
                X_joint_raw
            )
            .astype(np.float32)
        )


        pca_final = PCA(
            n_components=100,
            svd_solver="randomized",
            random_state=42,
        )


        pca_final.fit(
            X_joint_raw[
                train_indices
            ]
        )


        X_final = (
            pca_final.transform(
                X_joint_raw
            )
            .astype(np.float32)
        )


        metadata.update(
            {
                "feature_dim":
                    100,

                "low_pca_dim":
                    0,

                "full_pca_dim":
                    0,

                "joint_pca_dim":
                    100,

                "variance_low":
                    np.nan,

                "variance_full":
                    np.nan,

                "variance_joint":
                    float(
                        pca_final
                        .explained_variance_ratio_
                        .sum()
                    ),
            }
        )


        return (
            X_select,
            X_final,
            metadata,
        )


    # =====================================================
    # Separate PCA50 + PCA50 = 100-D
    # =====================================================

    if feature_name == "SeparatePCA50x2":

        low_select = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        full_select = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        low_select.fit(
            X_low_raw[
                fit_indices
            ]
        )


        full_select.fit(
            X_full_raw[
                fit_indices
            ]
        )


        X_low_select = (
            low_select.transform(
                X_low_raw
            )
            .astype(np.float32)
        )


        X_full_select = (
            full_select.transform(
                X_full_raw
            )
            .astype(np.float32)
        )


        X_select = np.concatenate(
            [
                X_low_select,
                X_full_select,
            ],
            axis=1,
        ).astype(np.float32)


        low_final = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        full_final = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
        )


        low_final.fit(
            X_low_raw[
                train_indices
            ]
        )


        full_final.fit(
            X_full_raw[
                train_indices
            ]
        )


        X_low_final = (
            low_final.transform(
                X_low_raw
            )
            .astype(np.float32)
        )


        X_full_final = (
            full_final.transform(
                X_full_raw
            )
            .astype(np.float32)
        )


        X_final = np.concatenate(
            [
                X_low_final,
                X_full_final,
            ],
            axis=1,
        ).astype(np.float32)


        metadata.update(
            {
                "feature_dim":
                    100,

                "low_pca_dim":
                    50,

                "full_pca_dim":
                    50,

                "joint_pca_dim":
                    0,

                "variance_low":
                    float(
                        low_final
                        .explained_variance_ratio_
                        .sum()
                    ),

                "variance_full":
                    float(
                        full_final
                        .explained_variance_ratio_
                        .sum()
                    ),

                "variance_joint":
                    np.nan,
            }
        )


        return (
            X_select,
            X_final,
            metadata,
        )


    raise ValueError(
        feature_name
    )


# =========================================================
# 11. Model selection
# =========================================================

def select_epoch(
    model_name,
    seed,
    X_select_all,
):

    set_seed(
        seed
    )


    input_dim = (
        X_select_all.shape[1]
    )


    scaler = StandardScaler()


    X_fit = scaler.fit_transform(
        X_select_all[
            fit_idx
        ]
    ).astype(np.float32)


    X_val = scaler.transform(
        X_select_all[
            val_idx
        ]
    ).astype(np.float32)


    y_fit_raw = Y[
        fit_idx
    ]


    y_val_raw = Y[
        val_idx
    ]


    y_mean = (
        y_fit_raw.mean(
            axis=0
        )
    )


    y_std = (
        y_fit_raw.std(
            axis=0
        )
    )


    y_std[
        y_std
        <
        1e-6
    ] = 1.0


    y_fit = (
        (
            y_fit_raw
            -
            y_mean
        )
        /
        y_std
    ).astype(np.float32)


    y_val = (
        (
            y_val_raw
            -
            y_mean
        )
        /
        y_std
    ).astype(np.float32)


    x_fit_t = torch.from_numpy(
        X_fit
    ).to(device)


    x_val_t = torch.from_numpy(
        X_val
    ).to(device)


    y_fit_t = torch.from_numpy(
        y_fit
    ).to(device)


    y_val_t = torch.from_numpy(
        y_val
    ).to(device)


    ef = edge_fit.to(
        device
    )


    ev = edge_val.to(
        device
    )


    model = make_model(
        model_name,
        input_dim,
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    best_val = np.inf

    best_epoch = 1

    bad_epochs = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()


        if model_name == "GraphSAGE":

            pred = model(
                x_fit_t,
                ef,
            )

        else:

            pred = model(
                x_fit_t
            )


        loss = F.mse_loss(
            pred,
            y_fit_t,
        )


        loss.backward()

        optimizer.step()


        model.eval()


        with torch.no_grad():

            if model_name == "GraphSAGE":

                val_pred = model(
                    x_val_t,
                    ev,
                )

            else:

                val_pred = model(
                    x_val_t
                )


            val_loss = F.mse_loss(
                val_pred,
                y_val_t,
            ).item()


        if (
            val_loss
            <
            best_val - 1e-6
        ):

            best_val = val_loss

            best_epoch = epoch

            bad_epochs = 0

        else:

            bad_epochs += 1


        if (
            bad_epochs
            >=
            PATIENCE
        ):

            break


    return (
        best_epoch,
        best_val,
    )


# =========================================================
# 12. Final neural training
# =========================================================

def train_final(
    model_name,
    seed,
    best_epoch,
    X_final_all,
):

    set_seed(
        seed
    )


    input_dim = (
        X_final_all.shape[1]
    )


    scaler = StandardScaler()


    X_train = scaler.fit_transform(
        X_final_all[
            train_pool_idx
        ]
    ).astype(np.float32)


    X_test = scaler.transform(
        X_final_all[
            test_idx
        ]
    ).astype(np.float32)


    y_train_raw = Y[
        train_pool_idx
    ]


    y_mean = (
        y_train_raw.mean(
            axis=0
        )
    )


    y_std = (
        y_train_raw.std(
            axis=0
        )
    )


    y_std[
        y_std
        <
        1e-6
    ] = 1.0


    y_train = (
        (
            y_train_raw
            -
            y_mean
        )
        /
        y_std
    ).astype(np.float32)


    x_train_t = torch.from_numpy(
        X_train
    ).to(device)


    x_test_t = torch.from_numpy(
        X_test
    ).to(device)


    y_train_t = torch.from_numpy(
        y_train
    ).to(device)


    etrain = edge_train.to(
        device
    )


    etest = edge_test.to(
        device
    )


    model = make_model(
        model_name,
        input_dim,
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    for _ in range(
        best_epoch
    ):

        model.train()

        optimizer.zero_grad()


        if model_name == "GraphSAGE":

            pred = model(
                x_train_t,
                etrain,
            )

        else:

            pred = model(
                x_train_t
            )


        loss = F.mse_loss(
            pred,
            y_train_t,
        )


        loss.backward()

        optimizer.step()


    model.eval()


    with torch.no_grad():

        if model_name == "GraphSAGE":

            pred_scaled = model(
                x_test_t,
                etest,
            )

        else:

            pred_scaled = model(
                x_test_t
            )


    pred_scaled = (
        pred_scaled
        .cpu()
        .numpy()
    )


    pred = (
        pred_scaled
        *
        y_std
        +
        y_mean
    )


    return pred.astype(
        np.float32
    )


# =========================================================
# 13. Feature sets
# =========================================================

FEATURE_NAMES = [
    "LowRes_PCA50",
    "FullRes_PCA50",
    "JointPCA50",
    "SeparatePCA25x2",
    "JointPCA100",
    "SeparatePCA50x2",
]


# =========================================================
# 14. Main benchmark
# =========================================================

all_rows = []

feature_metadata_rows = []

prediction_store = {}


for feature_name in FEATURE_NAMES:


    (
        X_select_all,
        X_final_all,
        feature_metadata,
    ) = build_feature_set(
        feature_name,
        fit_idx,
        train_pool_idx,
    )


    print(
        "Selection features:",
        X_select_all.shape
    )


    print(
        "Final features:",
        X_final_all.shape
    )


    assert (
        X_select_all.shape[0]
        ==
        len(barcodes)
    )


    assert (
        X_final_all.shape[0]
        ==
        len(barcodes)
    )


    assert (
        np.isnan(
            X_select_all
        ).sum()
        ==
        0
    )


    assert (
        np.isnan(
            X_final_all
        ).sum()
        ==
        0
    )


    feature_metadata_rows.append(
        feature_metadata
    )


    prediction_store[
        feature_name
    ] = {}


    # =====================================================
    # Ridge
    # =====================================================

    print(
        "\nRidge:",
        feature_name
    )


    ridge = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),

            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA
                )
            ),
        ]
    )


    ridge.fit(
        X_final_all[
            train_pool_idx
        ],
        Y[
            train_pool_idx
        ],
    )


    ridge_pred = ridge.predict(
        X_final_all[
            test_idx
        ]
    )


    prediction_store[
        feature_name
    ][
        "Ridge"
    ] = ridge_pred


    for j, gene in enumerate(
        TARGET_GENES
    ):

        metric = evaluate_gene(
            Y[
                test_idx,
                j
            ],
            ridge_pred[
                :,
                j
            ],
        )


        all_rows.append(
            {
                "feature_set":
                    feature_name,

                "feature_dim":
                    X_final_all.shape[1],

                "model":
                    "Ridge",

                "gene":
                    gene,

                "seed":
                    np.nan,

                "best_epoch":
                    np.nan,

                **metric,
            }
        )


    # =====================================================
    # Neural models
    # =====================================================

    for model_name in [
        "MLP",
        "GraphSAGE",
    ]:


        predictions = []


        for seed in SEEDS:


            best_epoch, val_loss = (
                select_epoch(
                    model_name,
                    seed,
                    X_select_all,
                )
            )


            print(
                feature_name,
                model_name,
                f"seed={seed}",
                f"epoch={best_epoch}",
                f"val={val_loss:.4f}",
            )


            pred = train_final(
                model_name,
                seed,
                best_epoch,
                X_final_all,
            )


            predictions.append(
                pred
            )


            for j, gene in enumerate(
                TARGET_GENES
            ):

                metric = evaluate_gene(
                    Y[
                        test_idx,
                        j
                    ],
                    pred[
                        :,
                        j
                    ],
                )


                all_rows.append(
                    {
                        "feature_set":
                            feature_name,

                        "feature_dim":
                            X_final_all.shape[1],

                        "model":
                            model_name,

                        "gene":
                            gene,

                        "seed":
                            seed,

                        "best_epoch":
                            best_epoch,

                        **metric,
                    }
                )


        prediction_store[
            feature_name
        ][
            model_name
        ] = np.mean(
            np.stack(
                predictions,
                axis=0,
            ),
            axis=0,
        )


# =========================================================
# 15. Save raw metrics
# =========================================================

results = pd.DataFrame(
    all_rows
)


results.to_csv(
    OUTDIR
    / "separate_pca_fusion_raw_metrics.csv",
    index=False,
)


feature_metadata_df = pd.DataFrame(
    feature_metadata_rows
)


feature_metadata_df.to_csv(
    OUTDIR
    / "separate_pca_feature_summary.csv",
    index=False,
)


# =========================================================
# 16. Summary
# =========================================================

summary_rows = []


for (
    feature_set,
    feature_dim,
    model,
    gene
), group in results.groupby(
    [
        "feature_set",
        "feature_dim",
        "model",
        "gene",
    ]
):


    summary_rows.append(
        {
            "feature_set":
                feature_set,

            "feature_dim":
                feature_dim,

            "model":
                model,

            "gene":
                gene,

            "pearson_mean":
                group[
                    "pearson_r"
                ].mean(),

            "pearson_std":
                (
                    group[
                        "pearson_r"
                    ].std(
                        ddof=1
                    )
                    if len(group) > 1
                    else 0.0
                ),

            "spearman_mean":
                group[
                    "spearman_rho"
                ].mean(),

            "spearman_std":
                (
                    group[
                        "spearman_rho"
                    ].std(
                        ddof=1
                    )
                    if len(group) > 1
                    else 0.0
                ),

            "r2_mean":
                group[
                    "r2"
                ].mean(),

            "r2_std":
                (
                    group[
                        "r2"
                    ].std(
                        ddof=1
                    )
                    if len(group) > 1
                    else 0.0
                ),

            "rmse_mean":
                group[
                    "rmse"
                ].mean(),

            "rmse_std":
                (
                    group[
                        "rmse"
                    ].std(
                        ddof=1
                    )
                    if len(group) > 1
                    else 0.0
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTDIR
    / "separate_pca_fusion_benchmark.csv",
    index=False,
)


# =========================================================
# 17. Fair fusion comparisons
#
# 50-D:
# JointPCA50 vs SeparatePCA25x2
#
# 100-D:
# JointPCA100 vs SeparatePCA50x2
#
# Dimension effect:
# JointPCA50 vs JointPCA100
# =========================================================

comparison_pairs = [
    (
        "JointPCA50",
        "SeparatePCA25x2",
        "50D_joint_vs_separate"
    ),

    (
        "JointPCA100",
        "SeparatePCA50x2",
        "100D_joint_vs_separate"
    ),

    (
        "JointPCA50",
        "JointPCA100",
        "joint_dimension_50_vs_100"
    ),

    (
        "SeparatePCA25x2",
        "SeparatePCA50x2",
        "separate_dimension_50_vs_100"
    ),
]


comparison_rows = []


for (
    reference_name,
    candidate_name,
    comparison_name
) in comparison_pairs:


    for model_name in [
        "Ridge",
        "MLP",
        "GraphSAGE",
    ]:


        for gene in TARGET_GENES:


            ref = summary[
                (
                    summary[
                        "feature_set"
                    ]
                    ==
                    reference_name
                )
                &
                (
                    summary[
                        "model"
                    ]
                    ==
                    model_name
                )
                &
                (
                    summary[
                        "gene"
                    ]
                    ==
                    gene
                )
            ].iloc[0]


            cand = summary[
                (
                    summary[
                        "feature_set"
                    ]
                    ==
                    candidate_name
                )
                &
                (
                    summary[
                        "model"
                    ]
                    ==
                    model_name
                )
                &
                (
                    summary[
                        "gene"
                    ]
                    ==
                    gene
                )
            ].iloc[0]


            comparison_rows.append(
                {
                    "comparison":
                        comparison_name,

                    "reference":
                        reference_name,

                    "candidate":
                        candidate_name,

                    "model":
                        model_name,

                    "gene":
                        gene,

                    "reference_pearson":
                        ref[
                            "pearson_mean"
                        ],

                    "candidate_pearson":
                        cand[
                            "pearson_mean"
                        ],

                    "delta_pearson":
                        (
                            cand[
                                "pearson_mean"
                            ]
                            -
                            ref[
                                "pearson_mean"
                            ]
                        ),

                    "reference_r2":
                        ref[
                            "r2_mean"
                        ],

                    "candidate_r2":
                        cand[
                            "r2_mean"
                        ],

                    "delta_r2":
                        (
                            cand[
                                "r2_mean"
                            ]
                            -
                            ref[
                                "r2_mean"
                            ]
                        ),

                    "reference_rmse":
                        ref[
                            "rmse_mean"
                        ],

                    "candidate_rmse":
                        cand[
                            "rmse_mean"
                        ],

                    "delta_rmse":
                        (
                            cand[
                                "rmse_mean"
                            ]
                            -
                            ref[
                                "rmse_mean"
                            ]
                        ),
                }
            )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    OUTDIR
    / "fusion_bottleneck_comparison.csv",
    index=False,
)


# =========================================================
# 18. Multiscale vs best single-scale
# =========================================================

best_single_rows = []


for feature_name in [
    "JointPCA50",
    "SeparatePCA25x2",
    "JointPCA100",
    "SeparatePCA50x2",
]:


    for model_name in [
        "Ridge",
        "MLP",
        "GraphSAGE",
    ]:


        for gene in TARGET_GENES:


            low = summary[
                (
                    summary[
                        "feature_set"
                    ]
                    ==
                    "LowRes_PCA50"
                )
                &
                (
                    summary[
                        "model"
                    ]
                    ==
                    model_name
                )
                &
                (
                    summary[
                        "gene"
                    ]
                    ==
                    gene
                )
            ].iloc[0]


            full = summary[
                (
                    summary[
                        "feature_set"
                    ]
                    ==
                    "FullRes_PCA50"
                )
                &
                (
                    summary[
                        "model"
                    ]
                    ==
                    model_name
                )
                &
                (
                    summary[
                        "gene"
                    ]
                    ==
                    gene
                )
            ].iloc[0]


            fusion = summary[
                (
                    summary[
                        "feature_set"
                    ]
                    ==
                    feature_name
                )
                &
                (
                    summary[
                        "model"
                    ]
                    ==
                    model_name
                )
                &
                (
                    summary[
                        "gene"
                    ]
                    ==
                    gene
                )
            ].iloc[0]


            best_single_pearson = max(
                low[
                    "pearson_mean"
                ],
                full[
                    "pearson_mean"
                ],
            )


            best_single_r2 = max(
                low[
                    "r2_mean"
                ],
                full[
                    "r2_mean"
                ],
            )


            best_single_rows.append(
                {
                    "fusion_feature_set":
                        feature_name,

                    "model":
                        model_name,

                    "gene":
                        gene,

                    "lowres_pearson":
                        low[
                            "pearson_mean"
                        ],

                    "fullres_pearson":
                        full[
                            "pearson_mean"
                        ],

                    "fusion_pearson":
                        fusion[
                            "pearson_mean"
                        ],

                    "delta_fusion_minus_best_single_pearson":
                        (
                            fusion[
                                "pearson_mean"
                            ]
                            -
                            best_single_pearson
                        ),

                    "lowres_r2":
                        low[
                            "r2_mean"
                        ],

                    "fullres_r2":
                        full[
                            "r2_mean"
                        ],

                    "fusion_r2":
                        fusion[
                            "r2_mean"
                        ],

                    "delta_fusion_minus_best_single_r2":
                        (
                            fusion[
                                "r2_mean"
                            ]
                            -
                            best_single_r2
                        ),
                }
            )


best_single_df = pd.DataFrame(
    best_single_rows
)


best_single_df.to_csv(
    OUTDIR
    / "fusion_vs_best_single_scale.csv",
    index=False,
)


# =========================================================
# 19. Paired-seed stability
#
# Neural models only.
# =========================================================

neural = results[
    results[
        "model"
    ].isin(
        [
            "MLP",
            "GraphSAGE",
        ]
    )
].copy()


pivot = neural.pivot_table(
    index=[
        "model",
        "gene",
        "seed",
    ],

    columns="feature_set",

    values=[
        "pearson_r",
        "r2",
    ],
)


stability_rows = []


stability_pairs = [
    (
        "JointPCA50",
        "SeparatePCA25x2",
        "50D_joint_vs_separate"
    ),

    (
        "JointPCA100",
        "SeparatePCA50x2",
        "100D_joint_vs_separate"
    ),

    (
        "JointPCA50",
        "JointPCA100",
        "joint_50_vs_100"
    ),

    (
        "SeparatePCA25x2",
        "SeparatePCA50x2",
        "separate_50_vs_100"
    ),
]


for (
    reference_name,
    candidate_name,
    comparison_name
) in stability_pairs:


    for (
        model_name,
        gene
    ), group in pivot.groupby(
        level=[
            0,
            1,
        ]
    ):


        delta_r = (
            group[
                (
                    "pearson_r",
                    candidate_name
                )
            ]
            -
            group[
                (
                    "pearson_r",
                    reference_name
                )
            ]
        )


        delta_r2 = (
            group[
                (
                    "r2",
                    candidate_name
                )
            ]
            -
            group[
                (
                    "r2",
                    reference_name
                )
            ]
        )


        stability_rows.append(
            {
                "comparison":
                    comparison_name,

                "reference":
                    reference_name,

                "candidate":
                    candidate_name,

                "model":
                    model_name,

                "gene":
                    gene,

                "delta_pearson_mean":
                    delta_r.mean(),

                "delta_pearson_std":
                    delta_r.std(),

                "candidate_pearson_wins":
                    int(
                        (
                            delta_r > 0
                        ).sum()
                    ),

                "delta_r2_mean":
                    delta_r2.mean(),

                "delta_r2_std":
                    delta_r2.std(),

                "candidate_r2_wins":
                    int(
                        (
                            delta_r2 > 0
                        ).sum()
                    ),

                "n_seeds":
                    len(
                        delta_r
                    ),
            }
        )


stability = pd.DataFrame(
    stability_rows
)


stability.to_csv(
    OUTDIR
    / "fusion_paired_seed_stability.csv",
    index=False,
)


# =========================================================
# 20. Save ensemble test predictions
# =========================================================

prediction_df = pd.DataFrame(
    {
        "barcode":
            manifest.iloc[
                test_idx
            ][
                "barcode"
            ].values,

        "fullres_x":
            manifest.iloc[
                test_idx
            ][
                "fullres_x"
            ].values,

        "fullres_y":
            manifest.iloc[
                test_idx
            ][
                "fullres_y"
            ].values,
    }
)


for feature_name in FEATURE_NAMES:

    for model_name in [
        "Ridge",
        "MLP",
        "GraphSAGE",
    ]:

        pred = prediction_store[
            feature_name
        ][
            model_name
        ]


        for j, gene in enumerate(
            TARGET_GENES
        ):

            prediction_df[
                (
                    f"{gene}_"
                    f"{feature_name}_"
                    f"{model_name}"
                )
            ] = pred[
                :,
                j
            ]


for j, gene in enumerate(
    TARGET_GENES
):

    prediction_df[
        f"{gene}_true"
    ] = Y[
        test_idx,
        j
    ]


prediction_df.to_csv(
    OUTDIR
    / "separate_pca_fusion_test_predictions.csv",
    index=False,
)


# =========================================================
# 21. Print key outputs
# =========================================================

print(
    "\n======================================"
)

print(
    "Feature summary"
)

print(
    "======================================"
)


print(
    feature_metadata_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Fusion bottleneck comparison"
)

print(
    "======================================"
)


print(
    comparison.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Paired-seed stability"
)

print(
    "======================================"
)


print(
    stability.to_string(
        index=False
    )
)


# =========================================================
# 22. Plot GraphSAGE Pearson
# =========================================================

graph_df = summary[
    summary[
        "model"
    ]
    ==
    "GraphSAGE"
].copy()


plot_features = [
    "LowRes_PCA50",
    "FullRes_PCA50",
    "JointPCA50",
    "SeparatePCA25x2",
    "JointPCA100",
    "SeparatePCA50x2",
]


x = np.arange(
    len(TARGET_GENES)
)


width = 0.13


fig, ax = plt.subplots(
    figsize=(
        14,
        7
    )
)


for k, feature_name in enumerate(
    plot_features
):


    values = []

    errors = []


    for gene in TARGET_GENES:

        row = graph_df[
            (
                graph_df[
                    "feature_set"
                ]
                ==
                feature_name
            )
            &
            (
                graph_df[
                    "gene"
                ]
                ==
                gene
            )
        ].iloc[0]


        values.append(
            row[
                "pearson_mean"
            ]
        )


        errors.append(
            row[
                "pearson_std"
            ]
        )


    offset = (
        k
        -
        (
            len(
                plot_features
            )
            - 1
        )
        / 2
    ) * width


    ax.bar(
        x + offset,
        values,
        width,
        yerr=errors,
        label=feature_name,
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    TARGET_GENES
)


ax.set_ylabel(
    "Pearson r"
)


ax.set_title(
    "Multiscale fusion bottleneck ablation\n"
    "Strict spatial-block GraphSAGE"
)


ax.legend(
    fontsize=8
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "39_separate_pca_fusion_graphsage.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 23. Fair comparison delta plot
#
# SeparatePCA50x2 - JointPCA100
# =========================================================

delta_df = comparison[
    (
        comparison[
            "comparison"
        ]
        ==
        "100D_joint_vs_separate"
    )
    &
    (
        comparison[
            "model"
        ]
        ==
        "GraphSAGE"
    )
].copy()


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    delta_df[
        "gene"
    ],
    delta_df[
        "delta_pearson"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r\n"
    "(Separate PCA 50+50 - Joint PCA100)"
)


ax.set_title(
    "Does scale-specific PCA preserve "
    "complementary information?"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "40_separate_vs_joint_pca100_delta.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 24. 50-D fair comparison
# =========================================================

delta50 = comparison[
    (
        comparison[
            "comparison"
        ]
        ==
        "50D_joint_vs_separate"
    )
    &
    (
        comparison[
            "model"
        ]
        ==
        "GraphSAGE"
    )
].copy()


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    delta50[
        "gene"
    ],
    delta50[
        "delta_pearson"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r\n"
    "(Separate PCA25+25 - Joint PCA50)"
)


ax.set_title(
    "Fusion strategy at fixed 50-D capacity"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "41_separate_vs_joint_pca50_delta.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 25. Dimension effect plot
# =========================================================

dimension_df = comparison[
    (
        comparison[
            "comparison"
        ]
        ==
        "joint_dimension_50_vs_100"
    )
    &
    (
        comparison[
            "model"
        ]
        ==
        "GraphSAGE"
    )
].copy()


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    dimension_df[
        "gene"
    ],
    dimension_df[
        "delta_pearson"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r\n"
    "(Joint PCA100 - Joint PCA50)"
)


ax.set_title(
    "Effect of increasing fusion dimension"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "42_joint_pca_dimension_effect.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 26. Final report
# =========================================================

print(
    "\nSaved:"
)

print(
    "outputs/separate_pca_feature_summary.csv"
)

print(
    "outputs/separate_pca_fusion_raw_metrics.csv"
)

print(
    "outputs/separate_pca_fusion_benchmark.csv"
)

print(
    "outputs/fusion_bottleneck_comparison.csv"
)

print(
    "outputs/fusion_vs_best_single_scale.csv"
)

print(
    "outputs/fusion_paired_seed_stability.csv"
)

print(
    "outputs/separate_pca_fusion_test_predictions.csv"
)

print(
    "figures/39_separate_pca_fusion_graphsage.png"
)

print(
    "figures/40_separate_vs_joint_pca100_delta.png"
)

print(
    "figures/41_separate_vs_joint_pca50_delta.png"
)

print(
    "figures/42_joint_pca_dimension_effect.png"
)


print(
    "\nMultiscale separate-PCA fusion experiment complete."
)
