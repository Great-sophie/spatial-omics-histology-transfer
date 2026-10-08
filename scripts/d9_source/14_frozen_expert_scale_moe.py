import copy
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

SCRIPT12_BASELINE = Path(
    "outputs/separate_pca_fusion_raw_metrics.csv"
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


PCA_DIM = 50

HIDDEN_DIM = 64
DROPOUT = 0.20

EXPERT_LR = 1e-3
EXPERT_WEIGHT_DECAY = 1e-4

EXPERT_MAX_EPOCHS = 500
EXPERT_PATIENCE = 60

STATIC_GATE_LR = 5e-2
DYNAMIC_GATE_LR = 1e-3

GATE_WEIGHT_DECAY = 1e-4
GATE_MAX_EPOCHS = 800
GATE_PATIENCE = 100

DYNAMIC_GATE_HIDDEN = 16
DYNAMIC_GATE_DROPOUT = 0.10


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


FAMILIES = [
    "MLP",
    "GraphSAGE",
]


GATE_TYPES = [
    "Static",
    "Dynamic",
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
    "Low-res:",
    X_low_raw.shape
)

print(
    "Full-res:",
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


print(
    "\nBarcode alignment:",
    len(barcodes),
    "spots"
)


# =========================================================
# 3. Load AnnData
# =========================================================

adata = sc.read_h5ad(
    ADATA_PATH
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
# 4. Fixed spatial split
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

A = (
    adata.obsp[
        "spatial_connectivities"
    ]
    .tocsr()
    .astype(np.float32)
    .copy()
)


A.setdiag(0)

A.eliminate_zeros()


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


# =========================================================
# 7. Strict train/test split
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
# 8. Spatial-block validation
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


# =========================================================
# 9. Strict scale-specific PCA
# =========================================================

print(
    "\nFitting scale-specific PCA..."
)


# ---------------------------------------------------------
# Selection PCA
# Fit only on fit region
# ---------------------------------------------------------

low_pca_select = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42,
)


full_pca_select = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42,
)


low_pca_select.fit(
    X_low_raw[
        fit_idx
    ]
)


full_pca_select.fit(
    X_full_raw[
        fit_idx
    ]
)


X_low_select = (
    low_pca_select.transform(
        X_low_raw
    )
    .astype(np.float32)
)


X_full_select = (
    full_pca_select.transform(
        X_full_raw
    )
    .astype(np.float32)
)


# ---------------------------------------------------------
# Final PCA
# Fit only on entire training region
# ---------------------------------------------------------

low_pca_final = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42,
)


full_pca_final = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42,
)


low_pca_final.fit(
    X_low_raw[
        train_pool_idx
    ]
)


full_pca_final.fit(
    X_full_raw[
        train_pool_idx
    ]
)


X_low_final = (
    low_pca_final.transform(
        X_low_raw
    )
    .astype(np.float32)
)


X_full_final = (
    full_pca_final.transform(
        X_full_raw
    )
    .astype(np.float32)
)


print(
    "Low PCA variance:",
    float(
        low_pca_final
        .explained_variance_ratio_
        .sum()
    )
)


print(
    "Full PCA variance:",
    float(
        full_pca_final
        .explained_variance_ratio_
        .sum()
    )
)


# =========================================================
# 10. Models
# =========================================================

class MLPExpert(nn.Module):

    def __init__(self):

        super().__init__()


        self.fc1 = nn.Linear(
            PCA_DIM,
            HIDDEN_DIM,
        )


        self.fc2 = nn.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM,
        )


        self.head = nn.Linear(
            HIDDEN_DIM,
            len(TARGET_GENES),
        )


    def forward(
        self,
        x,
        edge_index=None,
    ):

        x = F.relu(
            self.fc1(x)
        )


        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )


        x = F.relu(
            self.fc2(x)
        )


        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )


        return self.head(
            x
        )


class GraphSAGEExpert(nn.Module):

    def __init__(self):

        super().__init__()


        self.conv1 = SAGEConv(
            PCA_DIM,
            HIDDEN_DIM,
        )


        self.conv2 = SAGEConv(
            HIDDEN_DIM,
            HIDDEN_DIM,
        )


        self.head = nn.Linear(
            HIDDEN_DIM,
            len(TARGET_GENES),
        )


    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )


        x = F.relu(x)


        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )


        x = self.conv2(
            x,
            edge_index,
        )


        x = F.relu(x)


        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )


        return self.head(
            x
        )


def make_expert(
    family,
):

    if family == "MLP":

        return MLPExpert()


    if family == "GraphSAGE":

        return GraphSAGEExpert()


    raise ValueError(
        family
    )


# =========================================================
# 11. Prediction-level gates
#
# alpha = LOW-resolution weight
#
# y_hat =
# alpha * low_expert
# +
# (1-alpha) * full_expert
# =========================================================

class StaticPredictionGate(nn.Module):

    def __init__(self):

        super().__init__()


        self.logits = nn.Parameter(
            torch.zeros(
                len(TARGET_GENES)
            )
        )


    def forward(
        self,
        low_pred,
        full_pred,
    ):

        alpha_gene = torch.sigmoid(
            self.logits
        )


        alpha = (
            alpha_gene
            .unsqueeze(0)
            .expand(
                low_pred.shape[0],
                -1,
            )
        )


        fused = (
            alpha
            *
            low_pred
            +
            (
                1.0
                -
                alpha
            )
            *
            full_pred
        )


        return (
            fused,
            alpha
        )


class DynamicPredictionGate(nn.Module):

    def __init__(self):

        super().__init__()


        # One tiny gate network per gene.
        # Each gate only sees:
        #
        # [low prediction for gene g,
        #  full prediction for gene g]
        #
        # This keeps the interpretation gene-specific.

        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        2,
                        DYNAMIC_GATE_HIDDEN,
                    ),

                    nn.ReLU(),

                    nn.Dropout(
                        DYNAMIC_GATE_DROPOUT
                    ),

                    nn.Linear(
                        DYNAMIC_GATE_HIDDEN,
                        1,
                    ),
                )

                for _ in TARGET_GENES
            ]
        )


    def forward(
        self,
        low_pred,
        full_pred,
    ):

        alpha_list = []


        for g in range(
            len(TARGET_GENES)
        ):

            gate_input = torch.stack(
                [
                    low_pred[
                        :,
                        g
                    ],

                    full_pred[
                        :,
                        g
                    ],
                ],
                dim=1,
            )


            alpha_g = torch.sigmoid(
                self.gates[
                    g
                ](
                    gate_input
                )
            )


            alpha_list.append(
                alpha_g
            )


        alpha = torch.cat(
            alpha_list,
            dim=1,
        )


        fused = (
            alpha
            *
            low_pred
            +
            (
                1.0
                -
                alpha
            )
            *
            full_pred
        )


        return (
            fused,
            alpha
        )


def make_gate(
    gate_type,
):

    if gate_type == "Static":

        return StaticPredictionGate()


    if gate_type == "Dynamic":

        return DynamicPredictionGate()


    raise ValueError(
        gate_type
    )


# =========================================================
# 12. Expert forward helper
# =========================================================

def expert_forward(
    model,
    family,
    x,
    edge_index,
):

    if family == "GraphSAGE":

        return model(
            x,
            edge_index,
        )


    return model(
        x
    )


# =========================================================
# 13. Selection preprocessing
# =========================================================

low_scaler_select = StandardScaler()

full_scaler_select = StandardScaler()


X_low_fit = low_scaler_select.fit_transform(
    X_low_select[
        fit_idx
    ]
).astype(np.float32)


X_low_val = low_scaler_select.transform(
    X_low_select[
        val_idx
    ]
).astype(np.float32)


X_full_fit = full_scaler_select.fit_transform(
    X_full_select[
        fit_idx
    ]
).astype(np.float32)


X_full_val = full_scaler_select.transform(
    X_full_select[
        val_idx
    ]
).astype(np.float32)


Y_fit_raw = Y[
    fit_idx
]


Y_val_raw = Y[
    val_idx
]


Y_mean_select = Y_fit_raw.mean(
    axis=0
)


Y_std_select = Y_fit_raw.std(
    axis=0
)


Y_std_select[
    Y_std_select < 1e-6
] = 1.0


Y_fit = (
    (
        Y_fit_raw
        -
        Y_mean_select
    )
    /
    Y_std_select
).astype(np.float32)


Y_val = (
    (
        Y_val_raw
        -
        Y_mean_select
    )
    /
    Y_std_select
).astype(np.float32)


# =========================================================
# 14. Final preprocessing
# =========================================================

low_scaler_final = StandardScaler()

full_scaler_final = StandardScaler()


X_low_train = low_scaler_final.fit_transform(
    X_low_final[
        train_pool_idx
    ]
).astype(np.float32)


X_low_test = low_scaler_final.transform(
    X_low_final[
        test_idx
    ]
).astype(np.float32)


X_full_train = full_scaler_final.fit_transform(
    X_full_final[
        train_pool_idx
    ]
).astype(np.float32)


X_full_test = full_scaler_final.transform(
    X_full_final[
        test_idx
    ]
).astype(np.float32)


Y_train_raw = Y[
    train_pool_idx
]


Y_test_raw = Y[
    test_idx
]


Y_mean_final = Y_train_raw.mean(
    axis=0
)


Y_std_final = Y_train_raw.std(
    axis=0
)


Y_std_final[
    Y_std_final < 1e-6
] = 1.0


Y_train = (
    (
        Y_train_raw
        -
        Y_mean_final
    )
    /
    Y_std_final
).astype(np.float32)


# =========================================================
# 15. Expert model selection
# =========================================================

def select_expert(
    family,
    seed,
    X_fit_np,
    X_val_np,
):

    set_seed(
        seed
    )


    x_fit = torch.from_numpy(
        X_fit_np
    ).to(device)


    x_val = torch.from_numpy(
        X_val_np
    ).to(device)


    y_fit = torch.from_numpy(
        Y_fit
    ).to(device)


    y_val = torch.from_numpy(
        Y_val
    ).to(device)


    ef = edge_fit.to(
        device
    )


    ev = edge_val.to(
        device
    )


    model = make_expert(
        family
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT_LR,
        weight_decay=EXPERT_WEIGHT_DECAY,
    )


    best_loss = np.inf

    best_epoch = 1

    best_state = None

    bad_epochs = 0


    for epoch in range(
        1,
        EXPERT_MAX_EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad()


        pred = expert_forward(
            model,
            family,
            x_fit,
            ef,
        )


        loss = F.mse_loss(
            pred,
            y_fit,
        )


        loss.backward()

        optimizer.step()


        model.eval()


        with torch.no_grad():

            val_pred = expert_forward(
                model,
                family,
                x_val,
                ev,
            )


            val_loss = F.mse_loss(
                val_pred,
                y_val,
            ).item()


        if (
            val_loss
            <
            best_loss - 1e-6
        ):

            best_loss = val_loss

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            bad_epochs = 0

        else:

            bad_epochs += 1


        if (
            bad_epochs
            >=
            EXPERT_PATIENCE
        ):

            break


    if best_state is None:

        raise RuntimeError(
            "Expert selection failed."
        )


    model.load_state_dict(
        best_state
    )


    model.eval()


    return (
        best_epoch,
        best_loss,
        model,
    )


# =========================================================
# 16. Gate model selection
#
# Experts are FROZEN.
# Gate sees only expert predictions.
# =========================================================

def select_gate(
    family,
    gate_type,
    seed,
    low_expert,
    full_expert,
):

    set_seed(
        seed + 1000
    )


    x_low_fit = torch.from_numpy(
        X_low_fit
    ).to(device)


    x_low_val = torch.from_numpy(
        X_low_val
    ).to(device)


    x_full_fit = torch.from_numpy(
        X_full_fit
    ).to(device)


    x_full_val = torch.from_numpy(
        X_full_val
    ).to(device)


    y_fit = torch.from_numpy(
        Y_fit
    ).to(device)


    y_val = torch.from_numpy(
        Y_val
    ).to(device)


    ef = edge_fit.to(
        device
    )


    ev = edge_val.to(
        device
    )


    low_expert.eval()

    full_expert.eval()


    for p in low_expert.parameters():

        p.requires_grad = False


    for p in full_expert.parameters():

        p.requires_grad = False


    with torch.no_grad():

        low_fit_pred = expert_forward(
            low_expert,
            family,
            x_low_fit,
            ef,
        ).detach()


        low_val_pred = expert_forward(
            low_expert,
            family,
            x_low_val,
            ev,
        ).detach()


        full_fit_pred = expert_forward(
            full_expert,
            family,
            x_full_fit,
            ef,
        ).detach()


        full_val_pred = expert_forward(
            full_expert,
            family,
            x_full_val,
            ev,
        ).detach()


    gate = make_gate(
        gate_type
    ).to(device)


    if gate_type == "Static":

        gate_lr = STATIC_GATE_LR

        weight_decay = 0.0

    else:

        gate_lr = DYNAMIC_GATE_LR

        weight_decay = GATE_WEIGHT_DECAY


    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=gate_lr,
        weight_decay=weight_decay,
    )


    best_loss = np.inf

    best_epoch = 1

    best_state = None

    bad_epochs = 0


    for epoch in range(
        1,
        GATE_MAX_EPOCHS + 1,
    ):

        gate.train()

        optimizer.zero_grad()


        fused, _ = gate(
            low_fit_pred,
            full_fit_pred,
        )


        loss = F.mse_loss(
            fused,
            y_fit,
        )


        loss.backward()

        optimizer.step()


        gate.eval()


        with torch.no_grad():

            val_fused, _ = gate(
                low_val_pred,
                full_val_pred,
            )


            val_loss = F.mse_loss(
                val_fused,
                y_val,
            ).item()


        if (
            val_loss
            <
            best_loss - 1e-7
        ):

            best_loss = val_loss

            best_epoch = epoch

            best_state = copy.deepcopy(
                gate.state_dict()
            )

            bad_epochs = 0

        else:

            bad_epochs += 1


        if (
            bad_epochs
            >=
            GATE_PATIENCE
        ):

            break


    if best_state is None:

        raise RuntimeError(
            "Gate selection failed."
        )


    gate.load_state_dict(
        best_state
    )


    gate.eval()


    return (
        best_epoch,
        best_loss,
    )


# =========================================================
# 17. Final expert training
# =========================================================

def train_expert_fixed(
    family,
    seed,
    epochs,
    X_train_np,
):

    set_seed(
        seed
    )


    x_train = torch.from_numpy(
        X_train_np
    ).to(device)


    y_train = torch.from_numpy(
        Y_train
    ).to(device)


    etrain = edge_train.to(
        device
    )


    model = make_expert(
        family
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT_LR,
        weight_decay=EXPERT_WEIGHT_DECAY,
    )


    for _ in range(
        int(epochs)
    ):

        model.train()

        optimizer.zero_grad()


        pred = expert_forward(
            model,
            family,
            x_train,
            etrain,
        )


        loss = F.mse_loss(
            pred,
            y_train,
        )


        loss.backward()

        optimizer.step()


    model.eval()


    return model


# =========================================================
# 18. Final gate training
#
# Experts stay frozen.
# =========================================================

def train_gate_fixed(
    family,
    gate_type,
    seed,
    epochs,
    low_expert,
    full_expert,
):

    set_seed(
        seed + 1000
    )


    x_low_train = torch.from_numpy(
        X_low_train
    ).to(device)


    x_full_train = torch.from_numpy(
        X_full_train
    ).to(device)


    y_train = torch.from_numpy(
        Y_train
    ).to(device)


    etrain = edge_train.to(
        device
    )


    low_expert.eval()

    full_expert.eval()


    for p in low_expert.parameters():

        p.requires_grad = False


    for p in full_expert.parameters():

        p.requires_grad = False


    with torch.no_grad():

        low_train_pred = expert_forward(
            low_expert,
            family,
            x_low_train,
            etrain,
        ).detach()


        full_train_pred = expert_forward(
            full_expert,
            family,
            x_full_train,
            etrain,
        ).detach()


    gate = make_gate(
        gate_type
    ).to(device)


    if gate_type == "Static":

        gate_lr = STATIC_GATE_LR

        weight_decay = 0.0

    else:

        gate_lr = DYNAMIC_GATE_LR

        weight_decay = GATE_WEIGHT_DECAY


    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=gate_lr,
        weight_decay=weight_decay,
    )


    for _ in range(
        int(epochs)
    ):

        gate.train()

        optimizer.zero_grad()


        fused, _ = gate(
            low_train_pred,
            full_train_pred,
        )


        loss = F.mse_loss(
            fused,
            y_train,
        )


        loss.backward()

        optimizer.step()


    gate.eval()


    return gate


# =========================================================
# 19. Final test prediction
# =========================================================

def predict_frozen_moe(
    family,
    low_expert,
    full_expert,
    gate,
):

    x_low_test = torch.from_numpy(
        X_low_test
    ).to(device)


    x_full_test = torch.from_numpy(
        X_full_test
    ).to(device)


    etest = edge_test.to(
        device
    )


    low_expert.eval()

    full_expert.eval()

    gate.eval()


    with torch.no_grad():

        low_pred_std = expert_forward(
            low_expert,
            family,
            x_low_test,
            etest,
        )


        full_pred_std = expert_forward(
            full_expert,
            family,
            x_full_test,
            etest,
        )


        fused_std, alpha = gate(
            low_pred_std,
            full_pred_std,
        )


    low_pred_std = (
        low_pred_std
        .cpu()
        .numpy()
    )


    full_pred_std = (
        full_pred_std
        .cpu()
        .numpy()
    )


    fused_std = (
        fused_std
        .cpu()
        .numpy()
    )


    alpha = (
        alpha
        .cpu()
        .numpy()
    )


    low_pred = (
        low_pred_std
        *
        Y_std_final
        +
        Y_mean_final
    )


    full_pred = (
        full_pred_std
        *
        Y_std_final
        +
        Y_mean_final
    )


    fused_pred = (
        fused_std
        *
        Y_std_final
        +
        Y_mean_final
    )


    return (
        low_pred.astype(np.float32),
        full_pred.astype(np.float32),
        fused_pred.astype(np.float32),
        alpha.astype(np.float32),
    )


# =========================================================
# 20. Main experiment
# =========================================================

metric_rows = []

gate_rows = []

training_rows = []

prediction_store = {}

dynamic_graph_gate_store = []


for family in FAMILIES:


    print(
        "\n"
        "=============================================="
    )

    print(
        "MODEL FAMILY:",
        family
    )

    print(
        "=============================================="
    )


    for seed in SEEDS:


        print(
            f"\n--- {family} seed={seed} ---"
        )


        # =================================================
        # A. Select low expert
        # =================================================

        (
            low_epoch,
            low_val_loss,
            low_select_model,
        ) = select_expert(
            family=family,
            seed=seed,
            X_fit_np=X_low_fit,
            X_val_np=X_low_val,
        )


        # =================================================
        # B. Select full expert
        # =================================================

        (
            full_epoch,
            full_val_loss,
            full_select_model,
        ) = select_expert(
            family=family,
            seed=seed,
            X_fit_np=X_full_fit,
            X_val_np=X_full_val,
        )


        print(
            "Low expert:",
            f"epoch={low_epoch}",
            f"val={low_val_loss:.4f}"
        )


        print(
            "Full expert:",
            f"epoch={full_epoch}",
            f"val={full_val_loss:.4f}"
        )


        # =================================================
        # C. Select gate epochs
        # =================================================

        selected_gate_epochs = {}


        for gate_type in GATE_TYPES:


            (
                gate_epoch,
                gate_val_loss,
            ) = select_gate(
                family=family,
                gate_type=gate_type,
                seed=seed,
                low_expert=low_select_model,
                full_expert=full_select_model,
            )


            selected_gate_epochs[
                gate_type
            ] = gate_epoch


            print(
                gate_type,
                "gate:",
                f"epoch={gate_epoch}",
                f"val={gate_val_loss:.4f}"
            )


            training_rows.append(
                {
                    "family":
                        family,

                    "seed":
                        seed,

                    "gate_type":
                        gate_type,

                    "low_expert_epoch":
                        low_epoch,

                    "low_expert_val_loss":
                        low_val_loss,

                    "full_expert_epoch":
                        full_epoch,

                    "full_expert_val_loss":
                        full_val_loss,

                    "gate_epoch":
                        gate_epoch,

                    "gate_val_loss":
                        gate_val_loss,
                }
            )


        # =================================================
        # D. Final experts on entire training region
        # =================================================

        low_final_model = train_expert_fixed(
            family=family,
            seed=seed,
            epochs=low_epoch,
            X_train_np=X_low_train,
        )


        full_final_model = train_expert_fixed(
            family=family,
            seed=seed,
            epochs=full_epoch,
            X_train_np=X_full_train,
        )


        # =================================================
        # E. Train final frozen gates
        # =================================================

        for gate_type in GATE_TYPES:


            gate_model = train_gate_fixed(
                family=family,
                gate_type=gate_type,
                seed=seed,
                epochs=selected_gate_epochs[
                    gate_type
                ],
                low_expert=low_final_model,
                full_expert=full_final_model,
            )


            (
                low_pred,
                full_pred,
                moe_pred,
                alpha,
            ) = predict_frozen_moe(
                family=family,
                low_expert=low_final_model,
                full_expert=full_final_model,
                gate=gate_model,
            )


            moe_name = (
                f"Frozen{gate_type}MoE_"
                f"{family}"
            )


            # ---------------------------------------------
            # Expert metrics saved once per gate run.
            # Duplicate expert rows are avoided below.
            # ---------------------------------------------

            if gate_type == "Static":


                for j, gene in enumerate(
                    TARGET_GENES
                ):


                    low_metric = evaluate_gene(
                        Y_test_raw[
                            :,
                            j
                        ],
                        low_pred[
                            :,
                            j
                        ],
                    )


                    full_metric = evaluate_gene(
                        Y_test_raw[
                            :,
                            j
                        ],
                        full_pred[
                            :,
                            j
                        ],
                    )


                    metric_rows.append(
                        {
                            "model":
                                f"LowExpert_{family}",

                            "family":
                                family,

                            "gene":
                                gene,

                            "seed":
                                seed,

                            **low_metric,
                        }
                    )


                    metric_rows.append(
                        {
                            "model":
                                f"FullExpert_{family}",

                            "family":
                                family,

                            "gene":
                                gene,

                            "seed":
                                seed,

                            **full_metric,
                        }
                    )


            # ---------------------------------------------
            # MoE metrics
            # ---------------------------------------------

            for j, gene in enumerate(
                TARGET_GENES
            ):


                metric = evaluate_gene(
                    Y_test_raw[
                        :,
                        j
                    ],
                    moe_pred[
                        :,
                        j
                    ],
                )


                metric_rows.append(
                    {
                        "model":
                            moe_name,

                        "family":
                            family,

                        "gene":
                            gene,

                        "seed":
                            seed,

                        **metric,
                    }
                )


                gene_alpha = alpha[
                    :,
                    j
                ]


                gate_rows.append(
                    {
                        "model":
                            moe_name,

                        "family":
                            family,

                        "gate_type":
                            gate_type,

                        "gene":
                            gene,

                        "seed":
                            seed,

                        "low_weight_mean":
                            float(
                                gene_alpha.mean()
                            ),

                        "low_weight_std":
                            float(
                                gene_alpha.std()
                            ),

                        "low_weight_min":
                            float(
                                gene_alpha.min()
                            ),

                        "low_weight_max":
                            float(
                                gene_alpha.max()
                            ),

                        "full_weight_mean":
                            float(
                                1.0
                                -
                                gene_alpha.mean()
                            ),
                    }
                )


            # ---------------------------------------------
            # Store predictions
            # ---------------------------------------------

            prediction_store[
                (
                    moe_name,
                    seed
                )
            ] = moe_pred


            # ---------------------------------------------
            # Dynamic GraphSAGE spatial gates
            # ---------------------------------------------

            if (
                gate_type == "Dynamic"
                and
                family == "GraphSAGE"
            ):

                dynamic_graph_gate_store.append(
                    alpha
                )


# =========================================================
# 21. Save raw metrics
# =========================================================

metrics = pd.DataFrame(
    metric_rows
)


metrics.to_csv(
    OUTDIR
    / "frozen_expert_moe_raw_metrics.csv",
    index=False,
)


gate_weights = pd.DataFrame(
    gate_rows
)


gate_weights.to_csv(
    OUTDIR
    / "frozen_expert_moe_gate_weights_raw.csv",
    index=False,
)


training_df = pd.DataFrame(
    training_rows
)


training_df.to_csv(
    OUTDIR
    / "frozen_expert_moe_training_summary.csv",
    index=False,
)


# =========================================================
# 22. Performance summary
# =========================================================

summary = (
    metrics
    .groupby(
        [
            "model",
            "family",
            "gene",
        ]
    )
    .agg(

        pearson_mean=(
            "pearson_r",
            "mean"
        ),

        pearson_std=(
            "pearson_r",
            "std"
        ),

        spearman_mean=(
            "spearman_rho",
            "mean"
        ),

        spearman_std=(
            "spearman_rho",
            "std"
        ),

        r2_mean=(
            "r2",
            "mean"
        ),

        r2_std=(
            "r2",
            "std"
        ),

        rmse_mean=(
            "rmse",
            "mean"
        ),

        rmse_std=(
            "rmse",
            "std"
        ),
    )
    .reset_index()
)


summary.to_csv(
    OUTDIR
    / "frozen_expert_moe_summary.csv",
    index=False,
)


# =========================================================
# 23. Gate-weight summary
# =========================================================

gate_summary = (
    gate_weights
    .groupby(
        [
            "model",
            "family",
            "gate_type",
            "gene",
        ]
    )
    .agg(

        low_weight_mean=(
            "low_weight_mean",
            "mean"
        ),

        low_weight_seed_std=(
            "low_weight_mean",
            "std"
        ),

        within_spot_gate_std_mean=(
            "low_weight_std",
            "mean"
        ),

        low_weight_min_mean=(
            "low_weight_min",
            "mean"
        ),

        low_weight_max_mean=(
            "low_weight_max",
            "mean"
        ),

        full_weight_mean=(
            "full_weight_mean",
            "mean"
        ),
    )
    .reset_index()
)


gate_summary.to_csv(
    OUTDIR
    / "frozen_expert_moe_gate_weights_summary.csv",
    index=False,
)


# =========================================================
# 24. Paired comparisons against own experts
# =========================================================

comparison_rows = []


for family in FAMILIES:


    low_name = (
        f"LowExpert_{family}"
    )


    full_name = (
        f"FullExpert_{family}"
    )


    for gate_type in GATE_TYPES:


        moe_name = (
            f"Frozen{gate_type}MoE_"
            f"{family}"
        )


        moe = metrics[
            metrics[
                "model"
            ]
            ==
            moe_name
        ]


        low = metrics[
            metrics[
                "model"
            ]
            ==
            low_name
        ]


        full = metrics[
            metrics[
                "model"
            ]
            ==
            full_name
        ]


        for baseline_name, baseline_df in [
            (
                "LowExpert",
                low
            ),
            (
                "FullExpert",
                full
            ),
        ]:


            merged = moe.merge(
                baseline_df[
                    [
                        "gene",
                        "seed",
                        "pearson_r",
                        "r2",
                        "rmse",
                    ]
                ],
                on=[
                    "gene",
                    "seed",
                ],
                suffixes=(
                    "_moe",
                    "_baseline"
                ),
            )


            for gene in TARGET_GENES:


                g = merged[
                    merged[
                        "gene"
                    ]
                    ==
                    gene
                ]


                delta_r = (
                    g[
                        "pearson_r_moe"
                    ]
                    -
                    g[
                        "pearson_r_baseline"
                    ]
                )


                delta_r2 = (
                    g[
                        "r2_moe"
                    ]
                    -
                    g[
                        "r2_baseline"
                    ]
                )


                delta_rmse = (
                    g[
                        "rmse_moe"
                    ]
                    -
                    g[
                        "rmse_baseline"
                    ]
                )


                comparison_rows.append(
                    {
                        "moe_model":
                            moe_name,

                        "family":
                            family,

                        "baseline":
                            baseline_name,

                        "gene":
                            gene,

                        "delta_pearson_mean":
                            delta_r.mean(),

                        "delta_pearson_std":
                            delta_r.std(),

                        "moe_pearson_wins":
                            int(
                                (
                                    delta_r > 0
                                ).sum()
                            ),

                        "delta_r2_mean":
                            delta_r2.mean(),

                        "delta_r2_std":
                            delta_r2.std(),

                        "moe_r2_wins":
                            int(
                                (
                                    delta_r2 > 0
                                ).sum()
                            ),

                        "delta_rmse_mean":
                            delta_rmse.mean(),

                        "moe_rmse_wins":
                            int(
                                (
                                    delta_rmse < 0
                                ).sum()
                            ),

                        "n_seeds":
                            len(g),
                    }
                )


own_expert_comparison = pd.DataFrame(
    comparison_rows
)


own_expert_comparison.to_csv(
    OUTDIR
    / "frozen_expert_moe_vs_single_experts.csv",
    index=False,
)


# =========================================================
# 25. Compare against Script 12 SeparatePCA50x2
# =========================================================

concat_rows = []


if SCRIPT12_BASELINE.exists():


    baseline12 = pd.read_csv(
        SCRIPT12_BASELINE
    )


    baseline12 = baseline12[
        baseline12[
            "feature_set"
        ]
        ==
        "SeparatePCA50x2"
    ].copy()


    for family in FAMILIES:


        base_family = baseline12[
            baseline12[
                "model"
            ]
            ==
            family
        ]


        for gate_type in GATE_TYPES:


            moe_name = (
                f"Frozen{gate_type}MoE_"
                f"{family}"
            )


            moe = metrics[
                metrics[
                    "model"
                ]
                ==
                moe_name
            ]


            merged = moe.merge(
                base_family[
                    [
                        "gene",
                        "seed",
                        "pearson_r",
                        "r2",
                        "rmse",
                    ]
                ],
                on=[
                    "gene",
                    "seed",
                ],
                suffixes=(
                    "_moe",
                    "_concat"
                ),
            )


            for gene in TARGET_GENES:


                g = merged[
                    merged[
                        "gene"
                    ]
                    ==
                    gene
                ]


                delta_r = (
                    g[
                        "pearson_r_moe"
                    ]
                    -
                    g[
                        "pearson_r_concat"
                    ]
                )


                delta_r2 = (
                    g[
                        "r2_moe"
                    ]
                    -
                    g[
                        "r2_concat"
                    ]
                )


                delta_rmse = (
                    g[
                        "rmse_moe"
                    ]
                    -
                    g[
                        "rmse_concat"
                    ]
                )


                concat_rows.append(
                    {
                        "moe_model":
                            moe_name,

                        "baseline":
                            (
                                family
                                +
                                "_SeparatePCA50x2"
                            ),

                        "gene":
                            gene,

                        "delta_pearson_mean":
                            delta_r.mean(),

                        "delta_pearson_std":
                            delta_r.std(),

                        "moe_pearson_wins":
                            int(
                                (
                                    delta_r > 0
                                ).sum()
                            ),

                        "delta_r2_mean":
                            delta_r2.mean(),

                        "delta_r2_std":
                            delta_r2.std(),

                        "moe_r2_wins":
                            int(
                                (
                                    delta_r2 > 0
                                ).sum()
                            ),

                        "delta_rmse_mean":
                            delta_rmse.mean(),

                        "moe_rmse_wins":
                            int(
                                (
                                    delta_rmse < 0
                                ).sum()
                            ),

                        "n_seeds":
                            len(g),
                    }
                )


concat_comparison = pd.DataFrame(
    concat_rows
)


concat_comparison.to_csv(
    OUTDIR
    / "frozen_expert_moe_vs_concat.csv",
    index=False,
)


# =========================================================
# 26. Ensemble test predictions
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


for j, gene in enumerate(
    TARGET_GENES
):

    prediction_df[
        f"{gene}_true"
    ] = Y_test_raw[
        :,
        j
    ]


for family in FAMILIES:

    for gate_type in GATE_TYPES:


        moe_name = (
            f"Frozen{gate_type}MoE_"
            f"{family}"
        )


        preds = []


        for seed in SEEDS:

            preds.append(
                prediction_store[
                    (
                        moe_name,
                        seed
                    )
                ]
            )


        ensemble_pred = np.mean(
            np.stack(
                preds,
                axis=0,
            ),
            axis=0,
        )


        for j, gene in enumerate(
            TARGET_GENES
        ):

            prediction_df[
                (
                    f"{gene}_"
                    f"{moe_name}"
                )
            ] = ensemble_pred[
                :,
                j
            ]


prediction_df.to_csv(
    OUTDIR
    / "frozen_expert_moe_test_predictions.csv",
    index=False,
)


# =========================================================
# 27. Dynamic GraphSAGE spatial gates
# =========================================================

if len(
    dynamic_graph_gate_store
) > 0:


    dynamic_graph_alpha = np.mean(
        np.stack(
            dynamic_graph_gate_store,
            axis=0,
        ),
        axis=0,
    )


    spatial_gate_df = pd.DataFrame(
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


    for j, gene in enumerate(
        TARGET_GENES
    ):

        spatial_gate_df[
            f"{gene}_low_weight"
        ] = dynamic_graph_alpha[
            :,
            j
        ]


        spatial_gate_df[
            f"{gene}_full_weight"
        ] = (
            1.0
            -
            dynamic_graph_alpha[
                :,
                j
            ]
        )


    spatial_gate_df.to_csv(
        OUTDIR
        / "frozen_dynamic_moe_spatial_gate_weights.csv",
        index=False,
    )


# =========================================================
# 28. Print key results
# =========================================================

print(
    "\n======================================"
)

print(
    "Frozen Expert MoE Summary"
)

print(
    "======================================"
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Prediction-Level Gate Weights"
)

print(
    "alpha = LOW-resolution expert weight"
)

print(
    "1-alpha = FULL-resolution expert weight"
)

print(
    "======================================"
)


print(
    gate_summary.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "MoE vs Own Single-Scale Experts"
)

print(
    "======================================"
)


print(
    own_expert_comparison.to_string(
        index=False
    )
)


if not concat_comparison.empty:


    print(
        "\n======================================"
    )

    print(
        "MoE vs SeparatePCA50x2"
    )

    print(
        "======================================"
    )


    print(
        concat_comparison.to_string(
            index=False
        )
    )


# =========================================================
# 29. Static GraphSAGE interpretable weights
# =========================================================

static_graph = gate_summary[
    gate_summary[
        "model"
    ]
    ==
    "FrozenStaticMoE_GraphSAGE"
].copy()


static_graph[
    "gene"
] = pd.Categorical(
    static_graph[
        "gene"
    ],
    categories=TARGET_GENES,
    ordered=True,
)


static_graph = (
    static_graph
    .sort_values(
        "gene"
    )
)


fig, ax = plt.subplots(
    figsize=(9, 5)
)


x = np.arange(
    len(
        TARGET_GENES
    )
)


ax.bar(
    x,
    static_graph[
        "low_weight_mean"
    ],
    yerr=static_graph[
        "low_weight_seed_std"
    ],
)


ax.axhline(
    0.5,
    linewidth=1,
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    TARGET_GENES
)


ax.set_ylim(
    0,
    1
)


ax.set_ylabel(
    "Low-resolution expert weight"
)


ax.set_title(
    "Frozen-expert static MoE\n"
    "Gene-specific scale preference"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "51_frozen_static_moe_gene_weights.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 30. Dynamic GraphSAGE mean weights
# =========================================================

dynamic_graph = gate_summary[
    gate_summary[
        "model"
    ]
    ==
    "FrozenDynamicMoE_GraphSAGE"
].copy()


dynamic_graph[
    "gene"
] = pd.Categorical(
    dynamic_graph[
        "gene"
    ],
    categories=TARGET_GENES,
    ordered=True,
)


dynamic_graph = (
    dynamic_graph
    .sort_values(
        "gene"
    )
)


fig, ax = plt.subplots(
    figsize=(9, 5)
)


ax.bar(
    x,
    dynamic_graph[
        "low_weight_mean"
    ],
    yerr=dynamic_graph[
        "low_weight_seed_std"
    ],
)


ax.axhline(
    0.5,
    linewidth=1,
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    TARGET_GENES
)


ax.set_ylim(
    0,
    1
)


ax.set_ylabel(
    "Mean low-resolution expert weight"
)


ax.set_title(
    "Frozen-expert dynamic MoE\n"
    "Mean scale preference"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "52_frozen_dynamic_moe_gene_weights.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 31. Expert vs MoE GraphSAGE benchmark
# =========================================================

plot_models = [
    "LowExpert_GraphSAGE",
    "FullExpert_GraphSAGE",
    "FrozenStaticMoE_GraphSAGE",
    "FrozenDynamicMoE_GraphSAGE",
]


graph_summary = summary[
    summary[
        "model"
    ].isin(
        plot_models
    )
].copy()


x = np.arange(
    len(
        TARGET_GENES
    )
)


width = 0.19


fig, ax = plt.subplots(
    figsize=(12, 6)
)


for k, model_name in enumerate(
    plot_models
):

    values = []

    errors = []


    for gene in TARGET_GENES:


        row = graph_summary[
            (
                graph_summary[
                    "model"
                ]
                ==
                model_name
            )
            &
            (
                graph_summary[
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
                plot_models
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
        label=model_name,
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
    "Frozen prediction-level mixture of experts\n"
    "Strict spatial-block GraphSAGE"
)


ax.legend(
    fontsize=8
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "53_frozen_moe_graphsage_benchmark.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 32. Dynamic GraphSAGE spatial gate maps
# =========================================================

if len(
    dynamic_graph_gate_store
) > 0:


    for j, gene in enumerate(
        TARGET_GENES
    ):


        weights = dynamic_graph_alpha[
            :,
            j
        ]


        fig, ax = plt.subplots(
            figsize=(8, 8)
        )


        points = ax.scatter(
            manifest.iloc[
                test_idx
            ][
                "fullres_x"
            ],
            manifest.iloc[
                test_idx
            ][
                "fullres_y"
            ],
            c=weights,
            s=16,
            vmin=0,
            vmax=1,
        )


        ax.invert_yaxis()


        ax.set_aspect(
            "equal"
        )


        ax.set_xlabel(
            "Spatial x"
        )


        ax.set_ylabel(
            "Spatial y"
        )


        ax.set_title(
            f"{gene}: low-resolution expert weight"
        )


        plt.colorbar(
            points,
            ax=ax,
            label=(
                "Low expert weight\n"
                "0 = full-res, 1 = low-res"
            ),
        )


        plt.tight_layout()


        plt.savefig(
            FIGDIR
            / (
                f"{54+j:02d}_"
                f"{gene}_"
                "frozen_dynamic_moe_map.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )


        plt.close()


# =========================================================
# 33. Final report
# =========================================================

print(
    "\nSaved:"
)

print(
    "outputs/frozen_expert_moe_training_summary.csv"
)

print(
    "outputs/frozen_expert_moe_raw_metrics.csv"
)

print(
    "outputs/frozen_expert_moe_summary.csv"
)

print(
    "outputs/frozen_expert_moe_gate_weights_raw.csv"
)

print(
    "outputs/frozen_expert_moe_gate_weights_summary.csv"
)

print(
    "outputs/frozen_expert_moe_vs_single_experts.csv"
)

print(
    "outputs/frozen_expert_moe_vs_concat.csv"
)

print(
    "outputs/frozen_expert_moe_test_predictions.csv"
)

print(
    "outputs/frozen_dynamic_moe_spatial_gate_weights.csv"
)

print(
    "figures/51_frozen_static_moe_gene_weights.png"
)

print(
    "figures/52_frozen_dynamic_moe_gene_weights.png"
)

print(
    "figures/53_frozen_moe_graphsage_benchmark.png"
)

print(
    "figures/54-57 frozen dynamic MoE maps"
)


print(
    "\nFrozen-expert prediction-level MoE complete."
)
