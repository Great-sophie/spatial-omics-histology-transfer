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
from sklearn.metrics import r2_score, mean_squared_error

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

BASELINE_RAW_PATH = Path(
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

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 500
PATIENCE = 60

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

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if (
        np.std(y_true) > 0
        and
        np.std(y_pred) > 0
    ):

        pearson = pearsonr(
            y_true,
            y_pred
        ).statistic

        spearman = spearmanr(
            y_true,
            y_pred
        ).statistic

    else:

        pearson = np.nan
        spearman = np.nan


    r2 = r2_score(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
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
    np.isnan(X_low_raw).sum()
    ==
    0
)

assert (
    np.isnan(X_full_raw).sum()
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


if barcodes != full_barcodes:

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
    list(adata.obs_names)
    ==
    barcodes
)


# =========================================================
# 4. Load fixed spatial split
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
    .set_index("barcode")
    .loc[barcodes]
    .reset_index()
)


assert (
    list(
        split_df["barcode"]
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
        len(TARGET_GENES)
    ),
    dtype=np.float32,
)


print(
    "\nExpression targets:"
)


for j, gene in enumerate(
    TARGET_GENES
):

    values = adata[
        :,
        gene
    ].X

    if sparse.issparse(values):
        values = values.toarray()

    values = (
        np.asarray(values)
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
    ).astype(np.int64)

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
# 8. Block validation from training only
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


fit_idx = train_pool_idx[
    fit_rel
]

val_idx = train_pool_idx[
    val_rel
]


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
    fit_idx
)

edge_val = get_subgraph(
    A,
    val_idx
)

edge_train = get_subgraph(
    A,
    train_pool_idx
)

edge_test = get_subgraph(
    A,
    test_idx
)


# =========================================================
# 9. Scale-specific PCA
#
# Model-selection PCA:
# fit ONLY on fit region.
#
# Final PCA:
# fit ONLY on full training region.
# =========================================================

print(
    "\nFitting scale-specific PCA..."
)


# -------------------------
# Selection PCA
# -------------------------

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


# -------------------------
# Final PCA
# -------------------------

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
    "Low-res PCA variance:",
    float(
        low_pca_final
        .explained_variance_ratio_
        .sum()
    )
)

print(
    "Full-res PCA variance:",
    float(
        full_pca_final
        .explained_variance_ratio_
        .sum()
    )
)


# =========================================================
# 10. Save PCA summary
# =========================================================

pca_summary = pd.DataFrame(
    {
        "scale": [
            "LowRes",
            "FullRes",
        ],

        "raw_dimension": [
            X_low_raw.shape[1],
            X_full_raw.shape[1],
        ],

        "pca_dimension": [
            PCA_DIM,
            PCA_DIM,
        ],

        "train_only_variance": [
            float(
                low_pca_final
                .explained_variance_ratio_
                .sum()
            ),

            float(
                full_pca_final
                .explained_variance_ratio_
                .sum()
            ),
        ],
    }
)


pca_summary.to_csv(
    OUTDIR
    / "gene_gating_pca_summary.csv",
    index=False,
)


# =========================================================
# 11. Shared branch helpers
# =========================================================

class MLPBranch(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.dropout = dropout


    def forward(self, x):

        x = F.relu(
            self.fc1(x)
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = F.relu(
            self.fc2(x)
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return x


class GraphBranch(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.dropout = dropout


    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return x


# =========================================================
# 12. Gene-specific gating model
#
# gate_mode:
#
# static:
#   one learned alpha per gene
#
# dynamic:
#   alpha depends on spot morphology
#
# alpha = LowRes weight
# 1-alpha = FullRes weight
# =========================================================

class GeneSpecificScaleGate(nn.Module):

    def __init__(
        self,
        use_graph,
        gate_mode,
    ):

        super().__init__()

        self.use_graph = use_graph
        self.gate_mode = gate_mode


        if use_graph:

            self.low_branch = GraphBranch(
                PCA_DIM,
                HIDDEN_DIM,
                DROPOUT,
            )

            self.full_branch = GraphBranch(
                PCA_DIM,
                HIDDEN_DIM,
                DROPOUT,
            )

        else:

            self.low_branch = MLPBranch(
                PCA_DIM,
                HIDDEN_DIM,
                DROPOUT,
            )

            self.full_branch = MLPBranch(
                PCA_DIM,
                HIDDEN_DIM,
                DROPOUT,
            )


        if gate_mode == "static":

            # Start from 0.5 / 0.5
            self.gate_logits = nn.Parameter(
                torch.zeros(
                    len(TARGET_GENES)
                )
            )


        elif gate_mode == "dynamic":

            self.gate_net = nn.Sequential(
                nn.Linear(
                    HIDDEN_DIM * 2,
                    HIDDEN_DIM,
                ),

                nn.ReLU(),

                nn.Dropout(
                    DROPOUT
                ),

                nn.Linear(
                    HIDDEN_DIM,
                    len(TARGET_GENES),
                ),
            )


        else:

            raise ValueError(
                gate_mode
            )


        self.gene_heads = nn.ModuleList(
            [
                nn.Linear(
                    HIDDEN_DIM,
                    1
                )
                for _ in TARGET_GENES
            ]
        )


    def forward(
        self,
        x_low,
        x_full,
        edge_index=None,
        return_gates=False,
    ):


        if self.use_graph:

            h_low = self.low_branch(
                x_low,
                edge_index
            )

            h_full = self.full_branch(
                x_full,
                edge_index
            )

        else:

            h_low = self.low_branch(
                x_low
            )

            h_full = self.full_branch(
                x_full
            )


        # ---------------------------------------------
        # Gene-specific gates
        # ---------------------------------------------

        if self.gate_mode == "static":

            alpha_gene = torch.sigmoid(
                self.gate_logits
            )

            alpha = (
                alpha_gene
                .unsqueeze(0)
                .expand(
                    h_low.shape[0],
                    -1
                )
            )


        else:

            gate_input = torch.cat(
                [
                    h_low,
                    h_full,
                ],
                dim=1,
            )

            alpha = torch.sigmoid(
                self.gate_net(
                    gate_input
                )
            )


        # ---------------------------------------------
        # One fused representation per gene
        # ---------------------------------------------

        outputs = []


        for g in range(
            len(TARGET_GENES)
        ):

            a = alpha[
                :,
                g
            ].unsqueeze(1)


            fused = (
                a
                *
                h_low
                +
                (
                    1.0
                    -
                    a
                )
                *
                h_full
            )


            pred = self.gene_heads[
                g
            ](
                fused
            )


            outputs.append(
                pred
            )


        output = torch.cat(
            outputs,
            dim=1,
        )


        if return_gates:

            return (
                output,
                alpha
            )


        return output


# =========================================================
# 13. Model factory
# =========================================================

def make_model(
    model_name,
):

    if model_name == "StaticGate_MLP":

        return GeneSpecificScaleGate(
            use_graph=False,
            gate_mode="static",
        )


    if model_name == "DynamicGate_MLP":

        return GeneSpecificScaleGate(
            use_graph=False,
            gate_mode="dynamic",
        )


    if model_name == "StaticGate_GraphSAGE":

        return GeneSpecificScaleGate(
            use_graph=True,
            gate_mode="static",
        )


    if model_name == "DynamicGate_GraphSAGE":

        return GeneSpecificScaleGate(
            use_graph=True,
            gate_mode="dynamic",
        )


    raise ValueError(
        model_name
    )


# =========================================================
# 14. Model selection
# =========================================================

def select_epoch(
    model_name,
    seed,
):

    set_seed(seed)


    # ---------------------------------------------
    # Scale each representation independently
    # ---------------------------------------------

    low_scaler = StandardScaler()

    full_scaler = StandardScaler()


    low_fit = low_scaler.fit_transform(
        X_low_select[
            fit_idx
        ]
    ).astype(np.float32)


    low_val = low_scaler.transform(
        X_low_select[
            val_idx
        ]
    ).astype(np.float32)


    full_fit = full_scaler.fit_transform(
        X_full_select[
            fit_idx
        ]
    ).astype(np.float32)


    full_val = full_scaler.transform(
        X_full_select[
            val_idx
        ]
    ).astype(np.float32)


    # ---------------------------------------------
    # Standardize genes using fit set only
    # ---------------------------------------------

    y_fit_raw = Y[
        fit_idx
    ]

    y_val_raw = Y[
        val_idx
    ]


    y_mean = y_fit_raw.mean(
        axis=0
    )

    y_std = y_fit_raw.std(
        axis=0
    )

    y_std[
        y_std < 1e-6
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


    # ---------------------------------------------
    # Torch
    # ---------------------------------------------

    low_fit_t = torch.from_numpy(
        low_fit
    ).to(device)

    low_val_t = torch.from_numpy(
        low_val
    ).to(device)


    full_fit_t = torch.from_numpy(
        full_fit
    ).to(device)

    full_val_t = torch.from_numpy(
        full_val
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
        model_name
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


        if "GraphSAGE" in model_name:

            pred = model(
                low_fit_t,
                full_fit_t,
                ef,
            )

        else:

            pred = model(
                low_fit_t,
                full_fit_t,
            )


        loss = F.mse_loss(
            pred,
            y_fit_t
        )


        loss.backward()

        optimizer.step()


        model.eval()


        with torch.no_grad():

            if "GraphSAGE" in model_name:

                val_pred = model(
                    low_val_t,
                    full_val_t,
                    ev,
                )

            else:

                val_pred = model(
                    low_val_t,
                    full_val_t,
                )


            val_loss = F.mse_loss(
                val_pred,
                y_val_t
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
            >= PATIENCE
        ):

            break


    return (
        best_epoch,
        best_val,
    )


# =========================================================
# 15. Final training and prediction
# =========================================================

def train_final(
    model_name,
    seed,
    best_epoch,
):

    set_seed(seed)


    low_scaler = StandardScaler()
    full_scaler = StandardScaler()


    low_train = low_scaler.fit_transform(
        X_low_final[
            train_pool_idx
        ]
    ).astype(np.float32)


    low_test = low_scaler.transform(
        X_low_final[
            test_idx
        ]
    ).astype(np.float32)


    full_train = full_scaler.fit_transform(
        X_full_final[
            train_pool_idx
        ]
    ).astype(np.float32)


    full_test = full_scaler.transform(
        X_full_final[
            test_idx
        ]
    ).astype(np.float32)


    y_train_raw = Y[
        train_pool_idx
    ]


    y_mean = y_train_raw.mean(
        axis=0
    )

    y_std = y_train_raw.std(
        axis=0
    )

    y_std[
        y_std < 1e-6
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


    low_train_t = torch.from_numpy(
        low_train
    ).to(device)

    low_test_t = torch.from_numpy(
        low_test
    ).to(device)


    full_train_t = torch.from_numpy(
        full_train
    ).to(device)

    full_test_t = torch.from_numpy(
        full_test
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
        model_name
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


        if "GraphSAGE" in model_name:

            pred = model(
                low_train_t,
                full_train_t,
                etrain,
            )

        else:

            pred = model(
                low_train_t,
                full_train_t,
            )


        loss = F.mse_loss(
            pred,
            y_train_t
        )


        loss.backward()

        optimizer.step()


    # ---------------------------------------------
    # Test prediction + gates
    # ---------------------------------------------

    model.eval()


    with torch.no_grad():

        if "GraphSAGE" in model_name:

            (
                pred_scaled,
                gates
            ) = model(
                low_test_t,
                full_test_t,
                etest,
                return_gates=True,
            )

        else:

            (
                pred_scaled,
                gates
            ) = model(
                low_test_t,
                full_test_t,
                return_gates=True,
            )


    pred_scaled = (
        pred_scaled
        .cpu()
        .numpy()
    )


    gates = (
        gates
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


    return (
        pred.astype(np.float32),
        gates.astype(np.float32),
    )


# =========================================================
# 16. Run gated models
# =========================================================

MODEL_NAMES = [
    "StaticGate_MLP",
    "DynamicGate_MLP",
    "StaticGate_GraphSAGE",
    "DynamicGate_GraphSAGE",
]


metric_rows = []
gate_rows = []

prediction_store = {
    model_name: []
    for model_name in MODEL_NAMES
}

gate_store = {
    model_name: []
    for model_name in MODEL_NAMES
}


for model_name in MODEL_NAMES:

    print(
        "\n======================================"
    )

    print(
        model_name
    )

    print(
        "======================================"
    )


    for seed in SEEDS:


        best_epoch, val_loss = (
            select_epoch(
                model_name,
                seed,
            )
        )


        print(
            model_name,
            f"seed={seed}",
            f"epoch={best_epoch}",
            f"val={val_loss:.4f}",
        )


        pred, gates = train_final(
            model_name,
            seed,
            best_epoch,
        )


        prediction_store[
            model_name
        ].append(
            pred
        )


        gate_store[
            model_name
        ].append(
            gates
        )


        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

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


            metric_rows.append(
                {
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


            # -----------------------------------------
            # Gate summary
            #
            # alpha = low-res weight
            # -----------------------------------------

            gene_gate = gates[
                :,
                j
            ]


            gate_rows.append(
                {
                    "model":
                        model_name,

                    "gene":
                        gene,

                    "seed":
                        seed,

                    "low_weight_mean":
                        float(
                            gene_gate.mean()
                        ),

                    "low_weight_std":
                        float(
                            gene_gate.std()
                        ),

                    "low_weight_min":
                        float(
                            gene_gate.min()
                        ),

                    "low_weight_max":
                        float(
                            gene_gate.max()
                        ),

                    "full_weight_mean":
                        float(
                            1.0
                            -
                            gene_gate.mean()
                        ),
                }
            )


# =========================================================
# 17. Save raw metrics
# =========================================================

metrics = pd.DataFrame(
    metric_rows
)


metrics.to_csv(
    OUTDIR
    / "gene_specific_gating_raw_metrics.csv",
    index=False,
)


gates_df = pd.DataFrame(
    gate_rows
)


gates_df.to_csv(
    OUTDIR
    / "gene_specific_gating_weights_raw.csv",
    index=False,
)


# =========================================================
# 18. Metrics summary
# =========================================================

summary = (
    metrics
    .groupby(
        [
            "model",
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

        epoch_mean=(
            "best_epoch",
            "mean"
        ),
    )
    .reset_index()
)


summary.to_csv(
    OUTDIR
    / "gene_specific_gating_summary.csv",
    index=False,
)


# =========================================================
# 19. Gate-weight summary
# =========================================================

gate_summary = (
    gates_df
    .groupby(
        [
            "model",
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

        full_weight_mean=(
            "full_weight_mean",
            "mean"
        ),
    )
    .reset_index()
)


gate_summary.to_csv(
    OUTDIR
    / "gene_specific_gating_weights_summary.csv",
    index=False,
)


# =========================================================
# 20. Load SeparatePCA50x2 baseline
#
# This is the exact 100-D scale-specific baseline
# from script 12.
# =========================================================

if not BASELINE_RAW_PATH.exists():

    raise FileNotFoundError(
        "Baseline raw metrics missing: "
        "run script 12 first."
    )


baseline = pd.read_csv(
    BASELINE_RAW_PATH
)


baseline = baseline[
    (
        baseline[
            "feature_set"
        ]
        ==
        "SeparatePCA50x2"
    )
    &
    (
        baseline[
            "model"
        ].isin(
            [
                "MLP",
                "GraphSAGE",
            ]
        )
    )
].copy()


# =========================================================
# 21. Paired seed comparison vs concat baseline
# =========================================================

comparison_rows = []


for gated_model in MODEL_NAMES:


    if "GraphSAGE" in gated_model:

        baseline_model = "GraphSAGE"

    else:

        baseline_model = "MLP"


    gated_sub = metrics[
        metrics[
            "model"
        ]
        ==
        gated_model
    ]


    base_sub = baseline[
        baseline[
            "model"
        ]
        ==
        baseline_model
    ]


    merged = gated_sub.merge(
        base_sub[
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
            "_gated",
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
                "pearson_r_gated"
            ]
            -
            g[
                "pearson_r_baseline"
            ]
        )


        delta_r2 = (
            g[
                "r2_gated"
            ]
            -
            g[
                "r2_baseline"
            ]
        )


        delta_rmse = (
            g[
                "rmse_gated"
            ]
            -
            g[
                "rmse_baseline"
            ]
        )


        comparison_rows.append(
            {
                "gated_model":
                    gated_model,

                "baseline_model":
                    (
                        baseline_model
                        +
                        "_SeparatePCA50x2"
                    ),

                "gene":
                    gene,

                "delta_pearson_mean":
                    delta_r.mean(),

                "delta_pearson_std":
                    delta_r.std(),

                "gated_pearson_wins":
                    int(
                        (
                            delta_r > 0
                        ).sum()
                    ),

                "delta_r2_mean":
                    delta_r2.mean(),

                "delta_r2_std":
                    delta_r2.std(),

                "gated_r2_wins":
                    int(
                        (
                            delta_r2 > 0
                        ).sum()
                    ),

                "delta_rmse_mean":
                    delta_rmse.mean(),

                "gated_rmse_wins":
                    int(
                        (
                            delta_rmse < 0
                        ).sum()
                    ),

                "n_seeds":
                    len(g),
            }
        )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    OUTDIR
    / "gene_specific_gating_vs_concat.csv",
    index=False,
)


# =========================================================
# 22. Ensemble predictions
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
    ] = Y[
        test_idx,
        j
    ]


for model_name in MODEL_NAMES:


    ensemble_pred = np.mean(
        np.stack(
            prediction_store[
                model_name
            ],
            axis=0,
        ),
        axis=0,
    )


    for j, gene in enumerate(
        TARGET_GENES
    ):

        prediction_df[
            f"{gene}_{model_name}"
        ] = ensemble_pred[
            :,
            j
        ]


prediction_df.to_csv(
    OUTDIR
    / "gene_specific_gating_test_predictions.csv",
    index=False,
)


# =========================================================
# 23. Dynamic GraphSAGE spot-level gates
# =========================================================

dynamic_graph_gates = np.mean(
    np.stack(
        gate_store[
            "DynamicGate_GraphSAGE"
        ],
        axis=0,
    ),
    axis=0,
)


gate_spot_df = pd.DataFrame(
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

    gate_spot_df[
        f"{gene}_low_weight"
    ] = dynamic_graph_gates[
        :,
        j
    ]


    gate_spot_df[
        f"{gene}_full_weight"
    ] = (
        1.0
        -
        dynamic_graph_gates[
            :,
            j
        ]
    )


gate_spot_df.to_csv(
    OUTDIR
    / "dynamic_graphsage_spatial_gate_weights.csv",
    index=False,
)


# =========================================================
# 24. Print results
# =========================================================

print(
    "\n======================================"
)

print(
    "Gene-specific gating summary"
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
    "Learned scale weights"
)

print(
    "Low weight = preference for coarse/low-res"
)

print(
    "Full weight = preference for fine/full-res"
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
    "Gating vs SeparatePCA50x2 baseline"
)

print(
    "======================================"
)


print(
    comparison.to_string(
        index=False
    )
)


# =========================================================
# 25. Static gate weight plot
# =========================================================

static_graph = gate_summary[
    gate_summary[
        "model"
    ]
    ==
    "StaticGate_GraphSAGE"
].copy()


gene_order = TARGET_GENES


static_graph[
    "gene"
] = pd.Categorical(
    static_graph[
        "gene"
    ],
    categories=gene_order,
    ordered=True,
)


static_graph = (
    static_graph
    .sort_values("gene")
)


x = np.arange(
    len(
        static_graph
    )
)


fig, ax = plt.subplots(
    figsize=(9, 5)
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
    static_graph[
        "gene"
    ]
)


ax.set_ylim(
    0,
    1
)


ax.set_ylabel(
    "Learned low-resolution weight"
)


ax.set_title(
    "Static gene-specific scale preference\n"
    "GraphSAGE"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "43_static_gene_scale_weights_graphsage.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 26. Dynamic gate mean plot
# =========================================================

dynamic_graph = gate_summary[
    gate_summary[
        "model"
    ]
    ==
    "DynamicGate_GraphSAGE"
].copy()


dynamic_graph[
    "gene"
] = pd.Categorical(
    dynamic_graph[
        "gene"
    ],
    categories=gene_order,
    ordered=True,
)


dynamic_graph = (
    dynamic_graph
    .sort_values("gene")
)


fig, ax = plt.subplots(
    figsize=(9, 5)
)


ax.bar(
    np.arange(
        len(
            dynamic_graph
        )
    ),
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
    np.arange(
        len(
            dynamic_graph
        )
    )
)


ax.set_xticklabels(
    dynamic_graph[
        "gene"
    ]
)


ax.set_ylim(
    0,
    1
)


ax.set_ylabel(
    "Mean low-resolution weight"
)


ax.set_title(
    "Dynamic gene-specific scale preference\n"
    "GraphSAGE"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "44_dynamic_gene_scale_weights_graphsage.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 27. GraphSAGE performance comparison
# =========================================================

graph_models = [
    "StaticGate_GraphSAGE",
    "DynamicGate_GraphSAGE",
]


# baseline mean
baseline_graph = (
    baseline[
        baseline[
            "model"
        ]
        ==
        "GraphSAGE"
    ]
    .groupby(
        "gene"
    )[
        "pearson_r"
    ]
    .mean()
)


x = np.arange(
    len(
        TARGET_GENES
    )
)


width = 0.25


fig, ax = plt.subplots(
    figsize=(10, 6)
)


baseline_values = [
    baseline_graph.loc[
        gene
    ]
    for gene in TARGET_GENES
]


ax.bar(
    x - width,
    baseline_values,
    width,
    label="SeparatePCA50x2",
)


for offset, model_name in [
    (
        0,
        "StaticGate_GraphSAGE"
    ),

    (
        width,
        "DynamicGate_GraphSAGE"
    ),
]:

    values = []

    errors = []


    for gene in TARGET_GENES:

        row = summary[
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
    "Gene-specific scale gating\n"
    "Strict spatial-block GraphSAGE"
)

ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "45_gene_specific_gating_graphsage_benchmark.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 28. Gain over concat baseline
# =========================================================

delta_plot = comparison[
    comparison[
        "gated_model"
    ]
    ==
    "DynamicGate_GraphSAGE"
].copy()


delta_plot[
    "gene"
] = pd.Categorical(
    delta_plot[
        "gene"
    ],
    categories=TARGET_GENES,
    ordered=True,
)


delta_plot = (
    delta_plot
    .sort_values("gene")
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    delta_plot[
        "gene"
    ].astype(str),
    delta_plot[
        "delta_pearson_mean"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r\n"
    "(Dynamic gating - concat baseline)"
)


ax.set_title(
    "Benefit of dynamic gene-specific gating"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "46_dynamic_gating_gain_over_concat.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 29-32. Spatial dynamic gate maps
# =========================================================

for j, gene in enumerate(
    TARGET_GENES
):

    weights = dynamic_graph_gates[
        :,
        j
    ]


    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    sc = ax.scatter(
        gate_spot_df[
            "fullres_x"
        ],
        gate_spot_df[
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
        f"{gene}: learned low-resolution weight"
    )


    plt.colorbar(
        sc,
        ax=ax,
        label=(
            "Low-res weight\n"
            "0 = full-res, 1 = low-res"
        ),
    )


    plt.tight_layout()


    plt.savefig(
        FIGDIR
        / (
            f"{47+j:02d}_"
            f"{gene}_"
            "dynamic_scale_gate_map.png"
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
    "outputs/gene_gating_pca_summary.csv"
)

print(
    "outputs/gene_specific_gating_raw_metrics.csv"
)

print(
    "outputs/gene_specific_gating_summary.csv"
)

print(
    "outputs/gene_specific_gating_weights_raw.csv"
)

print(
    "outputs/gene_specific_gating_weights_summary.csv"
)

print(
    "outputs/gene_specific_gating_vs_concat.csv"
)

print(
    "outputs/gene_specific_gating_test_predictions.csv"
)

print(
    "outputs/dynamic_graphsage_spatial_gate_weights.csv"
)

print(
    "figures/43_static_gene_scale_weights_graphsage.png"
)

print(
    "figures/44_dynamic_gene_scale_weights_graphsage.png"
)

print(
    "figures/45_gene_specific_gating_graphsage_benchmark.png"
)

print(
    "figures/46_dynamic_gating_gain_over_concat.png"
)

print(
    "figures/47-50 dynamic spatial gate maps"
)


print(
    "\nGene-specific scale gating experiment complete."
)
