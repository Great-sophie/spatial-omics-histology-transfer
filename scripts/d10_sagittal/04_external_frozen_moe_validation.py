import copy
import random
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

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

ROOT = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D10_mouse_brain_external_validation"
)

OUT = ROOT / "outputs"
FIG = ROOT / "figures"

ADATA_PATH = (
    OUT
    / "V1_Mouse_Brain_Sagittal_Anterior_external.h5ad"
)

MANIFEST_PATH = (
    OUT
    / "external_multiscale_patch_manifest.csv"
)

LOW_PATH = (
    OUT
    / "dinov2_vits14_external_lowres_embeddings.npy"
)

FULL_PATH = (
    OUT
    / "dinov2_vits14_external_fullres_embeddings.npy"
)


TARGET_GENES = [
    "Camk2n1",
    "Slc17a7",
    "Plp1",
    "Cldn11",
]


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


PCA_DIM = 50

HIDDEN_DIM = 64
DROPOUT = 0.20

LR = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 500
PATIENCE = 60


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

def evaluate(
    y_true,
    y_pred,
):

    if (
        np.std(y_true) > 0
        and
        np.std(y_pred) > 0
    ):

        r = pearsonr(
            y_true,
            y_pred
        ).statistic

        rho = spearmanr(
            y_true,
            y_pred
        ).statistic

    else:

        r = np.nan
        rho = np.nan


    return {
        "pearson_r":
            float(r),

        "spearman_rho":
            float(rho),

        "r2":
            float(
                r2_score(
                    y_true,
                    y_pred
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred
                    )
                )
            ),
    }


# =========================================================
# Load data
# =========================================================

print("\nLoading D10 data...")


adata = sc.read_h5ad(
    ADATA_PATH
)


manifest = pd.read_csv(
    MANIFEST_PATH
)


manifest[
    "barcode"
] = (
    manifest[
        "barcode"
    ].astype(str)
)


X_low_raw = np.load(
    LOW_PATH
).astype(np.float32)


X_full_raw = np.load(
    FULL_PATH
).astype(np.float32)


print(
    "AnnData:",
    adata.shape
)

print(
    "Low:",
    X_low_raw.shape
)

print(
    "Full:",
    X_full_raw.shape
)


# =========================================================
# Alignment
# =========================================================

barcodes = (
    manifest[
        "barcode"
    ]
    .tolist()
)


adata = adata[
    barcodes
].copy()


if (
    list(
        adata.obs_names
    )
    !=
    barcodes
):

    raise RuntimeError(
        "Barcode alignment failed."
    )


if (
    len(adata)
    !=
    X_low_raw.shape[0]
    or
    len(adata)
    !=
    X_full_raw.shape[0]
):

    raise RuntimeError(
        "Embedding row count mismatch."
    )


print(
    "Barcode alignment OK:",
    len(barcodes)
)


# =========================================================
# Targets
# =========================================================

Y = np.zeros(
    (
        adata.n_obs,
        len(TARGET_GENES)
    ),
    dtype=np.float32,
)


print("\nTarget genes:")


for j, gene in enumerate(
    TARGET_GENES
):

    x = adata[
        :,
        gene
    ].X


    if sparse.issparse(x):

        x = x.toarray()


    x = (
        np.asarray(x)
        .ravel()
        .astype(np.float32)
    )


    Y[
        :,
        j
    ] = x


    print(
        gene,
        "mean=",
        round(
            float(
                x.mean()
            ),
            4
        ),
        "std=",
        round(
            float(
                x.std()
            ),
            4
        )
    )


# =========================================================
# Spatial graph
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


def induced_edge_index(
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
            sub.col
        ]
    ).astype(np.int64)


    return torch.from_numpy(
        edge_index
    )


# =========================================================
# Spatial blocks
#
# 4 x 4 quantile grid.
# Split is deterministic and independent of expression.
# =========================================================

x_coord = (
    manifest[
        "fullres_x"
    ]
    .astype(float)
    .to_numpy()
)

y_coord = (
    manifest[
        "fullres_y"
    ]
    .astype(float)
    .to_numpy()
)


x_rank = pd.Series(
    x_coord
).rank(
    method="first"
)


y_rank = pd.Series(
    y_coord
).rank(
    method="first"
)


x_bin = pd.qcut(
    x_rank,
    q=4,
    labels=False
).to_numpy()


y_bin = pd.qcut(
    y_rank,
    q=4,
    labels=False
).to_numpy()


spatial_block = (
    y_bin * 4
    +
    x_bin
).astype(int)


all_idx = np.arange(
    adata.n_obs
)


# =========================================================
# Fixed block-level test split
# =========================================================

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=2026,
)


train_rel, test_rel = next(
    outer_split.split(
        all_idx,
        groups=spatial_block
    )
)


train_idx = all_idx[
    train_rel
]

test_idx = all_idx[
    test_rel
]


train_blocks = spatial_block[
    train_idx
]


# =========================================================
# Fixed block-level validation split
# =========================================================

inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)


fit_rel, val_rel = next(
    inner_split.split(
        train_idx,
        groups=train_blocks
    )
)


fit_idx = train_idx[
    fit_rel
]

val_idx = train_idx[
    val_rel
]


print(
    "\nSpatial split:"
)

print(
    "Fit:",
    len(fit_idx)
)

print(
    "Validation:",
    len(val_idx)
)

print(
    "Train total:",
    len(train_idx)
)

print(
    "Test:",
    len(test_idx)
)

print(
    "Fit blocks:",
    sorted(
        np.unique(
            spatial_block[
                fit_idx
            ]
        ).tolist()
    )
)

print(
    "Validation blocks:",
    sorted(
        np.unique(
            spatial_block[
                val_idx
            ]
        ).tolist()
    )
)

print(
    "Test blocks:",
    sorted(
        np.unique(
            spatial_block[
                test_idx
            ]
        ).tolist()
    )
)


# =========================================================
# Save split manifest
# =========================================================

split_labels = np.full(
    adata.n_obs,
    "fit",
    dtype=object,
)

split_labels[
    val_idx
] = "validation"

split_labels[
    test_idx
] = "test"


split_df = pd.DataFrame(
    {
        "barcode":
            barcodes,

        "fullres_x":
            x_coord,

        "fullres_y":
            y_coord,

        "spatial_block":
            spatial_block,

        "split":
            split_labels,
    }
)


split_df.to_csv(
    OUT
    / "external_spatial_split_manifest.csv",
    index=False,
)


# =========================================================
# GraphSAGE
# =========================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
    ):

        super().__init__()


        self.conv1 = SAGEConv(
            input_dim,
            HIDDEN_DIM
        )


        self.conv2 = SAGEConv(
            HIDDEN_DIM,
            HIDDEN_DIM
        )


        self.head = nn.Linear(
            HIDDEN_DIM,
            len(TARGET_GENES)
        )


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
            p=DROPOUT,
            training=self.training
        )


        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )


        return self.head(x)


# =========================================================
# Model selection
# =========================================================

def select_epochs(
    X_fit,
    X_val,
    y_fit,
    y_val,
    edge_fit,
    edge_val,
    seed,
):

    set_seed(seed)


    X_fit_t = torch.from_numpy(
        X_fit
    ).to(device)


    X_val_t = torch.from_numpy(
        X_val
    ).to(device)


    y_fit_t = torch.from_numpy(
        y_fit
    ).to(device)


    y_val_t = torch.from_numpy(
        y_val
    ).to(device)


    edge_fit = edge_fit.to(
        device
    )


    edge_val = edge_val.to(
        device
    )


    model = GraphSAGE(
        X_fit.shape[1]
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    best_loss = np.inf
    best_epoch = 1
    best_state = None
    bad = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()


        pred = model(
            X_fit_t,
            edge_fit
        )


        loss = F.mse_loss(
            pred,
            y_fit_t
        )


        loss.backward()
        optimizer.step()


        model.eval()


        with torch.no_grad():

            val_pred = model(
                X_val_t,
                edge_val
            )


            val_loss = F.mse_loss(
                val_pred,
                y_val_t
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

            bad = 0

        else:

            bad += 1


        if bad >= PATIENCE:

            break


    model.load_state_dict(
        best_state
    )

    model.eval()


    return (
        best_epoch,
        best_loss,
        model
    )


# =========================================================
# Fixed-epoch final training
# =========================================================

def train_fixed(
    X_train,
    y_train,
    edge_train,
    epochs,
    seed,
):

    set_seed(seed)


    X_t = torch.from_numpy(
        X_train
    ).to(device)


    y_t = torch.from_numpy(
        y_train
    ).to(device)


    edge_t = edge_train.to(
        device
    )


    model = GraphSAGE(
        X_train.shape[1]
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    for _ in range(
        int(epochs)
    ):

        model.train()

        optimizer.zero_grad()


        pred = model(
            X_t,
            edge_t
        )


        loss = F.mse_loss(
            pred,
            y_t
        )


        loss.backward()
        optimizer.step()


    model.eval()


    return model


# =========================================================
# Prediction helper
# =========================================================

def predict(
    model,
    X,
    edge_index,
):

    X_t = torch.from_numpy(
        X
    ).to(device)


    edge_t = edge_index.to(
        device
    )


    model.eval()


    with torch.no_grad():

        pred = model(
            X_t,
            edge_t
        )


    return (
        pred
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# =========================================================
# Static alpha
#
# Minimize validation MSE:
#
# pred = alpha * low + (1-alpha) * full
#
# alpha constrained to [0,1].
# =========================================================

def fit_static_alpha(
    y_true,
    pred_low,
    pred_full,
):

    alphas = np.zeros(
        len(TARGET_GENES),
        dtype=np.float32
    )


    for g in range(
        len(TARGET_GENES)
    ):

        low = pred_low[
            :,
            g
        ]

        full = pred_full[
            :,
            g
        ]

        y = y_true[
            :,
            g
        ]


        d = (
            low - full
        )


        numerator = np.sum(
            d
            *
            (
                y - full
            )
        )


        denominator = np.sum(
            d * d
        )


        if denominator < 1e-12:

            alpha = 0.5

        else:

            alpha = (
                numerator
                /
                denominator
            )


        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0
            )
        )


        alphas[
            g
        ] = alpha


    return alphas


# =========================================================
# Selection preprocessing
# =========================================================

print(
    "\nFitting selection PCA..."
)


pca_low_sel = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_full_sel = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_low_sel.fit(
    X_low_raw[
        fit_idx
    ]
)


pca_full_sel.fit(
    X_full_raw[
        fit_idx
    ]
)


low_sel_all = (
    pca_low_sel.transform(
        X_low_raw
    )
    .astype(np.float32)
)


full_sel_all = (
    pca_full_sel.transform(
        X_full_raw
    )
    .astype(np.float32)
)


# Low branch scaler

sc_low_sel = StandardScaler()

low_fit = sc_low_sel.fit_transform(
    low_sel_all[
        fit_idx
    ]
).astype(np.float32)

low_val = sc_low_sel.transform(
    low_sel_all[
        val_idx
    ]
).astype(np.float32)


# Full branch scaler

sc_full_sel = StandardScaler()

full_fit = sc_full_sel.fit_transform(
    full_sel_all[
        fit_idx
    ]
).astype(np.float32)

full_val = sc_full_sel.transform(
    full_sel_all[
        val_idx
    ]
).astype(np.float32)


# Separate PCA50x2 concat

concat_sel_all = np.concatenate(
    [
        low_sel_all,
        full_sel_all
    ],
    axis=1
)


sc_concat_sel = StandardScaler()

concat_fit = sc_concat_sel.fit_transform(
    concat_sel_all[
        fit_idx
    ]
).astype(np.float32)

concat_val = sc_concat_sel.transform(
    concat_sel_all[
        val_idx
    ]
).astype(np.float32)


# =========================================================
# Selection expression normalization
# =========================================================

Y_fit_raw = Y[
    fit_idx
]

Y_val_raw = Y[
    val_idx
]


Y_mean_sel = Y_fit_raw.mean(
    axis=0
)

Y_std_sel = Y_fit_raw.std(
    axis=0
)

Y_std_sel[
    Y_std_sel < 1e-6
] = 1.0


Y_fit = (
    (
        Y_fit_raw
        -
        Y_mean_sel
    )
    /
    Y_std_sel
).astype(np.float32)


Y_val = (
    (
        Y_val_raw
        -
        Y_mean_sel
    )
    /
    Y_std_sel
).astype(np.float32)


# =========================================================
# Selection graphs
# =========================================================

edge_fit = induced_edge_index(
    A,
    fit_idx
)

edge_val = induced_edge_index(
    A,
    val_idx
)

edge_train = induced_edge_index(
    A,
    train_idx
)

edge_test = induced_edge_index(
    A,
    test_idx
)


print(
    "Fit graph edges:",
    edge_fit.shape[1]
)

print(
    "Val graph edges:",
    edge_val.shape[1]
)

print(
    "Train graph edges:",
    edge_train.shape[1]
)

print(
    "Test graph edges:",
    edge_test.shape[1]
)


# =========================================================
# Final preprocessing
# =========================================================

print(
    "\nFitting final train-only PCA..."
)


pca_low_final = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_full_final = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_low_final.fit(
    X_low_raw[
        train_idx
    ]
)


pca_full_final.fit(
    X_full_raw[
        train_idx
    ]
)


low_final_all = (
    pca_low_final.transform(
        X_low_raw
    )
    .astype(np.float32)
)


full_final_all = (
    pca_full_final.transform(
        X_full_raw
    )
    .astype(np.float32)
)


sc_low_final = StandardScaler()

low_train = sc_low_final.fit_transform(
    low_final_all[
        train_idx
    ]
).astype(np.float32)

low_test = sc_low_final.transform(
    low_final_all[
        test_idx
    ]
).astype(np.float32)


sc_full_final = StandardScaler()

full_train = sc_full_final.fit_transform(
    full_final_all[
        train_idx
    ]
).astype(np.float32)

full_test = sc_full_final.transform(
    full_final_all[
        test_idx
    ]
).astype(np.float32)


concat_final_all = np.concatenate(
    [
        low_final_all,
        full_final_all
    ],
    axis=1
)


sc_concat_final = StandardScaler()

concat_train = sc_concat_final.fit_transform(
    concat_final_all[
        train_idx
    ]
).astype(np.float32)

concat_test = sc_concat_final.transform(
    concat_final_all[
        test_idx
    ]
).astype(np.float32)


Y_train_raw = Y[
    train_idx
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


print(
    "Low PCA variance:",
    float(
        pca_low_final
        .explained_variance_ratio_
        .sum()
    )
)

print(
    "Full PCA variance:",
    float(
        pca_full_final
        .explained_variance_ratio_
        .sum()
    )
)


# =========================================================
# Experiment
# =========================================================

metric_rows = []
alpha_rows = []
training_rows = []
prediction_rows = []


for seed in SEEDS:

    print(
        "\n========================================"
    )

    print(
        "Seed:",
        seed
    )

    print(
        "========================================"
    )


    # -----------------------------------------------------
    # Select low expert
    # -----------------------------------------------------

    (
        low_epoch,
        low_val_loss,
        low_sel_model
    ) = select_epochs(
        low_fit,
        low_val,
        Y_fit,
        Y_val,
        edge_fit,
        edge_val,
        seed
    )


    # -----------------------------------------------------
    # Select full expert
    # -----------------------------------------------------

    (
        full_epoch,
        full_val_loss,
        full_sel_model
    ) = select_epochs(
        full_fit,
        full_val,
        Y_fit,
        Y_val,
        edge_fit,
        edge_val,
        seed
    )


    # -----------------------------------------------------
    # Select SeparatePCA50x2 GraphSAGE
    # -----------------------------------------------------

    (
        concat_epoch,
        concat_val_loss,
        _
    ) = select_epochs(
        concat_fit,
        concat_val,
        Y_fit,
        Y_val,
        edge_fit,
        edge_val,
        seed
    )


    # -----------------------------------------------------
    # Validation expert predictions
    # -----------------------------------------------------

    low_val_pred = predict(
        low_sel_model,
        low_val,
        edge_val
    )


    full_val_pred = predict(
        full_sel_model,
        full_val,
        edge_val
    )


    # -----------------------------------------------------
    # Learn STATIC gene-specific alpha on validation only
    # -----------------------------------------------------

    alpha = fit_static_alpha(
        Y_val,
        low_val_pred,
        full_val_pred
    )


    print(
        "Low epoch:",
        low_epoch
    )

    print(
        "Full epoch:",
        full_epoch
    )

    print(
        "Concat epoch:",
        concat_epoch
    )

    print(
        "alpha:",
        {
            gene:
                round(
                    float(
                        alpha[j]
                    ),
                    4
                )
            for j, gene
            in enumerate(
                TARGET_GENES
            )
        }
    )


    # -----------------------------------------------------
    # Train final models on ALL training regions
    # -----------------------------------------------------

    low_model = train_fixed(
        low_train,
        Y_train,
        edge_train,
        low_epoch,
        seed
    )


    full_model = train_fixed(
        full_train,
        Y_train,
        edge_train,
        full_epoch,
        seed
    )


    concat_model = train_fixed(
        concat_train,
        Y_train,
        edge_train,
        concat_epoch,
        seed
    )


    # -----------------------------------------------------
    # Strict test prediction
    # -----------------------------------------------------

    low_test_std = predict(
        low_model,
        low_test,
        edge_test
    )


    full_test_std = predict(
        full_model,
        full_test,
        edge_test
    )


    concat_test_std = predict(
        concat_model,
        concat_test,
        edge_test
    )


    moe_test_std = (
        alpha[
            None,
            :
        ]
        *
        low_test_std
        +
        (
            1.0
            -
            alpha[
                None,
                :
            ]
        )
        *
        full_test_std
    )


    # -----------------------------------------------------
    # Back to normalized/log-expression units
    # -----------------------------------------------------

    low_pred = (
        low_test_std
        *
        Y_std_final
        +
        Y_mean_final
    )


    full_pred = (
        full_test_std
        *
        Y_std_final
        +
        Y_mean_final
    )


    concat_pred = (
        concat_test_std
        *
        Y_std_final
        +
        Y_mean_final
    )


    moe_pred = (
        moe_test_std
        *
        Y_std_final
        +
        Y_mean_final
    )


    predictions = {
        "LowExpert_GraphSAGE":
            low_pred,

        "FullExpert_GraphSAGE":
            full_pred,

        "SeparatePCA50x2_GraphSAGE":
            concat_pred,

        "FrozenStaticMoE_GraphSAGE":
            moe_pred,
    }


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    for model_name, pred in predictions.items():

        for j, gene in enumerate(
            TARGET_GENES
        ):

            m = evaluate(
                Y_test_raw[
                    :,
                    j
                ],
                pred[
                    :,
                    j
                ]
            )


            metric_rows.append(
                {
                    "model":
                        model_name,

                    "gene":
                        gene,

                    "seed":
                        seed,

                    **m
                }
            )


    # -----------------------------------------------------
    # Alpha
    # -----------------------------------------------------

    for j, gene in enumerate(
        TARGET_GENES
    ):

        alpha_rows.append(
            {
                "seed":
                    seed,

                "gene":
                    gene,

                "low_weight":
                    float(
                        alpha[j]
                    ),

                "full_weight":
                    float(
                        1.0
                        -
                        alpha[j]
                    )
            }
        )


    # -----------------------------------------------------
    # Training metadata
    # -----------------------------------------------------

    training_rows.append(
        {
            "seed":
                seed,

            "low_epoch":
                low_epoch,

            "low_val_loss":
                low_val_loss,

            "full_epoch":
                full_epoch,

            "full_val_loss":
                full_val_loss,

            "concat_epoch":
                concat_epoch,

            "concat_val_loss":
                concat_val_loss,
        }
    )


    # -----------------------------------------------------
    # Store test predictions
    # -----------------------------------------------------

    for local_i, global_i in enumerate(
        test_idx
    ):

        row = {
            "barcode":
                barcodes[
                    global_i
                ],

            "seed":
                seed,
        }


        for j, gene in enumerate(
            TARGET_GENES
        ):

            row[
                f"{gene}_true"
            ] = float(
                Y_test_raw[
                    local_i,
                    j
                ]
            )


            row[
                f"{gene}_moe"
            ] = float(
                moe_pred[
                    local_i,
                    j
                ]
            )


        prediction_rows.append(
            row
        )


# =========================================================
# Save raw
# =========================================================

metrics = pd.DataFrame(
    metric_rows
)


alphas = pd.DataFrame(
    alpha_rows
)


training = pd.DataFrame(
    training_rows
)


predictions = pd.DataFrame(
    prediction_rows
)


metrics.to_csv(
    OUT
    / "external_replication_raw_metrics.csv",
    index=False
)


alphas.to_csv(
    OUT
    / "external_replication_gate_weights_raw.csv",
    index=False
)


training.to_csv(
    OUT
    / "external_replication_training.csv",
    index=False
)


predictions.to_csv(
    OUT
    / "external_replication_test_predictions.csv",
    index=False
)


# =========================================================
# Summary
# =========================================================

summary = (
    metrics
    .groupby(
        [
            "model",
            "gene"
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
        )
    )
    .reset_index()
)


summary.to_csv(
    OUT
    / "external_replication_summary.csv",
    index=False
)


# =========================================================
# Gate summary
# =========================================================

alpha_summary = (
    alphas
    .groupby(
        "gene"
    )
    .agg(
        low_weight_mean=(
            "low_weight",
            "mean"
        ),

        low_weight_std=(
            "low_weight",
            "std"
        ),

        full_weight_mean=(
            "full_weight",
            "mean"
        ),

        full_weight_std=(
            "full_weight",
            "std"
        )
    )
    .reset_index()
)


alpha_summary.to_csv(
    OUT
    / "external_replication_gate_weights_summary.csv",
    index=False
)


# =========================================================
# Paired comparisons
# =========================================================

comparison_rows = []


moe = metrics[
    metrics[
        "model"
    ]
    ==
    "FrozenStaticMoE_GraphSAGE"
]


for baseline_name in [
    "LowExpert_GraphSAGE",
    "FullExpert_GraphSAGE",
    "SeparatePCA50x2_GraphSAGE",
]:

    base = metrics[
        metrics[
            "model"
        ]
        ==
        baseline_name
    ]


    merged = moe.merge(
        base[
            [
                "gene",
                "seed",
                "pearson_r",
                "r2",
                "rmse"
            ]
        ],
        on=[
            "gene",
            "seed"
        ],
        suffixes=(
            "_moe",
            "_baseline"
        )
    )


    for gene in TARGET_GENES:

        g = merged[
            merged[
                "gene"
            ]
            ==
            gene
        ]


        dr = (
            g[
                "pearson_r_moe"
            ]
            -
            g[
                "pearson_r_baseline"
            ]
        )


        dr2 = (
            g[
                "r2_moe"
            ]
            -
            g[
                "r2_baseline"
            ]
        )


        drmse = (
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
                "baseline":
                    baseline_name,

                "gene":
                    gene,

                "delta_pearson_mean":
                    float(
                        dr.mean()
                    ),

                "pearson_wins":
                    int(
                        (
                            dr > 0
                        ).sum()
                    ),

                "delta_r2_mean":
                    float(
                        dr2.mean()
                    ),

                "r2_wins":
                    int(
                        (
                            dr2 > 0
                        ).sum()
                    ),

                "delta_rmse_mean":
                    float(
                        drmse.mean()
                    ),

                "rmse_wins":
                    int(
                        (
                            drmse < 0
                        ).sum()
                    ),

                "n_seeds":
                    len(g)
            }
        )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    OUT
    / "external_replication_moe_comparison.csv",
    index=False
)


# =========================================================
# Final report
# =========================================================

print(
    "\n========================================"
)

print(
    "D10 INDEPENDENT-SLIDE REPLICATION"
)

print(
    "========================================"
)


print(
    "\nPerformance:"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nFrozen static gate weights:"
)

print(
    alpha_summary.to_string(
        index=False
    )
)


print(
    "\nMoE paired comparisons:"
)

print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    OUT
    / "external_spatial_split_manifest.csv"
)

print(
    OUT
    / "external_replication_raw_metrics.csv"
)

print(
    OUT
    / "external_replication_summary.csv"
)

print(
    OUT
    / "external_replication_gate_weights_raw.csv"
)

print(
    OUT
    / "external_replication_gate_weights_summary.csv"
)

print(
    OUT
    / "external_replication_moe_comparison.csv"
)

print(
    OUT
    / "external_replication_test_predictions.csv"
)
