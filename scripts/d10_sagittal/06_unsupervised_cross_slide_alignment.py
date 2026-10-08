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


# ============================================================
# Configuration
# ============================================================

BASE = Path(
    "/media/sophie/Data2T/Biomedical AI  skill"
)

D9 = (
    BASE
    / "D9_histology_spatial_multimodal"
)

D10 = (
    BASE
    / "D10_mouse_brain_external_validation"
)

D9_OUT = D9 / "outputs"
D10_OUT = D10 / "outputs"


D9_ADATA = (
    BASE
    / "D8 Squidpy"
    / "outputs"
    / "visium_hne_spatial_graph.h5ad"
)

D9_LOW = (
    D9_OUT
    / "dinov2_vits14_image_embeddings.npy"
)

D9_FULL = (
    D9_OUT
    / "dinov2_vits14_fullres_image_embeddings.npy"
)

D9_SPLIT = (
    D9_OUT
    / "prediction_split_manifest.csv"
)


D9_MANIFEST_CANDIDATES = [
    D9_OUT / "spot_image_embedding_manifest.csv",
    D9_OUT / "spot_patch_manifest.csv",
    D9_OUT / "fullres_spot_patch_manifest.csv",
]


D10_ADATA = (
    D10_OUT
    / "V1_Mouse_Brain_Sagittal_Anterior_external.h5ad"
)

D10_MANIFEST = (
    D10_OUT
    / "external_multiscale_patch_manifest.csv"
)

D10_LOW = (
    D10_OUT
    / "dinov2_vits14_external_lowres_embeddings.npy"
)

D10_FULL = (
    D10_OUT
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


ALIGNMENTS = [
    "None",
    "ZScore",
    "CORAL",
]


PCA_DIM = 50

HIDDEN_DIM = 64
DROPOUT = 0.20

LR = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 500
PATIENCE = 60

CORAL_EPS = 1e-4


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def evaluate(
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


# ============================================================
# AnnData target extraction
# ============================================================

def get_targets(
    adata,
):

    Y = np.zeros(
        (
            adata.n_obs,
            len(TARGET_GENES)
        ),
        dtype=np.float32
    )


    for j, gene in enumerate(
        TARGET_GENES
    ):

        if gene not in adata.var_names:

            raise KeyError(
                gene
            )


        x = adata[
            :,
            gene
        ].X


        if sparse.issparse(x):

            x = x.toarray()


        Y[
            :,
            j
        ] = (
            np.asarray(x)
            .ravel()
            .astype(np.float32)
        )


    return Y


# ============================================================
# Graph helpers
# ============================================================

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


    return torch.from_numpy(
        np.vstack(
            [
                sub.row,
                sub.col
            ]
        ).astype(np.int64)
    )


def full_edge_index(
    adjacency,
):

    coo = adjacency.tocoo()


    return torch.from_numpy(
        np.vstack(
            [
                coo.row,
                coo.col
            ]
        ).astype(np.int64)
    )


# ============================================================
# Find source manifest
# ============================================================

D9_MANIFEST = None


for p in D9_MANIFEST_CANDIDATES:

    if not p.exists():

        continue


    temp = pd.read_csv(
        p
    )


    if "barcode" in temp.columns:

        D9_MANIFEST = p
        break


if D9_MANIFEST is None:

    raise FileNotFoundError(
        "No valid D9 manifest found."
    )


print(
    "\nD9 manifest:",
    D9_MANIFEST
)


# ============================================================
# Load D9
# ============================================================

src_adata = sc.read_h5ad(
    D9_ADATA
)


src_manifest = pd.read_csv(
    D9_MANIFEST
)


src_manifest[
    "barcode"
] = (
    src_manifest[
        "barcode"
    ].astype(str)
)


X_src_low_raw = np.load(
    D9_LOW
).astype(np.float32)


X_src_full_raw = np.load(
    D9_FULL
).astype(np.float32)


src_barcodes = (
    src_manifest[
        "barcode"
    ]
    .tolist()
)


src_adata = src_adata[
    src_barcodes
].copy()


# ============================================================
# Load D10
# ============================================================

tgt_adata = sc.read_h5ad(
    D10_ADATA
)


tgt_manifest = pd.read_csv(
    D10_MANIFEST
)


tgt_manifest[
    "barcode"
] = (
    tgt_manifest[
        "barcode"
    ].astype(str)
)


X_tgt_low_raw = np.load(
    D10_LOW
).astype(np.float32)


X_tgt_full_raw = np.load(
    D10_FULL
).astype(np.float32)


tgt_barcodes = (
    tgt_manifest[
        "barcode"
    ]
    .tolist()
)


tgt_adata = tgt_adata[
    tgt_barcodes
].copy()


print(
    "\nD9:",
    X_src_low_raw.shape,
    X_src_full_raw.shape
)


print(
    "D10:",
    X_tgt_low_raw.shape,
    X_tgt_full_raw.shape
)


# ============================================================
# Expression targets
# ============================================================

Y_src = get_targets(
    src_adata
)


# Important:
# D10 labels are not used for any preprocessing/training.
# They are only used later for final evaluation.
Y_tgt = get_targets(
    tgt_adata
)


# ============================================================
# Graphs
# ============================================================

A_src = (
    src_adata.obsp[
        "spatial_connectivities"
    ]
    .tocsr()
    .astype(np.float32)
    .copy()
)


A_tgt = (
    tgt_adata.obsp[
        "spatial_connectivities"
    ]
    .tocsr()
    .astype(np.float32)
    .copy()
)


A_src.setdiag(0)
A_src.eliminate_zeros()

A_tgt.setdiag(0)
A_tgt.eliminate_zeros()


edge_src_all = full_edge_index(
    A_src
)


edge_tgt_all = full_edge_index(
    A_tgt
)


# ============================================================
# D9 source-only fit / validation
# ============================================================

split = pd.read_csv(
    D9_SPLIT
)


split[
    "barcode"
] = (
    split[
        "barcode"
    ].astype(str)
)


split = (
    split
    .set_index(
        "barcode"
    )
    .loc[
        src_barcodes
    ]
    .reset_index()
)


train_pool = np.where(
    split[
        "spatial_split"
    ].values
    ==
    "train"
)[0]


groups = (
    split.iloc[
        train_pool
    ][
        "spatial_block"
    ]
    .values
)


gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)


fit_rel, val_rel = next(
    gss.split(
        train_pool,
        groups=groups
    )
)


fit_idx = train_pool[
    fit_rel
]


val_idx = train_pool[
    val_rel
]


edge_fit = induced_edge_index(
    A_src,
    fit_idx
)


edge_val = induced_edge_index(
    A_src,
    val_idx
)


print(
    "\nD9 source selection:"
)

print(
    "Fit:",
    len(fit_idx)
)

print(
    "Val:",
    len(val_idx)
)

print(
    "Final D9:",
    src_adata.n_obs
)

print(
    "Unlabelled D10 adaptation spots:",
    tgt_adata.n_obs
)


# ============================================================
# GraphSAGE
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim
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
        edge_index
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


# ============================================================
# Train/select
# ============================================================

def select_model(
    X_fit,
    X_val,
    Y_fit,
    Y_val,
    edge_fit,
    edge_val,
    seed
):

    set_seed(seed)


    xf = torch.from_numpy(
        X_fit
    ).to(device)


    xv = torch.from_numpy(
        X_val
    ).to(device)


    yf = torch.from_numpy(
        Y_fit
    ).to(device)


    yv = torch.from_numpy(
        Y_val
    ).to(device)


    ef = edge_fit.to(device)
    ev = edge_val.to(device)


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
            xf,
            ef
        )


        loss = F.mse_loss(
            pred,
            yf
        )


        loss.backward()
        optimizer.step()


        model.eval()


        with torch.no_grad():

            vp = model(
                xv,
                ev
            )


            vl = F.mse_loss(
                vp,
                yv
            ).item()


        if (
            vl
            <
            best_loss - 1e-6
        ):

            best_loss = vl
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


def train_fixed(
    X,
    Y,
    edge_index,
    epochs,
    seed
):

    set_seed(seed)


    x = torch.from_numpy(
        X
    ).to(device)


    y = torch.from_numpy(
        Y
    ).to(device)


    edge = edge_index.to(device)


    model = GraphSAGE(
        X.shape[1]
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
            x,
            edge
        )


        loss = F.mse_loss(
            pred,
            y
        )


        loss.backward()
        optimizer.step()


    model.eval()

    return model


def predict(
    model,
    X,
    edge
):

    x = torch.from_numpy(
        X
    ).to(device)


    edge = edge.to(device)


    model.eval()


    with torch.no_grad():

        pred = model(
            x,
            edge
        )


    return (
        pred
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# Static source-only alpha
# ============================================================

def fit_alpha(
    y,
    low,
    full
):

    alpha = np.zeros(
        len(TARGET_GENES),
        dtype=np.float32
    )


    for g in range(
        len(TARGET_GENES)
    ):

        d = (
            low[:, g]
            -
            full[:, g]
        )


        den = np.sum(
            d * d
        )


        if den < 1e-12:

            a = 0.5

        else:

            a = np.sum(
                d
                *
                (
                    y[:, g]
                    -
                    full[:, g]
                )
            ) / den


        alpha[g] = np.clip(
            a,
            0,
            1
        )


    return alpha


# ============================================================
# Unsupervised feature alignment
# ============================================================

def zscore_match(
    X_target,
    X_source,
    eps=1e-6
):

    src_mean = X_source.mean(
        axis=0,
        keepdims=True
    )

    src_std = X_source.std(
        axis=0,
        keepdims=True
    )


    tgt_mean = X_target.mean(
        axis=0,
        keepdims=True
    )

    tgt_std = X_target.std(
        axis=0,
        keepdims=True
    )


    src_std = np.maximum(
        src_std,
        eps
    )

    tgt_std = np.maximum(
        tgt_std,
        eps
    )


    aligned = (
        (
            X_target
            -
            tgt_mean
        )
        /
        tgt_std
    )


    aligned = (
        aligned
        *
        src_std
        +
        src_mean
    )


    return aligned.astype(
        np.float32
    )


def matrix_power_psd(
    C,
    power,
    eps=1e-4
):

    vals, vecs = np.linalg.eigh(
        C
    )


    vals = np.clip(
        vals,
        eps,
        None
    )


    powered = (
        vecs
        @
        np.diag(
            vals ** power
        )
        @
        vecs.T
    )


    return powered


def coral_match(
    X_target,
    X_source,
    eps=1e-4
):

    src_mean = X_source.mean(
        axis=0,
        keepdims=True
    )


    tgt_mean = X_target.mean(
        axis=0,
        keepdims=True
    )


    Xs = (
        X_source
        -
        src_mean
    )


    Xt = (
        X_target
        -
        tgt_mean
    )


    Cs = (
        np.cov(
            Xs,
            rowvar=False
        )
        +
        eps
        *
        np.eye(
            Xs.shape[1]
        )
    )


    Ct = (
        np.cov(
            Xt,
            rowvar=False
        )
        +
        eps
        *
        np.eye(
            Xt.shape[1]
        )
    )


    Ct_inv_sqrt = matrix_power_psd(
        Ct,
        -0.5,
        eps
    )


    Cs_sqrt = matrix_power_psd(
        Cs,
        0.5,
        eps
    )


    aligned = (
        Xt
        @
        Ct_inv_sqrt
        @
        Cs_sqrt
        +
        src_mean
    )


    return aligned.astype(
        np.float32
    )


# ============================================================
# D9 selection preprocessing
# ============================================================

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
    X_src_low_raw[
        fit_idx
    ]
)


pca_full_sel.fit(
    X_src_full_raw[
        fit_idx
    ]
)


low_sel = pca_low_sel.transform(
    X_src_low_raw
).astype(np.float32)


full_sel = pca_full_sel.transform(
    X_src_full_raw
).astype(np.float32)


sc_low_sel = StandardScaler()

low_fit = sc_low_sel.fit_transform(
    low_sel[
        fit_idx
    ]
).astype(np.float32)

low_val = sc_low_sel.transform(
    low_sel[
        val_idx
    ]
).astype(np.float32)


sc_full_sel = StandardScaler()

full_fit = sc_full_sel.fit_transform(
    full_sel[
        fit_idx
    ]
).astype(np.float32)

full_val = sc_full_sel.transform(
    full_sel[
        val_idx
    ]
).astype(np.float32)


concat_sel = np.concatenate(
    [
        low_sel,
        full_sel
    ],
    axis=1
)


sc_concat_sel = StandardScaler()

concat_fit = sc_concat_sel.fit_transform(
    concat_sel[
        fit_idx
    ]
).astype(np.float32)

concat_val = sc_concat_sel.transform(
    concat_sel[
        val_idx
    ]
).astype(np.float32)


Y_fit_raw = Y_src[
    fit_idx
]

Y_val_raw = Y_src[
    val_idx
]


Y_sel_mean = Y_fit_raw.mean(
    axis=0
)

Y_sel_std = Y_fit_raw.std(
    axis=0
)

Y_sel_std[
    Y_sel_std < 1e-6
] = 1.0


Y_fit = (
    (
        Y_fit_raw
        -
        Y_sel_mean
    )
    /
    Y_sel_std
).astype(np.float32)


Y_val = (
    (
        Y_val_raw
        -
        Y_sel_mean
    )
    /
    Y_sel_std
).astype(np.float32)


# ============================================================
# Final D9 PCA
# ============================================================

print(
    "\nFitting final D9-only PCA..."
)


pca_low = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_full = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_low.fit(
    X_src_low_raw
)


pca_full.fit(
    X_src_full_raw
)


src_low_pca = pca_low.transform(
    X_src_low_raw
).astype(np.float32)


src_full_pca = pca_full.transform(
    X_src_full_raw
).astype(np.float32)


tgt_low_pca_raw = pca_low.transform(
    X_tgt_low_raw
).astype(np.float32)


tgt_full_pca_raw = pca_full.transform(
    X_tgt_full_raw
).astype(np.float32)


# ============================================================
# Target alignment variants
#
# D10 expression labels are NOT used.
# ============================================================

target_pca_sets = {}


target_pca_sets[
    "None"
] = (
    tgt_low_pca_raw,
    tgt_full_pca_raw
)


target_pca_sets[
    "ZScore"
] = (
    zscore_match(
        tgt_low_pca_raw,
        src_low_pca
    ),

    zscore_match(
        tgt_full_pca_raw,
        src_full_pca
    )
)


target_pca_sets[
    "CORAL"
] = (
    coral_match(
        tgt_low_pca_raw,
        src_low_pca,
        CORAL_EPS
    ),

    coral_match(
        tgt_full_pca_raw,
        src_full_pca,
        CORAL_EPS
    )
)


# ============================================================
# D9-only final scalers
# ============================================================

sc_low = StandardScaler()

src_low = sc_low.fit_transform(
    src_low_pca
).astype(np.float32)


sc_full = StandardScaler()

src_full = sc_full.fit_transform(
    src_full_pca
).astype(np.float32)


src_concat_pca = np.concatenate(
    [
        src_low_pca,
        src_full_pca
    ],
    axis=1
)


sc_concat = StandardScaler()

src_concat = sc_concat.fit_transform(
    src_concat_pca
).astype(np.float32)


# ============================================================
# Transform aligned targets with D9 scalers
# ============================================================

target_features = {}


for alignment in ALIGNMENTS:

    low_pca_tgt, full_pca_tgt = (
        target_pca_sets[
            alignment
        ]
    )


    low_scaled = sc_low.transform(
        low_pca_tgt
    ).astype(np.float32)


    full_scaled = sc_full.transform(
        full_pca_tgt
    ).astype(np.float32)


    concat_pca_tgt = np.concatenate(
        [
            low_pca_tgt,
            full_pca_tgt
        ],
        axis=1
    )


    concat_scaled = sc_concat.transform(
        concat_pca_tgt
    ).astype(np.float32)


    target_features[
        alignment
    ] = {
        "low":
            low_scaled,

        "full":
            full_scaled,

        "concat":
            concat_scaled,
    }


# ============================================================
# Alignment diagnostics
# ============================================================

diagnostic_rows = []


for alignment in ALIGNMENTS:

    for feature_name, source_x in [
        (
            "Low",
            src_low
        ),
        (
            "Full",
            src_full
        ),
        (
            "Concat",
            src_concat
        ),
    ]:

        target_key = (
            feature_name.lower()
        )

        if target_key == "concat":

            target_x = (
                target_features[
                    alignment
                ][
                    "concat"
                ]
            )

        elif target_key == "low":

            target_x = (
                target_features[
                    alignment
                ][
                    "low"
                ]
            )

        else:

            target_x = (
                target_features[
                    alignment
                ][
                    "full"
                ]
            )


        diagnostic_rows.append(
            {
                "alignment":
                    alignment,

                "feature_set":
                    feature_name,

                "source_mean":
                    float(
                        source_x.mean()
                    ),

                "target_mean":
                    float(
                        target_x.mean()
                    ),

                "source_abs_mean":
                    float(
                        np.abs(
                            source_x
                        ).mean()
                    ),

                "target_abs_mean":
                    float(
                        np.abs(
                            target_x
                        ).mean()
                    ),

                "source_std":
                    float(
                        source_x.std()
                    ),

                "target_std":
                    float(
                        target_x.std()
                    ),
            }
        )


pd.DataFrame(
    diagnostic_rows
).to_csv(
    D10_OUT
    / "unsupervised_alignment_feature_diagnostics.csv",
    index=False
)


# ============================================================
# D9 final expression normalization
# ============================================================

Y_mean = Y_src.mean(
    axis=0
)

Y_std = Y_src.std(
    axis=0
)

Y_std[
    Y_std < 1e-6
] = 1.0


Y_src_std = (
    (
        Y_src
        -
        Y_mean
    )
    /
    Y_std
).astype(np.float32)


# ============================================================
# Main experiment
# ============================================================

metric_rows = []
gate_rows = []
training_rows = []


for seed in SEEDS:

    print(
        "\n======================================"
    )

    print(
        "Seed:",
        seed
    )

    print(
        "======================================"
    )


    # --------------------------------------------------------
    # D9-only model selection
    # --------------------------------------------------------

    (
        low_epoch,
        low_vloss,
        low_sel_model
    ) = select_model(
        low_fit,
        low_val,
        Y_fit,
        Y_val,
        edge_fit,
        edge_val,
        seed
    )


    (
        full_epoch,
        full_vloss,
        full_sel_model
    ) = select_model(
        full_fit,
        full_val,
        Y_fit,
        Y_val,
        edge_fit,
        edge_val,
        seed
    )


    (
        concat_epoch,
        concat_vloss,
        _
    ) = select_model(
        concat_fit,
        concat_val,
        Y_fit,
        Y_val,
        edge_fit,
        edge_val,
        seed
    )


    # --------------------------------------------------------
    # D9-only frozen gate
    # --------------------------------------------------------

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


    alpha = fit_alpha(
        Y_val,
        low_val_pred,
        full_val_pred
    )


    print(
        "epochs:",
        low_epoch,
        full_epoch,
        concat_epoch
    )


    print(
        "alpha:",
        {
            gene:
                round(
                    float(
                        alpha[g]
                    ),
                    3
                )
            for g, gene
            in enumerate(
                TARGET_GENES
            )
        }
    )


    # --------------------------------------------------------
    # Final source-only models
    # --------------------------------------------------------

    low_model = train_fixed(
        src_low,
        Y_src_std,
        edge_src_all,
        low_epoch,
        seed
    )


    full_model = train_fixed(
        src_full,
        Y_src_std,
        edge_src_all,
        full_epoch,
        seed
    )


    concat_model = train_fixed(
        src_concat,
        Y_src_std,
        edge_src_all,
        concat_epoch,
        seed
    )


    # --------------------------------------------------------
    # Apply each label-free target alignment
    # --------------------------------------------------------

    for alignment in ALIGNMENTS:

        target = target_features[
            alignment
        ]


        low_std = predict(
            low_model,
            target[
                "low"
            ],
            edge_tgt_all
        )


        full_std = predict(
            full_model,
            target[
                "full"
            ],
            edge_tgt_all
        )


        concat_std = predict(
            concat_model,
            target[
                "concat"
            ],
            edge_tgt_all
        )


        moe_std = (
            alpha[
                None,
                :
            ]
            *
            low_std
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
            full_std
        )


        predictions = {

            "LowExpert_GraphSAGE":
                (
                    low_std
                    *
                    Y_std
                    +
                    Y_mean
                ),

            "FullExpert_GraphSAGE":
                (
                    full_std
                    *
                    Y_std
                    +
                    Y_mean
                ),

            "SeparatePCA50x2_GraphSAGE":
                (
                    concat_std
                    *
                    Y_std
                    +
                    Y_mean
                ),

            "FrozenStaticMoE_GraphSAGE":
                (
                    moe_std
                    *
                    Y_std
                    +
                    Y_mean
                ),
        }


        # ----------------------------------------------------
        # D10 labels enter ONLY HERE
        # ----------------------------------------------------

        for model_name, pred in predictions.items():

            for g, gene in enumerate(
                TARGET_GENES
            ):

                m = evaluate(
                    Y_tgt[
                        :,
                        g
                    ],
                    pred[
                        :,
                        g
                    ]
                )


                metric_rows.append(
                    {
                        "alignment":
                            alignment,

                        "model":
                            model_name,

                        "gene":
                            gene,

                        "seed":
                            seed,

                        **m,
                    }
                )


    for g, gene in enumerate(
        TARGET_GENES
    ):

        gate_rows.append(
            {
                "seed":
                    seed,

                "gene":
                    gene,

                "low_weight":
                    float(
                        alpha[g]
                    ),

                "full_weight":
                    float(
                        1.0
                        -
                        alpha[g]
                    ),
            }
        )


    training_rows.append(
        {
            "seed":
                seed,

            "low_epoch":
                low_epoch,

            "low_val_loss":
                low_vloss,

            "full_epoch":
                full_epoch,

            "full_val_loss":
                full_vloss,

            "concat_epoch":
                concat_epoch,

            "concat_val_loss":
                concat_vloss,
        }
    )


# ============================================================
# Save raw metrics
# ============================================================

metrics = pd.DataFrame(
    metric_rows
)


metrics.to_csv(
    D10_OUT
    / "unsupervised_alignment_raw_metrics.csv",
    index=False
)


pd.DataFrame(
    gate_rows
).to_csv(
    D10_OUT
    / "unsupervised_alignment_source_gate_weights.csv",
    index=False
)


pd.DataFrame(
    training_rows
).to_csv(
    D10_OUT
    / "unsupervised_alignment_training.csv",
    index=False
)


# ============================================================
# Summary
# ============================================================

summary = (
    metrics
    .groupby(
        [
            "alignment",
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
        ),
    )
    .reset_index()
)


summary.to_csv(
    D10_OUT
    / "unsupervised_alignment_summary.csv",
    index=False
)


# ============================================================
# Paired alignment improvement vs None
# ============================================================

none = metrics[
    metrics[
        "alignment"
    ]
    ==
    "None"
].copy()


delta_rows = []


for alignment in [
    "ZScore",
    "CORAL"
]:

    candidate = metrics[
        metrics[
            "alignment"
        ]
        ==
        alignment
    ].copy()


    merged = candidate.merge(
        none[
            [
                "model",
                "gene",
                "seed",
                "pearson_r",
                "r2",
                "rmse",
            ]
        ],
        on=[
            "model",
            "gene",
            "seed"
        ],
        suffixes=(
            "_aligned",
            "_none"
        )
    )


    for model_name in (
        merged[
            "model"
        ].unique()
    ):

        for gene in TARGET_GENES:

            g = merged[
                (
                    merged[
                        "model"
                    ]
                    ==
                    model_name
                )
                &
                (
                    merged[
                        "gene"
                    ]
                    ==
                    gene
                )
            ]


            dr = (
                g[
                    "pearson_r_aligned"
                ]
                -
                g[
                    "pearson_r_none"
                ]
            )


            dr2 = (
                g[
                    "r2_aligned"
                ]
                -
                g[
                    "r2_none"
                ]
            )


            drmse = (
                g[
                    "rmse_aligned"
                ]
                -
                g[
                    "rmse_none"
                ]
            )


            delta_rows.append(
                {
                    "alignment":
                        alignment,

                    "model":
                        model_name,

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
                        len(g),
                }
            )


delta_df = pd.DataFrame(
    delta_rows
)


delta_df.to_csv(
    D10_OUT
    / "unsupervised_alignment_vs_zero_shot.csv",
    index=False
)


# ============================================================
# Best method per gene
# ============================================================

best_rows = []


for gene in TARGET_GENES:

    g = summary[
        summary[
            "gene"
        ]
        ==
        gene
    ]


    best_r = g.loc[
        g[
            "pearson_mean"
        ].idxmax()
    ]


    best_r2 = g.loc[
        g[
            "r2_mean"
        ].idxmax()
    ]


    best_rows.append(
        {
            "gene":
                gene,

            "best_pearson_alignment":
                best_r[
                    "alignment"
                ],

            "best_pearson_model":
                best_r[
                    "model"
                ],

            "best_pearson":
                float(
                    best_r[
                        "pearson_mean"
                    ]
                ),

            "best_r2_alignment":
                best_r2[
                    "alignment"
                ],

            "best_r2_model":
                best_r2[
                    "model"
                ],

            "best_r2":
                float(
                    best_r2[
                        "r2_mean"
                    ]
                ),
        }
    )


best_df = pd.DataFrame(
    best_rows
)


best_df.to_csv(
    D10_OUT
    / "unsupervised_alignment_best_models.csv",
    index=False
)


# ============================================================
# Final report
# ============================================================

print(
    "\n=============================================="
)

print(
    "UNSUPERVISED D9 -> D10 DOMAIN ALIGNMENT"
)

print(
    "=============================================="
)


print(
    "\nSummary:"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nImprovement vs strict zero-shot:"
)

print(
    delta_df.to_string(
        index=False
    )
)


print(
    "\nBest model / alignment:"
)

print(
    best_df.to_string(
        index=False
    )
)


print(
    "\nIMPORTANT:"
)

print(
    "ZScore and CORAL used D10 morphology embeddings, "
    "but NO D10 gene-expression labels."
)

print(
    "Therefore they are label-free transductive "
    "domain-adaptation experiments, not pure zero-shot."
)


print(
    "\nSaved:"
)

for name in [
    "unsupervised_alignment_feature_diagnostics.csv",
    "unsupervised_alignment_raw_metrics.csv",
    "unsupervised_alignment_summary.csv",
    "unsupervised_alignment_vs_zero_shot.csv",
    "unsupervised_alignment_best_models.csv",
]:

    print(
        D10_OUT
        /
        name
    )
