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
# 0. Configuration
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


# ------------------------------------------------------------
# D9 SOURCE
# ------------------------------------------------------------

D9_ADATA = (
    BASE
    / "D8 Squidpy"
    / "outputs"
    / "visium_hne_spatial_graph.h5ad"
)

D9_LOW_EMB = (
    D9_OUT
    / "dinov2_vits14_image_embeddings.npy"
)

D9_FULL_EMB = (
    D9_OUT
    / "dinov2_vits14_fullres_image_embeddings.npy"
)

D9_SPLIT = (
    D9_OUT
    / "prediction_split_manifest.csv"
)


# D9 manifest candidates.
# Script will use the first valid one.
D9_MANIFEST_CANDIDATES = [
    D9_OUT / "spot_image_embedding_manifest.csv",
    D9_OUT / "spot_patch_manifest.csv",
    D9_OUT / "fullres_spot_patch_manifest.csv",
]


# ------------------------------------------------------------
# D10 TARGET
# ------------------------------------------------------------

D10_ADATA = (
    D10_OUT
    / "V1_Mouse_Brain_Sagittal_Anterior_external.h5ad"
)

D10_MANIFEST = (
    D10_OUT
    / "external_multiscale_patch_manifest.csv"
)

D10_LOW_EMB = (
    D10_OUT
    / "dinov2_vits14_external_lowres_embeddings.npy"
)

D10_FULL_EMB = (
    D10_OUT
    / "dinov2_vits14_external_fullres_embeddings.npy"
)


# ------------------------------------------------------------
# Targets / model
# ------------------------------------------------------------

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
# 1. Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Metrics
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
# 3. Helpers
# ============================================================

def extract_targets(
    adata,
):

    Y = np.zeros(
        (
            adata.n_obs,
            len(TARGET_GENES)
        ),
        dtype=np.float32,
    )


    for j, gene in enumerate(
        TARGET_GENES
    ):

        if gene not in adata.var_names:

            raise KeyError(
                f"{gene} missing."
            )


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


    return Y


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


    edges = np.vstack(
        [
            sub.row,
            sub.col
        ]
    ).astype(np.int64)


    return torch.from_numpy(
        edges
    )


def full_edge_index(
    adjacency,
):

    coo = adjacency.tocoo()


    edges = np.vstack(
        [
            coo.row,
            coo.col
        ]
    ).astype(np.int64)


    return torch.from_numpy(
        edges
    )


# ============================================================
# 4. Find D9 manifest
# ============================================================

D9_MANIFEST = None


for candidate in D9_MANIFEST_CANDIDATES:

    if candidate.exists():

        temp = pd.read_csv(
            candidate
        )

        if (
            "barcode"
            in temp.columns
        ):

            D9_MANIFEST = candidate

            break


if D9_MANIFEST is None:

    raise FileNotFoundError(
        "Could not find a valid D9 manifest "
        "containing a barcode column."
    )


print(
    "\nD9 manifest:",
    D9_MANIFEST
)


# ============================================================
# 5. Load SOURCE D9
# ============================================================

print(
    "\n========================================"
)

print(
    "LOAD SOURCE: D9"
)

print(
    "========================================"
)


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
    ]
    .astype(str)
)


X_src_low_raw = np.load(
    D9_LOW_EMB
).astype(np.float32)


X_src_full_raw = np.load(
    D9_FULL_EMB
).astype(np.float32)


print(
    "D9 AnnData:",
    src_adata.shape
)

print(
    "D9 low embedding:",
    X_src_low_raw.shape
)

print(
    "D9 full embedding:",
    X_src_full_raw.shape
)

print(
    "D9 manifest:",
    src_manifest.shape
)


if (
    len(src_manifest)
    !=
    X_src_low_raw.shape[0]
):

    raise RuntimeError(
        "D9 low embedding / manifest "
        "row mismatch."
    )


if (
    len(src_manifest)
    !=
    X_src_full_raw.shape[0]
):

    raise RuntimeError(
        "D9 full embedding / manifest "
        "row mismatch."
    )


src_barcodes = (
    src_manifest[
        "barcode"
    ]
    .tolist()
)


missing_src = [
    b
    for b in src_barcodes
    if b not in src_adata.obs_names
]


if missing_src:

    raise RuntimeError(
        f"D9 AnnData missing "
        f"{len(missing_src)} manifest barcodes."
    )


src_adata = src_adata[
    src_barcodes
].copy()


assert (
    list(
        src_adata.obs_names
    )
    ==
    src_barcodes
)


print(
    "D9 barcode alignment: OK"
)


# ============================================================
# 6. Load TARGET D10
# ============================================================

print(
    "\n========================================"
)

print(
    "LOAD TARGET: D10"
)

print(
    "========================================"
)


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
    ]
    .astype(str)
)


X_tgt_low_raw = np.load(
    D10_LOW_EMB
).astype(np.float32)


X_tgt_full_raw = np.load(
    D10_FULL_EMB
).astype(np.float32)


print(
    "D10 AnnData:",
    tgt_adata.shape
)

print(
    "D10 low embedding:",
    X_tgt_low_raw.shape
)

print(
    "D10 full embedding:",
    X_tgt_full_raw.shape
)

print(
    "D10 manifest:",
    tgt_manifest.shape
)


if (
    len(tgt_manifest)
    !=
    X_tgt_low_raw.shape[0]
):

    raise RuntimeError(
        "D10 low embedding / manifest "
        "row mismatch."
    )


if (
    len(tgt_manifest)
    !=
    X_tgt_full_raw.shape[0]
):

    raise RuntimeError(
        "D10 full embedding / manifest "
        "row mismatch."
    )


tgt_barcodes = (
    tgt_manifest[
        "barcode"
    ]
    .tolist()
)


missing_tgt = [
    b
    for b in tgt_barcodes
    if b not in tgt_adata.obs_names
]


if missing_tgt:

    raise RuntimeError(
        f"D10 AnnData missing "
        f"{len(missing_tgt)} manifest barcodes."
    )


tgt_adata = tgt_adata[
    tgt_barcodes
].copy()


assert (
    list(
        tgt_adata.obs_names
    )
    ==
    tgt_barcodes
)


print(
    "D10 barcode alignment: OK"
)


# ============================================================
# 7. Numerical QC
# ============================================================

for name, x in [
    (
        "D9 low",
        X_src_low_raw
    ),
    (
        "D9 full",
        X_src_full_raw
    ),
    (
        "D10 low",
        X_tgt_low_raw
    ),
    (
        "D10 full",
        X_tgt_full_raw
    ),
]:

    print(
        name,
        "| shape=",
        x.shape,
        "| NaN=",
        int(
            np.isnan(x).sum()
        ),
        "| Inf=",
        int(
            np.isinf(x).sum()
        )
    )


    if (
        np.isnan(x).any()
        or
        np.isinf(x).any()
    ):

        raise RuntimeError(
            f"Invalid values in {name}"
        )


# ============================================================
# 8. Expression targets
# ============================================================

Y_src = extract_targets(
    src_adata
)


Y_tgt = extract_targets(
    tgt_adata
)


print(
    "\nExpression statistics:"
)


expression_shift_rows = []


for j, gene in enumerate(
    TARGET_GENES
):

    s = Y_src[
        :,
        j
    ]

    t = Y_tgt[
        :,
        j
    ]


    print(
        gene,
        "| D9 mean/std:",
        round(
            float(
                s.mean()
            ),
            4
        ),
        "/",
        round(
            float(
                s.std()
            ),
            4
        ),
        "| D10 mean/std:",
        round(
            float(
                t.mean()
            ),
            4
        ),
        "/",
        round(
            float(
                t.std()
            ),
            4
        ),
    )


    expression_shift_rows.append(
        {
            "gene":
                gene,

            "d9_mean":
                float(
                    s.mean()
                ),

            "d9_std":
                float(
                    s.std()
                ),

            "d9_nonzero_fraction":
                float(
                    (
                        s > 0
                    ).mean()
                ),

            "d10_mean":
                float(
                    t.mean()
                ),

            "d10_std":
                float(
                    t.std()
                ),

            "d10_nonzero_fraction":
                float(
                    (
                        t > 0
                    ).mean()
                ),
        }
    )


pd.DataFrame(
    expression_shift_rows
).to_csv(
    D10_OUT
    / "cross_slide_expression_shift.csv",
    index=False
)


# ============================================================
# 9. Graphs
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


print(
    "\nSource graph:",
    A_src.shape,
    "nnz=",
    A_src.nnz
)

print(
    "Target graph:",
    A_tgt.shape,
    "nnz=",
    A_tgt.nnz
)


# ============================================================
# 10. D9 source-only model-selection split
# ============================================================

src_split = pd.read_csv(
    D9_SPLIT
)


src_split[
    "barcode"
] = (
    src_split[
        "barcode"
    ]
    .astype(str)
)


src_split = (
    src_split
    .set_index(
        "barcode"
    )
    .loc[
        src_barcodes
    ]
    .reset_index()
)


if (
    "spatial_split"
    not in src_split.columns
):

    raise KeyError(
        "D9 prediction_split_manifest.csv "
        "does not contain spatial_split."
    )


if (
    "spatial_block"
    not in src_split.columns
):

    raise KeyError(
        "D9 prediction_split_manifest.csv "
        "does not contain spatial_block."
    )


source_train_pool = np.where(
    src_split[
        "spatial_split"
    ].values
    ==
    "train"
)[0]


source_groups = (
    src_split.iloc[
        source_train_pool
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
        source_train_pool,
        groups=source_groups
    )
)


src_fit_idx = source_train_pool[
    fit_rel
]


src_val_idx = source_train_pool[
    val_rel
]


print(
    "\nD9 source-only selection:"
)

print(
    "fit:",
    len(
        src_fit_idx
    )
)

print(
    "validation:",
    len(
        src_val_idx
    )
)

print(
    "final source training:",
    src_adata.n_obs,
    "spots"
)

print(
    "external target:",
    tgt_adata.n_obs,
    "spots"
)


# ============================================================
# 11. GraphSAGE
# ============================================================

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


        x = F.relu(
            x
        )


        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )


        x = self.conv2(
            x,
            edge_index
        )


        x = F.relu(
            x
        )


        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )


        return self.head(
            x
        )


# ============================================================
# 12. Model selection
# ============================================================

def select_model(
    X_fit,
    X_val,
    Y_fit,
    Y_val,
    edge_fit,
    edge_val,
    seed,
):

    set_seed(
        seed
    )


    x_fit = torch.from_numpy(
        X_fit
    ).to(device)


    x_val = torch.from_numpy(
        X_val
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
    bad_epochs = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()


        pred = model(
            x_fit,
            ef
        )


        loss = F.mse_loss(
            pred,
            y_fit
        )


        loss.backward()

        optimizer.step()


        model.eval()


        with torch.no_grad():

            val_pred = model(
                x_val,
                ev
            )


            val_loss = F.mse_loss(
                val_pred,
                y_val
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
            PATIENCE
        ):

            break


    if best_state is None:

        raise RuntimeError(
            "Model selection failed."
        )


    model.load_state_dict(
        best_state
    )

    model.eval()


    return (
        best_epoch,
        best_loss,
        model
    )


# ============================================================
# 13. Final fixed-epoch source training
# ============================================================

def train_final_source_model(
    X_source,
    Y_source,
    edge_source,
    epochs,
    seed,
):

    set_seed(
        seed
    )


    x = torch.from_numpy(
        X_source
    ).to(device)


    y = torch.from_numpy(
        Y_source
    ).to(device)


    edge = edge_source.to(
        device
    )


    model = GraphSAGE(
        X_source.shape[1]
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    for _ in range(
        int(
            epochs
        )
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


# ============================================================
# 14. Predict
# ============================================================

def predict(
    model,
    X,
    edge_index,
):

    x = torch.from_numpy(
        X
    ).to(device)


    edge = edge_index.to(
        device
    )


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
# 15. Source-validation static MoE alpha
#
# y = alpha * low + (1-alpha) * full
#
# Alpha learned ONLY on D9 validation.
# D10 labels are never used.
# ============================================================

def fit_convex_alpha(
    y_true,
    low_pred,
    full_pred,
):

    alpha = np.zeros(
        len(TARGET_GENES),
        dtype=np.float32
    )


    for g in range(
        len(TARGET_GENES)
    ):

        y = y_true[
            :,
            g
        ]


        low = low_pred[
            :,
            g
        ]


        full = full_pred[
            :,
            g
        ]


        d = (
            low
            -
            full
        )


        denominator = np.sum(
            d * d
        )


        if denominator < 1e-12:

            a = 0.5

        else:

            a = np.sum(
                d
                *
                (
                    y
                    -
                    full
                )
            ) / denominator


        alpha[
            g
        ] = float(
            np.clip(
                a,
                0.0,
                1.0
            )
        )


    return alpha


# ============================================================
# 16. SOURCE selection preprocessing
#
# Absolutely NO D10 fitting here.
# ============================================================

print(
    "\n========================================"
)

print(
    "D9 SOURCE-ONLY MODEL SELECTION"
)

print(
    "========================================"
)


# ------------------------------------------------------------
# Separate source-only PCA
# ------------------------------------------------------------

pca_low_select = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_full_select = PCA(
    n_components=PCA_DIM,
    svd_solver="randomized",
    random_state=42
)


pca_low_select.fit(
    X_src_low_raw[
        src_fit_idx
    ]
)


pca_full_select.fit(
    X_src_full_raw[
        src_fit_idx
    ]
)


src_low_select_all = (
    pca_low_select.transform(
        X_src_low_raw
    )
    .astype(np.float32)
)


src_full_select_all = (
    pca_full_select.transform(
        X_src_full_raw
    )
    .astype(np.float32)
)


# ------------------------------------------------------------
# Low scaler
# ------------------------------------------------------------

sc_low_select = StandardScaler()


src_low_fit = sc_low_select.fit_transform(
    src_low_select_all[
        src_fit_idx
    ]
).astype(np.float32)


src_low_val = sc_low_select.transform(
    src_low_select_all[
        src_val_idx
    ]
).astype(np.float32)


# ------------------------------------------------------------
# Full scaler
# ------------------------------------------------------------

sc_full_select = StandardScaler()


src_full_fit = sc_full_select.fit_transform(
    src_full_select_all[
        src_fit_idx
    ]
).astype(np.float32)


src_full_val = sc_full_select.transform(
    src_full_select_all[
        src_val_idx
    ]
).astype(np.float32)


# ------------------------------------------------------------
# Separate PCA50x2 concat
# ------------------------------------------------------------

src_concat_select_all = np.concatenate(
    [
        src_low_select_all,
        src_full_select_all
    ],
    axis=1
)


sc_concat_select = StandardScaler()


src_concat_fit = sc_concat_select.fit_transform(
    src_concat_select_all[
        src_fit_idx
    ]
).astype(np.float32)


src_concat_val = sc_concat_select.transform(
    src_concat_select_all[
        src_val_idx
    ]
).astype(np.float32)


# ------------------------------------------------------------
# Source expression normalization
# ------------------------------------------------------------

Y_src_fit_raw = Y_src[
    src_fit_idx
]


Y_src_val_raw = Y_src[
    src_val_idx
]


Y_select_mean = Y_src_fit_raw.mean(
    axis=0
)


Y_select_std = Y_src_fit_raw.std(
    axis=0
)


Y_select_std[
    Y_select_std < 1e-6
] = 1.0


Y_src_fit = (
    (
        Y_src_fit_raw
        -
        Y_select_mean
    )
    /
    Y_select_std
).astype(np.float32)


Y_src_val = (
    (
        Y_src_val_raw
        -
        Y_select_mean
    )
    /
    Y_select_std
).astype(np.float32)


# ------------------------------------------------------------
# Source selection graphs
# ------------------------------------------------------------

edge_src_fit = induced_edge_index(
    A_src,
    src_fit_idx
)


edge_src_val = induced_edge_index(
    A_src,
    src_val_idx
)


# ============================================================
# 17. FINAL source preprocessing
#
# Fit on ALL D9.
#
# D10 only TRANSFORM.
# ============================================================

print(
    "\n========================================"
)

print(
    "FIT FINAL PREPROCESSING ON ALL D9"
)

print(
    "========================================"
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


# D9 only
pca_low_final.fit(
    X_src_low_raw
)


pca_full_final.fit(
    X_src_full_raw
)


# Transform D9
src_low_pca = (
    pca_low_final.transform(
        X_src_low_raw
    )
    .astype(np.float32)
)


src_full_pca = (
    pca_full_final.transform(
        X_src_full_raw
    )
    .astype(np.float32)
)


# Transform D10 using D9 PCA
tgt_low_pca = (
    pca_low_final.transform(
        X_tgt_low_raw
    )
    .astype(np.float32)
)


tgt_full_pca = (
    pca_full_final.transform(
        X_tgt_full_raw
    )
    .astype(np.float32)
)


print(
    "D9 low PCA variance:",
    float(
        pca_low_final
        .explained_variance_ratio_
        .sum()
    )
)


print(
    "D9 full PCA variance:",
    float(
        pca_full_final
        .explained_variance_ratio_
        .sum()
    )
)


# ------------------------------------------------------------
# D9-only scalers
# ------------------------------------------------------------

sc_low_final = StandardScaler()


src_low_final = sc_low_final.fit_transform(
    src_low_pca
).astype(np.float32)


tgt_low_final = sc_low_final.transform(
    tgt_low_pca
).astype(np.float32)


sc_full_final = StandardScaler()


src_full_final = sc_full_final.fit_transform(
    src_full_pca
).astype(np.float32)


tgt_full_final = sc_full_final.transform(
    tgt_full_pca
).astype(np.float32)


# ------------------------------------------------------------
# Concat
# ------------------------------------------------------------

src_concat_pca = np.concatenate(
    [
        src_low_pca,
        src_full_pca
    ],
    axis=1
)


tgt_concat_pca = np.concatenate(
    [
        tgt_low_pca,
        tgt_full_pca
    ],
    axis=1
)


sc_concat_final = StandardScaler()


src_concat_final = sc_concat_final.fit_transform(
    src_concat_pca
).astype(np.float32)


tgt_concat_final = sc_concat_final.transform(
    tgt_concat_pca
).astype(np.float32)


# ------------------------------------------------------------
# D9-only expression normalization
# ------------------------------------------------------------

Y_final_mean = Y_src.mean(
    axis=0
)


Y_final_std = Y_src.std(
    axis=0
)


Y_final_std[
    Y_final_std < 1e-6
] = 1.0


Y_src_final = (
    (
        Y_src
        -
        Y_final_mean
    )
    /
    Y_final_std
).astype(np.float32)


# ------------------------------------------------------------
# Full source / target graphs
# ------------------------------------------------------------

edge_src_all = full_edge_index(
    A_src
)


edge_tgt_all = full_edge_index(
    A_tgt
)


print(
    "D9 final graph edges:",
    edge_src_all.shape[1]
)


print(
    "D10 inference graph edges:",
    edge_tgt_all.shape[1]
)


# ============================================================
# 18. Distribution-shift diagnostics
# ============================================================

feature_shift_rows = []


for name, src_x, tgt_x in [
    (
        "Low_PCA50_scaled",
        src_low_final,
        tgt_low_final
    ),
    (
        "Full_PCA50_scaled",
        src_full_final,
        tgt_full_final
    ),
    (
        "SeparatePCA50x2_scaled",
        src_concat_final,
        tgt_concat_final
    ),
]:

    feature_shift_rows.append(
        {
            "feature_set":
                name,

            "source_abs_mean":
                float(
                    np.abs(
                        src_x
                    ).mean()
                ),

            "target_abs_mean":
                float(
                    np.abs(
                        tgt_x
                    ).mean()
                ),

            "source_std":
                float(
                    src_x.std()
                ),

            "target_std":
                float(
                    tgt_x.std()
                ),
        }
    )


pd.DataFrame(
    feature_shift_rows
).to_csv(
    D10_OUT
    / "cross_slide_feature_shift.csv",
    index=False
)


# ============================================================
# 19. Main cross-slide experiment
# ============================================================

metric_rows = []
alpha_rows = []
training_rows = []
prediction_rows = []


for seed in SEEDS:

    print(
        "\n========================================"
    )

    print(
        f"SEED {seed}"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # A. Select low expert using D9 only
    # --------------------------------------------------------

    (
        low_epoch,
        low_val_loss,
        low_selection_model
    ) = select_model(
        src_low_fit,
        src_low_val,
        Y_src_fit,
        Y_src_val,
        edge_src_fit,
        edge_src_val,
        seed
    )


    # --------------------------------------------------------
    # B. Select full expert using D9 only
    # --------------------------------------------------------

    (
        full_epoch,
        full_val_loss,
        full_selection_model
    ) = select_model(
        src_full_fit,
        src_full_val,
        Y_src_fit,
        Y_src_val,
        edge_src_fit,
        edge_src_val,
        seed
    )


    # --------------------------------------------------------
    # C. Select concat model using D9 only
    # --------------------------------------------------------

    (
        concat_epoch,
        concat_val_loss,
        _
    ) = select_model(
        src_concat_fit,
        src_concat_val,
        Y_src_fit,
        Y_src_val,
        edge_src_fit,
        edge_src_val,
        seed
    )


    # --------------------------------------------------------
    # D. D9 validation predictions for alpha
    # --------------------------------------------------------

    low_val_pred = predict(
        low_selection_model,
        src_low_val,
        edge_src_val
    )


    full_val_pred = predict(
        full_selection_model,
        src_full_val,
        edge_src_val
    )


    alpha = fit_convex_alpha(
        Y_src_val,
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
        "Frozen D9 alpha:",
        {
            gene:
                round(
                    float(
                        alpha[g]
                    ),
                    4
                )

            for g, gene
            in enumerate(
                TARGET_GENES
            )
        }
    )


    # --------------------------------------------------------
    # E. Final models trained on ALL D9
    # --------------------------------------------------------

    low_model = train_final_source_model(
        src_low_final,
        Y_src_final,
        edge_src_all,
        low_epoch,
        seed
    )


    full_model = train_final_source_model(
        src_full_final,
        Y_src_final,
        edge_src_all,
        full_epoch,
        seed
    )


    concat_model = train_final_source_model(
        src_concat_final,
        Y_src_final,
        edge_src_all,
        concat_epoch,
        seed
    )


    # --------------------------------------------------------
    # F. ZERO-SHOT D10 inference
    #
    # No D10 training.
    # No D10 label calibration.
    # --------------------------------------------------------

    low_tgt_std = predict(
        low_model,
        tgt_low_final,
        edge_tgt_all
    )


    full_tgt_std = predict(
        full_model,
        tgt_full_final,
        edge_tgt_all
    )


    concat_tgt_std = predict(
        concat_model,
        tgt_concat_final,
        edge_tgt_all
    )


    moe_tgt_std = (
        alpha[
            None,
            :
        ]
        *
        low_tgt_std
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
        full_tgt_std
    )


    # --------------------------------------------------------
    # G. Convert using SOURCE D9 expression statistics ONLY
    # --------------------------------------------------------

    low_pred = (
        low_tgt_std
        *
        Y_final_std
        +
        Y_final_mean
    )


    full_pred = (
        full_tgt_std
        *
        Y_final_std
        +
        Y_final_mean
    )


    concat_pred = (
        concat_tgt_std
        *
        Y_final_std
        +
        Y_final_mean
    )


    moe_pred = (
        moe_tgt_std
        *
        Y_final_std
        +
        Y_final_mean
    )


    model_predictions = {
        "D9toD10_LowExpert_GraphSAGE":
            low_pred,

        "D9toD10_FullExpert_GraphSAGE":
            full_pred,

        "D9toD10_SeparatePCA50x2_GraphSAGE":
            concat_pred,

        "D9toD10_FrozenStaticMoE_GraphSAGE":
            moe_pred,
    }


    # --------------------------------------------------------
    # H. Evaluate AFTER prediction
    #
    # D10 labels appear for the first time here.
    # --------------------------------------------------------

    for model_name, pred in model_predictions.items():

        for g, gene in enumerate(
            TARGET_GENES
        ):

            metric = evaluate(
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
                    "model":
                        model_name,

                    "gene":
                        gene,

                    "seed":
                        seed,

                    **metric,
                }
            )


    # --------------------------------------------------------
    # I. Save frozen gate
    # --------------------------------------------------------

    for g, gene in enumerate(
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


    # --------------------------------------------------------
    # J. Training metadata
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # K. Target predictions
    # --------------------------------------------------------

    for i, barcode in enumerate(
        tgt_barcodes
    ):

        row = {
            "barcode":
                barcode,

            "seed":
                seed,
        }


        for g, gene in enumerate(
            TARGET_GENES
        ):

            row[
                f"{gene}_true"
            ] = float(
                Y_tgt[
                    i,
                    g
                ]
            )


            row[
                f"{gene}_low"
            ] = float(
                low_pred[
                    i,
                    g
                ]
            )


            row[
                f"{gene}_full"
            ] = float(
                full_pred[
                    i,
                    g
                ]
            )


            row[
                f"{gene}_concat"
            ] = float(
                concat_pred[
                    i,
                    g
                ]
            )


            row[
                f"{gene}_moe"
            ] = float(
                moe_pred[
                    i,
                    g
                ]
            )


        prediction_rows.append(
            row
        )


# ============================================================
# 20. Save raw results
# ============================================================

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
    D10_OUT
    / "cross_slide_d9_to_d10_raw_metrics.csv",
    index=False
)


alphas.to_csv(
    D10_OUT
    / "cross_slide_d9_to_d10_gate_weights_raw.csv",
    index=False
)


training.to_csv(
    D10_OUT
    / "cross_slide_d9_to_d10_training.csv",
    index=False
)


predictions.to_csv(
    D10_OUT
    / "cross_slide_d9_to_d10_predictions.csv",
    index=False
)


# ============================================================
# 21. Performance summary
# ============================================================

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
        ),
    )
    .reset_index()
)


summary.to_csv(
    D10_OUT
    / "cross_slide_d9_to_d10_summary.csv",
    index=False
)


# ============================================================
# 22. Frozen D9 gate summary
# ============================================================

gate_summary = (
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
        ),
    )
    .reset_index()
)


gate_summary.to_csv(
    D10_OUT
    / "cross_slide_d9_to_d10_gate_weights_summary.csv",
    index=False
)


# ============================================================
# 23. Paired MoE comparisons
# ============================================================

comparison_rows = []


moe = metrics[
    metrics[
        "model"
    ]
    ==
    "D9toD10_FrozenStaticMoE_GraphSAGE"
]


baseline_names = [
    "D9toD10_LowExpert_GraphSAGE",
    "D9toD10_FullExpert_GraphSAGE",
    "D9toD10_SeparatePCA50x2_GraphSAGE",
]


for baseline_name in baseline_names:

    baseline = metrics[
        metrics[
            "model"
        ]
        ==
        baseline_name
    ]


    merged = moe.merge(
        baseline[
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
                "baseline":
                    baseline_name,

                "gene":
                    gene,

                "delta_pearson_mean":
                    float(
                        delta_r.mean()
                    ),

                "pearson_wins":
                    int(
                        (
                            delta_r > 0
                        ).sum()
                    ),

                "delta_r2_mean":
                    float(
                        delta_r2.mean()
                    ),

                "r2_wins":
                    int(
                        (
                            delta_r2 > 0
                        ).sum()
                    ),

                "delta_rmse_mean":
                    float(
                        delta_rmse.mean()
                    ),

                "rmse_wins":
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
    D10_OUT
    / "cross_slide_d9_to_d10_moe_comparison.csv",
    index=False
)


# ============================================================
# 24. Best model per gene
# ============================================================

best_rows = []


for gene in TARGET_GENES:

    g = summary[
        summary[
            "gene"
        ]
        ==
        gene
    ].copy()


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


best_models = pd.DataFrame(
    best_rows
)


best_models.to_csv(
    D10_OUT
    / "cross_slide_d9_to_d10_best_models.csv",
    index=False
)


# ============================================================
# 25. Final report
# ============================================================

print(
    "\n============================================"
)

print(
    "D9 -> D10 ZERO-SHOT CROSS-SLIDE TRANSFER"
)

print(
    "============================================"
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
    "\nD9-derived frozen gate:"
)


print(
    gate_summary.to_string(
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
    "\nBest models:"
)


print(
    best_models.to_string(
        index=False
    )
)


print(
    "\nIMPORTANT:"
)

print(
    "All PCA/scalers/model weights/gates "
    "were learned from D9 only."
)

print(
    "D10 expression labels were used only "
    "after prediction for evaluation."
)


print(
    "\nSaved:"
)

for p in [
    "cross_slide_expression_shift.csv",
    "cross_slide_feature_shift.csv",
    "cross_slide_d9_to_d10_training.csv",
    "cross_slide_d9_to_d10_raw_metrics.csv",
    "cross_slide_d9_to_d10_summary.csv",
    "cross_slide_d9_to_d10_gate_weights_raw.csv",
    "cross_slide_d9_to_d10_gate_weights_summary.csv",
    "cross_slide_d9_to_d10_moe_comparison.csv",
    "cross_slide_d9_to_d10_predictions.csv",
    "cross_slide_d9_to_d10_best_models.csv",
]:

    print(
        D10_OUT / p
    )
