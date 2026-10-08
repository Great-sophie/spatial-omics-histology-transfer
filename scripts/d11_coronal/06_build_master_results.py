from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE = Path(
    "/media/sophie/Data2T/Biomedical AI  skill"
)

D10 = BASE / "D10_mouse_brain_external_validation"
D11 = BASE / "D11_mouse_brain_coronal_validation"

D10_OUT = D10 / "outputs"
D11_OUT = D11 / "outputs"

FIG = D11 / "figures"
FIG.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

# D10: models retrained within D10
D10_WITHIN = (
    D10_OUT
    / "external_replication_summary.csv"
)

# D9 -> D10 sagittal transfer
D10_TRANSFER = (
    D10_OUT
    / "unsupervised_alignment_summary.csv"
)

# D9 -> D11 coronal transfer
D11_TRANSFER = (
    D11_OUT
    / "coronal_unsupervised_alignment_summary.csv"
)


TARGET_GENES = [
    "Camk2n1",
    "Slc17a7",
    "Plp1",
    "Cldn11",
]


# ============================================================
# Check files
# ============================================================

for p in [
    D10_WITHIN,
    D10_TRANSFER,
    D11_TRANSFER,
]:
    print(
        p,
        "FOUND" if p.exists() else "MISSING"
    )

    if not p.exists():
        raise FileNotFoundError(str(p))


# ============================================================
# Load
# ============================================================

within = pd.read_csv(
    D10_WITHIN
)

d10 = pd.read_csv(
    D10_TRANSFER
)

d11 = pd.read_csv(
    D11_TRANSFER
)


# pandas may interpret literal "None" as NaN
for df in [d10, d11]:
    df["alignment"] = (
        df["alignment"]
        .fillna("None")
        .astype(str)
    )


print(
    "\nD10 within:",
    within.shape
)

print(
    "D10 transfer:",
    d10.shape
)

print(
    "D11 transfer:",
    d11.shape
)


# ============================================================
# Column QC
# ============================================================

within_required = [
    "model",
    "gene",
    "pearson_mean",
    "r2_mean",
]


transfer_required = [
    "alignment",
    "model",
    "gene",
    "pearson_mean",
    "r2_mean",
]


for c in within_required:
    if c not in within.columns:
        raise KeyError(
            f"D10 within missing: {c}"
        )


for name, df in [
    ("D10 transfer", d10),
    ("D11 transfer", d11),
]:
    for c in transfer_required:
        if c not in df.columns:
            raise KeyError(
                f"{name} missing: {c}"
            )


# ============================================================
# Helper
# ============================================================

def best_row_by_pearson(df):

    if len(df) == 0:
        return None

    return df.loc[
        df["pearson_mean"].idxmax()
    ]


def get_best_within(
    gene
):

    g = within[
        within["gene"] == gene
    ].copy()

    return best_row_by_pearson(
        g
    )


def get_strict(
    df,
    gene
):

    g = df[
        (
            df["gene"] == gene
        )
        &
        (
            df["alignment"] == "None"
        )
    ].copy()

    return best_row_by_pearson(
        g
    )


def get_label_free_best(
    df,
    gene
):

    # Only target-label-free adaptation methods.
    # "Best" is retrospective/descriptive because
    # target labels were used to compare final metrics.
    g = df[
        (
            df["gene"] == gene
        )
        &
        (
            df["alignment"].isin(
                [
                    "ZScore",
                    "CORAL"
                ]
            )
        )
    ].copy()

    return best_row_by_pearson(
        g
    )


# ============================================================
# Build master table
# ============================================================

rows = []


for gene in TARGET_GENES:

    w = get_best_within(
        gene
    )

    d10_strict = get_strict(
        d10,
        gene
    )

    d10_adapt = get_label_free_best(
        d10,
        gene
    )

    d11_strict = get_strict(
        d11,
        gene
    )

    d11_adapt = get_label_free_best(
        d11,
        gene
    )


    row = {
        "gene":
            gene,

        # --------------------------------------
        # D10 within-slide
        # --------------------------------------
        "d10_within_best_model":
            w["model"],

        "d10_within_best_r":
            float(
                w["pearson_mean"]
            ),

        "d10_within_best_r2":
            float(
                w["r2_mean"]
            ),

        # --------------------------------------
        # D9 -> D10 strict
        # --------------------------------------
        "d10_strict_model":
            d10_strict["model"],

        "d10_strict_r":
            float(
                d10_strict[
                    "pearson_mean"
                ]
            ),

        "d10_strict_r2":
            float(
                d10_strict[
                    "r2_mean"
                ]
            ),

        # --------------------------------------
        # D9 -> D10 label-free best
        # --------------------------------------
        "d10_label_free_alignment":
            d10_adapt[
                "alignment"
            ],

        "d10_label_free_model":
            d10_adapt[
                "model"
            ],

        "d10_label_free_r":
            float(
                d10_adapt[
                    "pearson_mean"
                ]
            ),

        "d10_label_free_r2":
            float(
                d10_adapt[
                    "r2_mean"
                ]
            ),

        # --------------------------------------
        # D9 -> D11 strict
        # --------------------------------------
        "d11_strict_model":
            d11_strict[
                "model"
            ],

        "d11_strict_r":
            float(
                d11_strict[
                    "pearson_mean"
                ]
            ),

        "d11_strict_r2":
            float(
                d11_strict[
                    "r2_mean"
                ]
            ),

        # --------------------------------------
        # D9 -> D11 label-free best
        # --------------------------------------
        "d11_label_free_alignment":
            d11_adapt[
                "alignment"
            ],

        "d11_label_free_model":
            d11_adapt[
                "model"
            ],

        "d11_label_free_r":
            float(
                d11_adapt[
                    "pearson_mean"
                ]
            ),

        "d11_label_free_r2":
            float(
                d11_adapt[
                    "r2_mean"
                ]
            ),
    }


    # ------------------------------------------
    # Key deltas
    # ------------------------------------------

    row[
        "strict_delta_d11_minus_d10_r"
    ] = (
        row[
            "d11_strict_r"
        ]
        -
        row[
            "d10_strict_r"
        ]
    )


    row[
        "label_free_delta_d11_minus_d10_r"
    ] = (
        row[
            "d11_label_free_r"
        ]
        -
        row[
            "d10_label_free_r"
        ]
    )


    row[
        "d10_strict_drop_vs_within_r"
    ] = (
        row[
            "d10_strict_r"
        ]
        -
        row[
            "d10_within_best_r"
        ]
    )


    row[
        "d10_label_free_recovery_r"
    ] = (
        row[
            "d10_label_free_r"
        ]
        -
        row[
            "d10_strict_r"
        ]
    )


    row[
        "d11_label_free_change_r"
    ] = (
        row[
            "d11_label_free_r"
        ]
        -
        row[
            "d11_strict_r"
        ]
    )


    rows.append(
        row
    )


master = pd.DataFrame(
    rows
)


master.to_csv(
    D11_OUT
    / "final_master_gene_results.csv",
    index=False
)


# ============================================================
# Long-format version for plotting/statistics
# ============================================================

long_rows = []


for _, row in master.iterrows():

    gene = row[
        "gene"
    ]


    long_rows.extend(
        [
            {
                "gene":
                    gene,

                "regime":
                    "D10 within-slide retraining",

                "target":
                    "D10 sagittal",

                "transfer_type":
                    "within-slide",

                "alignment":
                    "supervised_retraining",

                "model":
                    row[
                        "d10_within_best_model"
                    ],

                "pearson_r":
                    row[
                        "d10_within_best_r"
                    ],

                "r2":
                    row[
                        "d10_within_best_r2"
                    ],
            },

            {
                "gene":
                    gene,

                "regime":
                    "D9→D10 strict zero-shot",

                "target":
                    "D10 sagittal",

                "transfer_type":
                    "cross-slide",

                "alignment":
                    "None",

                "model":
                    row[
                        "d10_strict_model"
                    ],

                "pearson_r":
                    row[
                        "d10_strict_r"
                    ],

                "r2":
                    row[
                        "d10_strict_r2"
                    ],
            },

            {
                "gene":
                    gene,

                "regime":
                    "D9→D10 label-free best",

                "target":
                    "D10 sagittal",

                "transfer_type":
                    "cross-slide",

                "alignment":
                    row[
                        "d10_label_free_alignment"
                    ],

                "model":
                    row[
                        "d10_label_free_model"
                    ],

                "pearson_r":
                    row[
                        "d10_label_free_r"
                    ],

                "r2":
                    row[
                        "d10_label_free_r2"
                    ],
            },

            {
                "gene":
                    gene,

                "regime":
                    "D9→D11 strict zero-shot",

                "target":
                    "D11 coronal",

                "transfer_type":
                    "cross-slide",

                "alignment":
                    "None",

                "model":
                    row[
                        "d11_strict_model"
                    ],

                "pearson_r":
                    row[
                        "d11_strict_r"
                    ],

                "r2":
                    row[
                        "d11_strict_r2"
                    ],
            },

            {
                "gene":
                    gene,

                "regime":
                    "D9→D11 label-free best",

                "target":
                    "D11 coronal",

                "transfer_type":
                    "cross-slide",

                "alignment":
                    row[
                        "d11_label_free_alignment"
                    ],

                "model":
                    row[
                        "d11_label_free_model"
                    ],

                "pearson_r":
                    row[
                        "d11_label_free_r"
                    ],

                "r2":
                    row[
                        "d11_label_free_r2"
                    ],
            },
        ]
    )


long_df = pd.DataFrame(
    long_rows
)


long_df.to_csv(
    D11_OUT
    / "final_master_long_results.csv",
    index=False
)


# ============================================================
# Primary strict-zero-shot table
# ============================================================

primary = master[
    [
        "gene",
        "d10_strict_model",
        "d10_strict_r",
        "d10_strict_r2",
        "d11_strict_model",
        "d11_strict_r",
        "d11_strict_r2",
        "strict_delta_d11_minus_d10_r",
    ]
].copy()


primary.to_csv(
    D11_OUT
    / "final_primary_strict_zero_shot.csv",
    index=False
)


# ============================================================
# Secondary label-free table
# ============================================================

secondary = master[
    [
        "gene",
        "d10_label_free_alignment",
        "d10_label_free_model",
        "d10_label_free_r",
        "d10_label_free_r2",
        "d11_label_free_alignment",
        "d11_label_free_model",
        "d11_label_free_r",
        "d11_label_free_r2",
        "label_free_delta_d11_minus_d10_r",
    ]
].copy()


secondary.to_csv(
    D11_OUT
    / "final_secondary_label_free_alignment.csv",
    index=False
)


# ============================================================
# Figure 09
# Main cross-slide figure
#
# Primary:
# D10 strict vs D11 strict
# ============================================================

plot = master.set_index(
    "gene"
).loc[
    TARGET_GENES
].reset_index()


x = np.arange(
    len(
        TARGET_GENES
    )
)

width = 0.36


fig, ax = plt.subplots(
    figsize=(
        9,
        5.7
    )
)


b1 = ax.bar(
    x - width / 2,
    plot[
        "d10_strict_r"
    ],
    width,
    label="D9 → D10 sagittal"
)


b2 = ax.bar(
    x + width / 2,
    plot[
        "d11_strict_r"
    ],
    width,
    label="D9 → D11 coronal"
)


ax.axhline(
    0,
    linewidth=1
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    TARGET_GENES
)

ax.set_ylim(
    0,
    max(
        0.7,
        float(
            max(
                plot[
                    "d10_strict_r"
                ].max(),
                plot[
                    "d11_strict_r"
                ].max()
            )
        )
        +
        0.1
    )
)


ax.set_ylabel(
    "Pearson r"
)


ax.set_title(
    "Cross-slide strict zero-shot transfer"
)


ax.legend(
    frameon=False
)


for bars in [
    b1,
    b2
]:

    for bar in bars:

        h = bar.get_height()

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


plt.tight_layout()


plt.savefig(
    FIG
    / "09_final_strict_zero_shot_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# Figure 10
# Within-slide vs transfer
# ============================================================

regimes = [
    (
        "D10 within-slide",
        "d10_within_best_r"
    ),
    (
        "D9→D10 strict",
        "d10_strict_r"
    ),
    (
        "D9→D11 strict",
        "d11_strict_r"
    ),
]


width = 0.23


fig, ax = plt.subplots(
    figsize=(
        10,
        5.8
    )
)


offsets = [
    -width,
    0,
    width
]


for (
    label,
    column
), offset in zip(
    regimes,
    offsets
):

    ax.bar(
        x + offset,
        plot[
            column
        ],
        width,
        label=label
    )


ax.axhline(
    0,
    linewidth=1
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
    "Within-slide prediction versus cross-slide transfer"
)


ax.legend(
    frameon=False
)


plt.tight_layout()


plt.savefig(
    FIG
    / "10_final_within_vs_cross_slide.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# Figure 11
# Label-free adaptation effect
# ============================================================

adapt_rows = []


for _, row in master.iterrows():

    adapt_rows.append(
        {
            "gene":
                row["gene"],

            "target":
                "D10 sagittal",

            "delta_r":
                (
                    row[
                        "d10_label_free_r"
                    ]
                    -
                    row[
                        "d10_strict_r"
                    ]
                ),
        }
    )


    adapt_rows.append(
        {
            "gene":
                row["gene"],

            "target":
                "D11 coronal",

            "delta_r":
                (
                    row[
                        "d11_label_free_r"
                    ]
                    -
                    row[
                        "d11_strict_r"
                    ]
                ),
        }
    )


adapt = pd.DataFrame(
    adapt_rows
)


adapt.to_csv(
    D11_OUT
    / "final_label_free_alignment_effect.csv",
    index=False
)


fig, ax = plt.subplots(
    figsize=(
        9,
        5.5
    )
)


d10a = (
    adapt[
        adapt[
            "target"
        ]
        ==
        "D10 sagittal"
    ]
    .set_index(
        "gene"
    )
    .loc[
        TARGET_GENES
    ]
)


d11a = (
    adapt[
        adapt[
            "target"
        ]
        ==
        "D11 coronal"
    ]
    .set_index(
        "gene"
    )
    .loc[
        TARGET_GENES
    ]
)


ax.bar(
    x - 0.18,
    d10a[
        "delta_r"
    ],
    0.36,
    label="D10 sagittal"
)


ax.bar(
    x + 0.18,
    d11a[
        "delta_r"
    ],
    0.36,
    label="D11 coronal"
)


ax.axhline(
    0,
    linewidth=1
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    TARGET_GENES
)


ax.set_ylabel(
    "Δ Pearson r\n"
    "(best label-free alignment − strict zero-shot)"
)


ax.set_title(
    "Effect of label-free feature alignment"
)


ax.legend(
    frameon=False
)


plt.tight_layout()


plt.savefig(
    FIG
    / "11_final_label_free_alignment_effect.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# Console summary
# ============================================================

print(
    "\n==========================================="
)

print(
    "FINAL MASTER RESULTS"
)

print(
    "==========================================="
)


print(
    "\nPRIMARY — STRICT ZERO-SHOT"
)

print(
    primary.to_string(
        index=False
    )
)


print(
    "\nSECONDARY — LABEL-FREE ALIGNMENT"
)

print(
    secondary.to_string(
        index=False
    )
)


print(
    "\nMASTER"
)

print(
    master.to_string(
        index=False
    )
)


print(
    "\nImportant interpretation:"
)

print(
    "1. Strict zero-shot D10 vs D11 is the primary "
    "cross-slide comparison."
)

print(
    "2. Label-free 'best' alignment is descriptive, "
    "because the winning alignment was identified "
    "after target performance evaluation."
)

print(
    "3. D10 within-slide retraining is a reference "
    "for available morphology-expression signal, "
    "not a direct external-transfer comparator."
)

print(
    "4. D11 coronal > D10 sagittal is consistent with "
    "anatomical-distribution compatibility, but one "
    "target slide per orientation does not establish "
    "a causal orientation effect."
)


print(
    "\nSaved CSVs:"
)

for name in [
    "final_master_gene_results.csv",
    "final_master_long_results.csv",
    "final_primary_strict_zero_shot.csv",
    "final_secondary_label_free_alignment.csv",
    "final_label_free_alignment_effect.csv",
]:

    print(
        D11_OUT / name
    )


print(
    "\nSaved figures:"
)

for name in [
    "09_final_strict_zero_shot_comparison.png",
    "10_final_within_vs_cross_slide.png",
    "11_final_label_free_alignment_effect.png",
]:

    print(
        FIG / name
    )
