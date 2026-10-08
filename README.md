# Spatial Omics–Histology Transfer with Squidpy and GraphSAGE

A reproducible spatial-omics framework integrating **10x Genomics Visium H&E**, **Scanpy/Squidpy**, **DINOv2**, and **GraphSAGE** to study how histological resolution, spatial context, and anatomical distribution shift affect spatial gene-expression prediction.

<p align="center">
  <b>Spatial transcriptomics × Histology × Foundation models × Graph neural networks</b>
</p>

---

## Overview

This project asks a simple but important question:

> **How transferable is the relationship between H&E morphology and spatial gene expression across histological scales and independent tissue sections?**

The pipeline combines image-derived morphology with Visium spatial topology:

```text
10x Genomics Visium
        │
        ├── gene-expression matrix
        ├── spatial coordinates
        └── H&E image
                │
                ▼
          Scanpy / AnnData
                │
                ▼
              Squidpy
        spatial_neighbors()
                │
                ▼
          spatial graph
                │
       ┌────────┴────────┐
       ▼                 ▼
 low-resolution      full-resolution
   H&E patch           H&E patch
       │                 │
       └────────┬────────┘
                ▼
             DINOv2
                │
                ▼
      multiscale morphology
                │
                ▼
            GraphSAGE
                │
                ▼
     spatial gene prediction
```

Squidpy is used to construct the Visium neighborhood graph stored in:

```python
adata.obsp["spatial_connectivities"]
```

This graph is subsequently used by GraphSAGE for spatial message passing.

---

## Key questions

The project evaluates three related problems:

1. **Resolution dependence**  
   Does spatial gene-expression predictability change between low- and full-resolution histology?

2. **Multiscale complementarity**  
   Do coarse tissue context and fine histological detail provide complementary molecular information?

3. **Cross-slide generalization**  
   Can a morphology→expression model trained on one tissue section transfer to an independent section without target-label retraining?

---

## Datasets

Three public 10x Genomics Visium mouse-brain H&E datasets are used.

| ID | Dataset | Section | Role | Tissue spots |
|---|---|---|---|---:|
| **D9** | `V1_Adult_Mouse_Brain` | Coronal | Source | 2,688 |
| **D10** | `V1_Mouse_Brain_Sagittal_Anterior` | Sagittal anterior | Independent target | 2,696 |
| **D11** | `Visium_Fresh_Frozen_Adult_Mouse_Brain` | Coronal | Independent target | 2,797 |

Raw Visium matrices, whole-slide TIFF images, extracted patches, H5AD objects, and DINOv2 embedding arrays are intentionally excluded from this repository because of file size.

---

## Target genes

The current experiments focus on four genes with distinct spatial expression patterns:

- **Camk2n1**
- **Slc17a7**
- **Plp1**
- **Cldn11**

---

## Multiscale histology representation

For each Visium spot, two H&E views are extracted using approximately the same tissue field of view.

The crop width is defined as:

```text
2.5 × Visium spot diameter
```

Two image representations are generated:

- **Low-resolution context** — emphasizes broader tissue organization.
- **Full-resolution morphology** — preserves finer local histological detail.

Both views are resized to `224 × 224` pixels before feature extraction.

### DINOv2 embeddings

Morphology features are extracted with:

```text
DINOv2 ViT-S/14
```

The normalized CLS token provides a **384-dimensional** representation for each patch.

### Multiscale fusion

Low- and full-resolution embeddings are independently projected with PCA:

```text
Low PCA50
Full PCA50
```

The main multiscale representation is:

```text
SeparatePCA50x2
= Low PCA50 + Full PCA50
= 100 dimensions
```

All PCA models and feature scalers are fitted using source/training data only.

---

## Spatial prediction model

Gene expression is predicted using a two-layer **GraphSAGE** network operating on the Squidpy-derived Visium neighborhood graph.

Main settings:

```text
Hidden dimension      64
Dropout               0.20
Optimizer             AdamW
Learning rate         1e-3
Weight decay          1e-4
Maximum epochs        500
Early-stop patience   60
Random seeds          42–46
```

The four genes are predicted jointly.

Model variants include:

- `LowExpert_GraphSAGE`
- `FullExpert_GraphSAGE`
- `SeparatePCA50x2_GraphSAGE`
- `FrozenStaticMoE_GraphSAGE`

---

## Strict cross-slide zero-shot transfer

For strict zero-shot transfer, all learned quantities are estimated from **D9 only**:

- PCA transformations
- feature scalers
- GraphSAGE parameters
- source expression standardization
- mixture weights

For D10 and D11:

- no target gene-expression labels are used for fitting;
- no target-specific model retraining is performed;
- the target spatial graph is used only for unlabeled GraphSAGE message passing.

Target gene-expression values are accessed only after prediction for evaluation.

---

## Main result: anatomical compatibility matters

Strict zero-shot transfer from the D9 coronal source was substantially stronger to the independent D11 coronal target than to the D10 sagittal target.

### Strict zero-shot Pearson correlation

| Gene | D9 → D10 sagittal | D9 → D11 coronal | Δr |
|---|---:|---:|---:|
| Camk2n1 | 0.159 | **0.431** | +0.273 |
| Slc17a7 | 0.313 | **0.581** | +0.268 |
| Plp1 | 0.139 | **0.577** | +0.438 |
| Cldn11 | 0.156 | **0.498** | +0.342 |

For **Plp1** and **Cldn11**, the absolute-expression calibration also improved:

| Gene | D10 strict R² | D11 strict R² |
|---|---:|---:|
| Plp1 | -0.543 | **0.232** |
| Cldn11 | -1.424 | **0.092** |

The result is particularly strong for Plp1 and Cldn11, where the coronal target retained both stronger spatial association and improved calibration.

---

## Matched cross-target comparison

To avoid relying only on target-specific best-model selection, D10 and D11 were also compared using matched combinations of:

- gene;
- model architecture;
- alignment strategy;
- random seed.

This produced **60 matched runs per gene**.

| Gene | Mean Δ Pearson (D11 − D10) | D11 higher |
|---|---:|---:|
| Camk2n1 | +0.153 | 78.3% |
| Slc17a7 | +0.044 | 55.0% |
| Plp1 | **+0.235** | **88.3%** |
| Cldn11 | **+0.190** | **86.7%** |

Across all genes and models under the strict no-alignment condition:

```text
D10 mean Pearson r   0.155
D11 mean Pearson r   0.411
Mean Δr             +0.256
D11 wins             95%
```

These observations are **consistent with anatomical and tissue-composition compatibility influencing morphology-to-expression generalization**.

Because only one sagittal and one independent coronal target are currently included, the experiment does **not** establish a causal effect of section orientation itself.

---

## Label-free feature alignment

Two target-label-free domain-alignment strategies are evaluated.

### Z-score moment alignment

Target morphology features are standardized using target feature statistics and remapped to the source mean and variance.

### CORAL

Target PCA features are covariance-aligned to the source feature distribution.

Neither method uses target gene-expression labels during adaptation.

Selected genes benefit from label-free alignment; however, gains in Pearson correlation do not consistently translate into improved R². This indicates that global feature-distribution matching alone does not fully resolve cross-slide expression calibration.

---

## Representative findings

The experiments support three main observations:

### 1. Resolution dependence is gene-specific

Different genes favor different histological scales. A single resolution is therefore not universally optimal for morphology-based molecular prediction.

### 2. Multiscale morphology can provide complementary signal

Within-slide analyses show that combining coarse contextual morphology and fine histological detail can improve prediction for selected targets, particularly **Plp1** and **Cldn11**.

### 3. Transferability depends on anatomical context

The strongest cross-slide result is not simply explained by marginal embedding similarity. D9→D11 coronal transfer remains substantially stronger than D9→D10 sagittal transfer, suggesting that preservation of the morphology→expression relationship depends on source–target anatomical compatibility.

---

## Repository structure

```text
spatial-omics-histology-transfer/
├── README.md
├── environment.yml
├── .gitignore
│
├── scripts/
│   ├── d9_source/
│   ├── d10_sagittal/
│   └── d11_coronal/
│
├── results/
│   ├── d9/
│   ├── d10/
│   ├── d11/
│   └── final/
│
└── figures/
    └── final/
```

### `scripts/d9_source`

Source-slide preprocessing, multiscale feature extraction, spatial prediction, and source-model experiments.

### `scripts/d10_sagittal`

Independent sagittal-slide preprocessing, within-slide replication, strict D9→D10 transfer, and label-free alignment.

### `scripts/d11_coronal`

Independent coronal-slide preprocessing, D9→D11 transfer, D10-vs-D11 matched comparisons, and final result aggregation.

### `results/final`

Compact final CSV tables used for cross-slide comparison and manuscript-level summaries.

### `figures/final`

Main result figures generated from the final analyses.

---

## Installation

A Conda environment definition is provided:

```bash
conda env create -f environment.yml
conda activate spatialomics
```

Core dependencies include:

- Python 3.11
- Scanpy
- Squidpy
- NumPy
- Pandas
- SciPy
- scikit-learn
- PyTorch
- torchvision
- PyTorch Geometric
- Matplotlib
- Pillow

DINOv2 is loaded through PyTorch Hub in the embedding-extraction scripts.

---

## Reproducibility notes

Large derived artifacts are excluded from GitHub:

```text
*.tif / *.tiff
*.h5
*.h5ad
*.npy
patches_lowres/
patches_fullres/
prediction-level outputs
```

The repository instead provides the analysis scripts, compact result tables, and figures required to reproduce the workflow from the corresponding public 10x Genomics datasets.

When reproducing the experiments, keep the same:

- Visium sample IDs;
- spot-centered field-of-view definition;
- DINOv2 backbone;
- scale-specific PCA procedure;
- Squidpy spatial graph construction;
- GraphSAGE architecture;
- source-only preprocessing for cross-slide transfer;
- random seeds.

---

## Evaluation metrics

Models are evaluated using:

- **Pearson correlation**
- **Spearman correlation**
- **R²**
- **RMSE**

Pearson/Spearman quantify spatial association and relative expression structure, whereas R² and RMSE are used to assess absolute-expression calibration.

This distinction is important: a model can recover spatial ordering while remaining poorly calibrated in absolute expression units.

---

## Interpretation

The current results support the following working conclusion:

> **Histological resolution dependence is gene-specific and transfer-regime-dependent. Multiscale morphology provides complementary molecular information within slides, while cross-slide generalization depends strongly on anatomical compatibility between source and target tissue sections.**

The current study should be interpreted as a focused methodological analysis rather than a definitive test of section-orientation effects. Additional independent coronal and sagittal sections will be required to separate orientation, tissue composition, staining, and slide-specific sources of distribution shift.

---

## Figures

Key final figures are available in:

```text
figures/final/
```

including strict zero-shot D10-vs-D11 comparisons, matched cross-target analyses, and within-slide versus cross-slide performance summaries.

---

## Technologies

`Spatial omics` · `Spatial transcriptomics` · `Squidpy` · `Scanpy` · `AnnData` · `10x Genomics Visium` · `Histology` · `DINOv2` · `GraphSAGE` · `PyTorch Geometric` · `Computational pathology` · `Multimodal learning`

---

## Citation

If you use this repository in academic work, please cite the associated manuscript when available.

A formal citation will be added after manuscript submission.

---

## License

Please add a repository license before reuse or redistribution outside the scope permitted by the original public datasets and third-party software dependencies.
