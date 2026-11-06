# Unsupervised Classification of Character Fonts Using K-Means Clustering

Clustering 20×20 pixel images of characters from three distinct fonts, with
dimensionality reduction and an analysis of cluster quality against
the true font labels from scratch. Dimensionality reduction (PCA via eigendecomposition and
SVD) and all cluster quality metrics are implemented directly with NumPy;
scikit-learn is used only for the K-means fit itself.

## Overview

The goal is to test how well **unsupervised** K-means clustering can recover
three known font classes, **CURLZ**, **JOKERMAN**, and **ONYX**, from raw
pixel data alone, without using the labels during clustering. The labels are
held back and used only afterward, to measure how pure the discovered clusters
turned out to be.

- **Data:** 728 character images (bold, non-italic), each a 20×20 black-and-white
  grid flattened to 400 pixel features, drawn from the three fonts.
- **Reduction:** standardization → correlation-matrix PCA → principal-component
  analysis of variance explained.
- **Clustering:** K-means swept over k = 1…20, then analyzed at the k matching
  the three true classes.
- **Evaluation:** cluster dispersion, centroids, radii, inter-cluster
  separability, and per class purity vs. the true fonts.

## Method

1. **Data preparation** — Three font datasets are filtered (missing values and
   non-pixel metadata removed, restricted to bold/non-italic cases) and
   combined into a single labeled matrix (`DATA.csv`).
2. **Standardization** — Features are z-scored to mean 0 / standard deviation 1,
   and a correlation matrix is computed to examine the most and
   least correlated pixel pairs.
3. **Dimensionality reduction** — PCA is performed **from scratch** via
   eigendecomposition of the correlation matrix; the percentage of variance
   explained (PVE) is used to choose the number of components retained. The
   reduction is cross-checked with an SVD-based reconstruction of the data.
4. **Visualization** — The first three principal components are plotted in 3D,
   colored by font, to show class structure in the reduced space.
5. **K-means** — Clustering is run for k = 1…20 with 100 random initializations
   per k, recording the best within-cluster dispersion and the computation time
   at each k. A normalized performance curve and an elbow analysis are used to
   assess the number of clusters.
6. **Cluster-quality analysis** — Custom functions compute each cluster's
   centroid, dispersion, and radius, a cluster-to-cluster separability matrix,
   and a **purity** measure comparing the discovered clusters against the true
   font labels.
7. **Label-informed baseline** — A "high-interest clustering" is built from the
   true-class centroids, and K-means is re-run seeded with those centroids, to
   compare label-informed grouping against the purely unsupervised result.

## Repository contents

| File | Description |
|------|-------------|
| `code.ipynb` | Full analysis notebook — data prep, PCA/SVD, K-means, and cluster evaluation |
| `DATA.csv` | The combined, filtered three-font dataset used by the notebook |
| `Report.pdf` | Written report with figures, derivations, and discussion of results |

## Tech stack

- **Python** — NumPy, pandas, matplotlib
- **scikit-learn** — `KMeans` (used only for the clustering fit)
- Jupyter Notebook

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook code.ipynb
```

The notebook reads `DATA.csv` from the repository root. (The first cells that
rebuild `DATA.csv` from the three original per-font CSVs expect those source
files locally; the combined `DATA.csv` is included so the rest of the notebook
runs without them.)

## Background

This project was developed for a graduate course in mathematical machine
learning. The character-image data comes from the UCI Character Font Images
dataset. The emphasis is on implementing the underlying mathematics,
standardization, PCA by eigendecomposition and SVD, and the clustering-quality
metrics directly.
