# Overview

Computes a clonotype space to visually represent relationships between clonotype sequences using dimensionality reduction techniques, enabling intuitive exploration of immune repertoire diversity and structure. The block processes amino acid sequences from T-cell receptor (TCR) or B-cell receptor (BCR) clonotype data and generates 2D visualizations where similar clonotypes cluster together based on sequence similarity.

The algorithm employs a two-stage pipeline: first, amino acid sequences are converted into k-mer count vectors that capture sequence composition patterns, then dimensionality is reduced via truncated SVD followed by UMAP projection to 2D coordinates. This approach processes both bulk and single-cell VDJ data, concatenating multi-chain sequences when necessary, and outputs interactive visualizations with metadata overlay. The k-mer-based representation enables comparison of clonotypes based on shared sequence motifs, while UMAP preserves local neighborhood structure, ensuring that clonotypes with similar sequences appear close together in the visualization.

The block uses UMAP for dimensionality reduction. When using this block in your research, cite the UMAP publication (McInnes et al. 2018) listed below.

The following publication describes the methodology used:

> McInnes, L., Healy, J., Saul, N., & Großberger, L. (2018). UMAP: Uniform Manifold Approximation and Projection. _Journal of Open Source Software_ **3**, 861 (2018). [https://doi.org/10.21105/joss.00861](https://doi.org/10.21105/joss.00861)
