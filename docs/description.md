# Overview

Computes a sequence space to visually represent relationships between input sequences — clonotypes or peptides — using dimensionality reduction techniques, enabling intuitive exploration of repertoire and library diversity. The block processes amino acid sequences from T-cell or B-cell receptor (TCR/BCR) clonotype data or peptide datasets, and generates 2D visualizations where similar sequences cluster together based on sequence similarity.

The algorithm employs a two-stage pipeline: first, amino acid sequences are converted into k-mer count vectors that capture sequence composition patterns, then dimensionality is reduced via truncated SVD followed by UMAP projection to 2D coordinates. This approach processes bulk and single-cell VDJ data (concatenating multi-chain sequences when necessary) as well as peptide data, and outputs interactive visualizations with metadata overlay. The k-mer-based representation enables comparison of sequences based on shared motifs, while UMAP preserves local neighborhood structure, ensuring that sequences with similar composition appear close together in the visualization.

The block uses UMAP for dimensionality reduction. When using this block in your research, cite the UMAP publication (McInnes et al. 2018) listed below.

The following publication describes the methodology used:

> McInnes, L., Healy, J., Saul, N., & Großberger, L. (2018). UMAP: Uniform Manifold Approximation and Projection. _Journal of Open Source Software_ **3**, 861 (2018). [https://doi.org/10.21105/joss.00861](https://doi.org/10.21105/joss.00861)
