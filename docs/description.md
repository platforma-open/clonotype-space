# Clonotype Space

This block computes a "clonotype space" to visually represent the relationship between clonotype sequences using dimensionality reduction techniques. In adaptive immunity, clonotypes represent unique T-cell receptor (TCR) or B-cell receptor (BCR) sequences characterized by their amino acid sequences, which determine antigen specificity. The clonotype space provides an intuitive 2D visualization where similar clonotypes cluster together, enabling researchers to explore immune repertoire diversity, identify clonal expansion patterns, analyze disease biomarkers, and monitor vaccine responses.

The algorithm employs a two-stage pipeline: first, amino acid sequences are converted into k-mer count vectors, then dimensionality is reduced via truncated SVD followed by UMAP projection to 2D coordinates. This approach processes both bulk and single-cell VDJ data, concatenating multi-chain sequences when necessary, and outputs interactive visualizations with metadata overlay.

If you use this block in your research, please cite:

>  McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.