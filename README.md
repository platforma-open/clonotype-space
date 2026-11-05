# Sequence Space

Map an antibody, TCR, or peptide library in two dimensions so that similar sequences sit close together. This Platforma block converts sequences into numeric features — k-mer composition or protein language model embeddings — reduces them with UMAP, and renders an interactive 2D map you can color by any property produced by any upstream block.

Open-source analysis block for Platforma, the biologics discovery platform by MiLaboratories. For the full no-code workflow, see [platforma.bio](https://platforma.bio/).

> **Naming:** this block appears as **Sequence Space** in the Platforma app. Older documentation and this repository name call it **Clonotype Space**. They are the same block.

## What it does

The Sequence Space block gives you one picture of an entire repertoire or library. Every clonotype, variant, or peptide becomes a point; points that are close together have similar sequences. Instead of reading a table row by row, you see the structure of the library — which regions are dense, which are sparse, and where any subset of interest sits relative to everything else.

Two feature modes are available:

* **Sequence features** — sequences are converted to k-mer count vectors (3-mers for amino acids, 6-mers for nucleotides, positional 2-mers for peptides shorter than 10 aa), reduced with truncated SVD, then projected to 2D with UMAP. No model required; works on any sequence.
* **Embeddings** — vectors from the [Sequence Embeddings](https://github.com/platforma-open/sequence-embeddings) block are centered, PCA-reduced, L2-normalized, and projected with UMAP. Protein language models group sequences by learned biological similarity, so related sequences can land together even when their amino acids differ.

Multi-chain sequences are concatenated before featurization, so a paired heavy/light or TRA/TRB clonotype is placed as a single point. Any clonotype-level column in the project — abundance, enrichment, cluster ID, liability score, binding data, sample metadata — can be mapped onto the plot as color or highlight. Selecting points opens a multiple sequence alignment for that selection.

## Inputs & outputs

* **Input:** amino acid or nucleotide sequences from bulk or single-cell V(D)J clonotypes ([MiXCR Clonotyping](https://github.com/platforma-open/mixcr-clonotyping), [Import V(D)J Data](https://github.com/platforma-open/import-vdj-data)), scFv constructs, peptide libraries ([Peptide Profiling](https://github.com/platforma-open/peptide-extraction)), or amplicon variants ([Amplicon Profiling](https://github.com/platforma-open/synthetic-repertoire-profiler)) — or embedding vectors from [Sequence Embeddings](https://github.com/platforma-open/sequence-embeddings)
* **Output:** two UMAP coordinates per sequence, exposed as columns for downstream blocks, plus an interactive scatter plot with metadata overlay and a multiple sequence alignment panel for selected points

## Specifications

| | |
|---|---|
| Block title in app | Sequence Space |
| Feature modes | k-mer composition (SVD → UMAP); protein language model embeddings (PCA → UMAP) |
| Sequence types | Amino acid, nucleotide |
| Accepted inputs | Bulk and single-cell V(D)J clonotypes, scFv constructs, peptides, amplicon variants, sequence embeddings |
| Key parameters | Number of neighbors (≥ 2), minimum distance (0–1) |
| Compute | CPU by default; optional NVIDIA GPU (RAPIDS cuML) accelerates SVD and UMAP |
| Outputs | UMAP Dim1 / UMAP Dim2 per sequence; interactive scatter plot; multiple sequence alignment |
| Built on | [UMAP](https://github.com/lmcinnes/umap), scikit-learn, RAPIDS cuML |

## Use cases

* **Library diversity:** see whether a repertoire or synthetic library spreads across sequence space or collapses into a few dense regions.
* **Selection tracking:** overlay values from [Enrichment Analysis](https://github.com/platforma-open/clonotype-enrichment) to see which regions of the library were enriched across selection rounds.
* **Lead diversity check:** confirm candidates from [Lead Selection](https://github.com/platforma-open/antibody-tcr-lead-selection) are distributed across the library rather than concentrated in one neighborhood.
* **Developability in context:** overlay liability scores from [Antibody Sequence Liabilities](https://github.com/platforma-open/antibody-sequence-liabilities) to see how they relate to enrichment and diversity.
* **Functional data overlay:** map binding or affinity measurements onto the plot to see where active sequences sit.
* **Expanded clones:** overlay differential abundance results to locate treatment- or infection-expanded clonotypes within the full repertoire.
* **Cluster inspection:** color by cluster assignments from [Sequence Clustering](https://github.com/platforma-open/clonotype-clustering) or [Embedding Clustering](https://github.com/platforma-open/embedding-clustering) to check how well clusters separate.


## FAQ

### What is a sequence space map?

A 2D projection of a sequence library in which each point is one sequence and distance approximates sequence similarity. It lets you judge diversity, spot dense families, and see where a subset of interest sits relative to the whole library.

### Can I use nucleotide sequences?

Yes. Choose amino acid or nucleotide as the sequence type; nucleotide input uses 6-mers instead of 3-mers.

### Does it work on short peptides?

Yes. When the shortest peptide in the input is under 10 amino acids, the block switches to positional 2-mers so that short sequences are still separated meaningfully.

### Which feature mode should I use?

Use **sequence features** for fast, model-free maps driven by sequence composition. Use **embeddings** when you want sequences grouped by learned biological similarity rather than shared substrings — most useful for diverse libraries where similar function is not reflected in similar amino acids.


## Part of the Platforma ecosystem

This block is part of [Platforma](https://platforma.bio/) by [MiLaboratories](https://github.com/milaboratory), built on [UMAP](https://github.com/lmcinnes/umap). Explore the other open-source blocks at [github.com/platforma-open](https://github.com/platforma-open) and the docs for antibody discovery at [docs.platforma.bio/biology-guides/antibody-discovery](https://docs.platforma.bio/biology-guides/antibody-discovery/).
