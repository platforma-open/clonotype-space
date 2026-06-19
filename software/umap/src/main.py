#!/usr/bin/env python3
"""
kmer_umap.py: Compute UMAP embeddings of biological sequences via k-mer count vectors and
truncated SVD.

Pipeline:
    sequences → k-mer counts (sparse) → TruncatedSVD → UMAP coordinates

Usage:
    python main.py \
        -i input.tsv -c sequence \
        -u umap.tsv \
        [--alphabet aminoacid|nucleotide] \
        [--umap-components 2] \
        [--umap-neighbors 15] [--umap-min-dist 0.5] \
        [--k-mer-size 3] \
        [--max-sequences 200000] \
        [--svd-backend auto|cuml|sklearn] \
        [--umap-backend auto|cuml|sklearn] \
        [--output-dir .]

Inputs:
    A TSV file with one or more sequence columns (column names start with the prefix given by
    `-c`/`--seq-col-start`, default `sequence`). Selected columns are concatenated row-wise
    into a single sequence.

Outputs:
    A TSV file (`-u`/`--umap-output`) with columns `clonotypeKey`, `UMAP1`, `UMAP2`, ...
    All input rows are present in the output; rows whose sequences contain invalid alphabet
    characters get null coordinates.

    A `skipped_clonotypes_summary.txt` listing skipped (invalid) sequences.

Two-phase pipeline (CPU path, used when no GPU is available):
    Phase 1 — fit:
        - Take a random sample of size --max-sequences (or all rows if smaller).
        - Build the k-mer count matrix on the sample.
        - Fit TruncatedSVD; pick the smallest k explaining 95% variance (cap 500).
        - Sub-sample further to UMAP_FIT_MAX_SAMPLE_SIZE (100k) rows for UMAP fit.
    Phase 2 — transform all rows:
        - Process all valid sequences in chunks of TRANSFORM_CHUNK_SIZE.
        - Each chunk: k-mer count → svd.transform() → umap.transform().

GPU path (when cuML is available and the matrix fits in GPU memory):
    Single fit_transform pass on all sequences. No sub-sampling.

Determinism note:
    Output is deterministic for a given input file order and software stack. The CPU and GPU
    paths use different SVD algorithms (sklearn randomized vs. cupy svds), so the same input
    can yield different embeddings across machines. The auto backend therefore couples output
    to hardware — pin a backend if reproducibility across runs/machines matters.
"""

import warnings

# Configure warnings BEFORE other imports so they take effect during module init.
warnings.filterwarnings("ignore", message="Spectral initialisation failed!")
warnings.filterwarnings("ignore", message="Falling back to random initialisation!")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to .* by setting random_state")

import argparse
import functools
import itertools
import os
import resource
import sys
import time

import numpy as np
import polars as pl
from scipy import sparse


# ============================================================================
# Constants
# ============================================================================

# Random seed for all stochastic operations (sampling, SVD, UMAP).
RANDOM_STATE = 42

# SVD / PCA configuration. The 95%-variance target and 500-component cap are shared by the k-mer SVD
# path and the embedding centered-PCA path.
SVD_TARGET_VARIANCE = 0.95
SVD_MAX_COMPONENTS = 500

# Embedding mode: a vector whose centered-PCA residual norm falls below this is "degenerate" — it sits
# at (≈) the dataset mean, so it has no reliable direction to normalize. Excluded from the UMAP fit and
# given null coordinates.
DEGENERATE_NORM_THRESHOLD = 1e-8

# Maximum sequences used to fit UMAP on the CPU path. Larger fit-samples are sub-sampled
# down to this size before UMAP fitting.
UMAP_FIT_MAX_SAMPLE_SIZE = 100000

# Chunk size for Phase 2 transform (k-mer → svd.transform → umap.transform).
# Chosen to keep per-chunk memory bounded while minimising Python loop overhead.
TRANSFORM_CHUNK_SIZE = 50000

# Threshold below which k-mer counting runs single-threaded; the multiprocessing overhead
# is not worth it for small inputs.
KMER_PARALLEL_THRESHOLD = 5000

# Output column names. KEY_COL is provided by the block workflow; standalone runs synthesize
# it from a row index.
KEY_COL = 'clonotypeKey'
SEQ_COL = 'combined_sequence'

# GPU memory budgeting for sparse SVD.
SPARSE_MEMORY_MULTIPLIER = 3.0
MEMORY_BUFFER_GB = 2.0

# Alphabet character lists — must stay in sync with the regex in
# load_and_filter_input() and with the alphabet used by kmer_count_vectors().
_AMINOACID_CHARS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', '*', '_']
_NUCLEOTIDE_CHARS = ['A', 'C', 'G', 'T', 'N']


# ============================================================================
# Live-flushed print
# ============================================================================

# Force flush on every print so workflow logs stream live.
print = functools.partial(print, flush=True)


# ============================================================================
# SVD components wrapper
# ============================================================================

class _SVDTransformer:
    """
    Lightweight wrapper around an SVD V matrix.

    Both CPU and GPU paths produce a `(n_components, n_features)` components matrix;
    transforming new data is just `X @ components.T`. This class provides that uniformly
    without depending on the underlying sklearn or cuml model object.
    """

    def __init__(self, components):
        # components shape: (n_components, n_features)
        self.components_ = components

    def transform(self, X):
        return X @ self.components_.T


# ============================================================================
# K-mer counting
# ============================================================================

def _process_sequence_chunk(args):
    """
    Worker function for parallel k-mer counting.

    Args:
        args: Tuple of (sequences_chunk, start_idx, k, kmer_to_index)

    Returns:
        Tuple of (rows, cols) numpy int32 arrays for COO matrix construction.
    """
    sequences_chunk, start_idx, k, kmer_to_index = args

    rows = []
    cols = []
    for local_idx, seq in enumerate(sequences_chunk):
        global_idx = start_idx + local_idx
        seq_upper = str(seq).upper()
        for pos in range(len(seq_upper) - k + 1):
            kmer = seq_upper[pos:pos + k]
            kmer_idx = kmer_to_index.get(kmer)
            if kmer_idx is not None:
                rows.append(global_idx)
                cols.append(kmer_idx)

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


def kmer_count_vectors(sequences, k=3, alphabet='aminoacid', n_jobs=-1, verbose=True):
    """
    Convert sequences to a k-mer count matrix using parallel processing.

    Args:
        sequences: List of sequences (already uppercase).
        k: Size of k-mers to count.
        alphabet: 'aminoacid' or 'nucleotide'.
        n_jobs: Workers; -1 = all CPUs, 1 = single-threaded, N = N workers.
        verbose: Print progress messages.

    Returns:
        scipy.sparse.csr_matrix: sparse k-mer count matrix.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    if verbose:
        print(f"Generating {k}-mer count vectors for {alphabet} sequences...")

    if alphabet == 'aminoacid':
        alphabet_chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', '*', '_']
    else:
        alphabet_chars = ['A', 'C', 'G', 'T', 'N']

    all_kmers = [''.join(p) for p in itertools.product(alphabet_chars, repeat=k)]
    kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}

    num_seqs = len(sequences)
    num_kmers = len(all_kmers)

    if n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    if num_seqs < KMER_PARALLEL_THRESHOLD:
        n_jobs = 1

    if n_jobs == 1:
        if verbose:
            print(f"Processing {num_seqs} sequences (single-threaded)...")
        rows_arr, cols_arr = _process_sequence_chunk((sequences, 0, k, kmer_to_index))
    else:
        # 4 chunks per worker for better load balancing.
        chunk_size = max(1000, num_seqs // (n_jobs * 4))
        chunks = []
        for i in range(0, num_seqs, chunk_size):
            chunk_end = min(i + chunk_size, num_seqs)
            chunks.append((sequences[i:chunk_end], i, k, kmer_to_index))

        if verbose:
            print(f"Processing {num_seqs} sequences using {n_jobs} parallel workers "
                  f"({len(chunks)} chunks)...")

        # Spawn context avoids fork issues on macOS and with GPU libraries.
        mp_context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_context) as executor:
            results = list(executor.map(_process_sequence_chunk, chunks))

        if verbose:
            print("Merging results from parallel workers...")
        rows_parts = [r for r, _ in results]
        cols_parts = [c for _, c in results]
        rows_arr = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.int32)
        cols_arr = np.concatenate(cols_parts) if cols_parts else np.array([], dtype=np.int32)

    # tocsr() sums duplicate (row, col) entries natively in C — no Python Counter needed.
    if verbose:
        print(f"Building sparse matrix from {len(rows_arr)} k-mer occurrences...")
    data_arr = np.ones(len(rows_arr), dtype=np.int32)
    matrix = sparse.coo_matrix(
        (data_arr, (rows_arr, cols_arr)),
        shape=(num_seqs, num_kmers),
        dtype=np.int32,
    ).tocsr()

    if verbose:
        print(f"Sparse matrix created: {matrix.shape}, {matrix.nnz} non-zero entries")
    return matrix


# ============================================================================
# Position-tagged k-mer encoding
# ============================================================================

def _encode_position_tagged_fixed_length(sequences, seq_len, k, chars):
    """
    Inner encoder. Assumes every sequence is exactly `seq_len` chars (no
    padding done here). Builds the (n_seqs, (seq_len - k + 1) * |chars|^k)
    CSR matrix via the vectorized ASCII-lookup + direct-CSR path.
    """
    num_chars = len(chars)
    n_seqs = len(sequences)
    num_positions = seq_len - k + 1
    num_kmers = num_chars ** k
    num_features = num_positions * num_kmers

    # ASCII byte → alphabet-index lookup table. ord('A') maps to 0, ord('C')
    # to 1, etc. Unknown chars stay at -1 so they trip the check below.
    ascii_to_idx = np.full(256, -1, dtype=np.int32)
    for i, c in enumerate(chars):
        ascii_to_idx[ord(c)] = i

    # Vectorization trick: concatenate all sequences into one big ASCII blob
    # (C-implemented), then view as a (n_seqs, seq_len) uint8 array zero-copy.
    # reshape() guarantees uniform length
    arr = np.frombuffer(''.join(sequences).encode('ascii'),
                        dtype=np.uint8).reshape(n_seqs, seq_len)
    char_idx = ascii_to_idx[arr]  # broadcast lookup, shape (n_seqs, seq_len)
    if not (char_idx >= 0).all():
        raise ValueError("position_tagged_kmer_vectors received an unknown "
                         "character — load_and_filter_input should have dropped it.")

    # Encode each window of k characters as a base-`num_chars` integer:
    #   kmer_index = sum_{i in [0, k)} char_idx[:, p+i] * num_chars^(k-1-i)
    # This gives a unique kmer_index per distinct k-mer content.
    powers = num_chars ** np.arange(k - 1, -1, -1, dtype=np.int64)
    kmer_at_position = np.zeros((n_seqs, num_positions), dtype=np.int64)
    for p in range(num_positions):
        kmer_at_position[:, p] = (char_idx[:, p:p + k] * powers).sum(axis=1)

    # Final column index = position * num_kmers + kmer_index. Position offsets
    # are added per row in one broadcast. Every row has exactly num_positions
    # non-zeros in increasing column order, so we can build the CSR directly
    # with a uniform-stride indptr (no COO → CSR conversion needed).
    col_offsets = (np.arange(num_positions, dtype=np.int64) * num_kmers).reshape(1, -1)
    col_indices = (col_offsets + kmer_at_position).ravel().astype(np.int32)
    indptr = np.arange(0, n_seqs * num_positions + 1, num_positions, dtype=np.int32)
    data = np.ones(n_seqs * num_positions, dtype=np.int8)
    return sparse.csr_matrix(
        (data, col_indices, indptr),
        shape=(n_seqs, num_features),
        dtype=np.int8,
    )


def position_tagged_kmer_vectors(sequences, k=2, alphabet='aminoacid', verbose=True,
                                 max_len=None):
    """
    Position-tagged k-mer encoding (N-terminus aligned).

    Each k-mer is identified by both its content AND its starting position in
    the sequence: (k-mer 'WT' at position 0) and (k-mer 'WT' at position 3)
    are distinct features. This combines positional encoding's exactness with
    k-mer composition's shift-robustness — a 1-aa substitution only flips the
    k-mers that overlap that position, so similar peptides stay nearby in
    feature space.

    Variable lengths are handled by **right-padding** (N-term alignment):
    shorter sequences are padded with a filler character ('_' for aminoacid,
    'N' for nucleotide) at the C-terminus so all sequences reach max length.
    Real residues stay at positions 0..L-1; padded positions sit at the tail.

    `max_len` pins the encoding width across calls. Required when feeding
    chunks into a pre-fit SVD/UMAP transformer. If omitted, max_len is taken 
    from the input.

    Feature layout: feature_index = position * |chars|^k + kmer_index.
        - Total features: (L_max - k + 1) * |chars|^k
        - Each sequence: (L_max - k + 1) non-zero entries
    """
    chars = _AMINOACID_CHARS if alphabet == 'aminoacid' else _NUCLEOTIDE_CHARS
    pad_char = '_' if alphabet == 'aminoacid' else 'N'

    n_seqs = len(sequences)
    if max_len is None:
        max_len = max(map(len, sequences))
    min_len = min(map(len, sequences))
    is_variable = min_len != max_len
    if max_len < k:
        raise ValueError(
            f"Position_tagged_kmer_vectors needs sequences of length >= k={k}, "
            f"but the longest input sequence has length {max_len}."
        )
    num_positions = max_len - k + 1
    num_kmers = len(chars) ** k

    if verbose:
        print(f"Position-tagged {k}-mer encoding: {n_seqs} sequences × "
              f"{num_positions} positions × {num_kmers} k-mers = "
              f"{num_positions * num_kmers} features...")
        if is_variable:
            print(f"  Input length range: {min_len} to {max_len}. "
                  f"Right-padding shorter sequences with '{pad_char}'.")

    # Pad shorter sequences with the filler so reshape works. ljust is a no-op
    # for already-max-length entries — the comprehension only allocates new
    # strings where needed.
    if is_variable:
        sequences = [s if len(s) == max_len else s.ljust(max_len, pad_char)
                     for s in sequences]

    matrix = _encode_position_tagged_fixed_length(sequences, max_len, k, chars)

    if verbose:
        print(f"Position-tagged k-mer matrix created: {matrix.shape}, "
              f"{matrix.nnz} non-zero entries")
    return matrix


# ============================================================================
# Model factories
# ============================================================================

def create_svd_model(backend, n_components, random_state=RANDOM_STATE):
    """Create a TruncatedSVD model. Returns (model, 'gpu'|'cpu')."""
    if backend in ('cuml', 'auto'):
        try:
            import cuml  # noqa: F401
            from cuml.decomposition import TruncatedSVD as cuML_TruncatedSVD
            print("Using GPU-accelerated TruncatedSVD (RAPIDS cuML)...")
            return cuML_TruncatedSVD(n_components=n_components, random_state=random_state), 'gpu'
        except Exception as e:
            if backend == 'cuml':
                print(f"Error: RAPIDS cuML not available or CUDA error: {e}")
                raise
            print(f"RAPIDS cuML not available, falling back to CPU-based TruncatedSVD: {e}")

    from sklearn.decomposition import TruncatedSVD
    print("Using CPU-based TruncatedSVD (scikit-learn)...")
    return TruncatedSVD(n_components=n_components, random_state=random_state), 'cpu'


def create_umap_model(backend, components, neighbors, min_dist):
    """Create a UMAP model. Returns (model, 'gpu'|'cpu')."""
    common_params = {
        'n_components': components,
        'n_neighbors': neighbors,
        'min_dist': min_dist,
        'random_state': RANDOM_STATE,
    }

    if backend in ('cuml', 'auto'):
        try:
            import cuml  # noqa: F401
            import cuml.manifold.umap as cuml_umap
            print("Using GPU-accelerated UMAP (RAPIDS cuML)...\n")
            return cuml_umap.UMAP(**common_params, init='spectral', n_epochs=2000), 'gpu'
        except Exception as e:
            print(f"RAPIDS cuML not available or CUDA error: {e}")

    if backend == 'parametric-umap':
        from umap.parametric_umap import ParametricUMAP
        return ParametricUMAP(n_components=common_params['n_components']), 'gpu'

    import umap as umap_learn
    print("Using CPU-based UMAP (umap-learn)...\n")
    return umap_learn.UMAP(n_jobs=-1, **common_params), 'cpu'


# ============================================================================
# SVD pipeline
# ============================================================================

def estimate_sparse_memory_gb(matrix):
    """Estimate GPU memory required for a CSR sparse matrix in GB."""
    nnz = matrix.nnz
    # data (float32) + indices (int32) + indptr (int32)
    sparse_memory_bytes = (nnz * 4 + nnz * 4 + (matrix.shape[0] + 1) * 4)
    return sparse_memory_bytes / (1024 ** 3)


def estimate_dense_memory_gb(matrix):
    """Estimate memory of a dense float32 representation in GB."""
    return (matrix.shape[0] * matrix.shape[1] * 4) / (1024 ** 3)


def get_gpu_memory_info():
    """Get GPU memory info as (free_gb, total_gb)."""
    import cupy as cp
    free, total = cp.cuda.Device().mem_info
    return free / (1024 ** 3), total / (1024 ** 3)


def compute_explained_variance_cupy(singular_values, matrix_gpu, n_samples):
    """Compute explained variance ratio from CuPy SVD singular values."""
    import cupy as cp
    explained_variance = (singular_values ** 2) / (n_samples - 1)
    total_sum_squares = float(matrix_gpu.power(2).sum())
    total_variance = total_sum_squares / (n_samples - 1)
    if total_variance == 0:
        return cp.zeros_like(explained_variance)
    return cp.asnumpy(explained_variance / total_variance)


def run_cupy_sparse_svd(matrix_gpu, n_components, random_seed=RANDOM_STATE):
    """Run CuPy sparse SVD; return (u, s, vt) in descending-singular-value order."""
    import cupy as cp
    from cupyx.scipy.sparse.linalg import svds as cupy_svds
    cp.random.seed(random_seed)
    u, s, vt = cupy_svds(matrix_gpu, k=n_components)
    return u[:, ::-1], s[::-1], vt[::-1, :]


def fallback_to_cpu_svd(matrix, n_components):
    """Fit a CPU-based TruncatedSVD. Returns (svd_model, explained_variance_ratio)."""
    print("Falling back to CPU-based SVD...")
    svd_model, _ = create_svd_model('sklearn', n_components, random_state=RANDOM_STATE)
    svd_model.fit(matrix)
    return svd_model, svd_model.explained_variance_ratio_


def compute_svd_embedding(matrix, svd_backend='auto',
                          target_variance=SVD_TARGET_VARIANCE,
                          max_components=SVD_MAX_COMPONENTS):
    """
    Fit SVD on `matrix` and return embedding plus a transformer for new data.

    Pipeline:
        1. Pick backend (GPU via cupy if cuML imports + GPU memory sufficient, else CPU sklearn).
        2. Fit SVD with min(matrix.shape[0]-1, matrix.shape[1], max_components) components.
        3. Determine optimal k: smallest count whose cumulative variance >= target_variance.
           If target is never reached, use the full component count.
        4. Slice the fitted model to k components (no re-fit) and produce the embedding.

    Fallbacks (logged but silent — caller does not see which path was taken):
        - GPU import error / out-of-memory → CPU SVD.
        - GPU memory insufficient for the matrix → CPU SVD.
        - GPU SVD post-fit transform error → fresh sklearn fit at k components.

    Returns:
        (svd_embed, svd_transformer, n_components, explained_var_sum)
    """
    n_components_max = min(matrix.shape[0] - 1, matrix.shape[1], max_components)
    print(f"Computing SVD with up to {n_components_max} components...")

    matrix_gpu = None
    use_cupy_sparse_svd = False
    svd_u = svd_s = svd_vt = None
    explained_variance_ratio = None
    svd_cpu_model = None

    try:
        _, initial_backend = create_svd_model(svd_backend, n_components_max, random_state=RANDOM_STATE)
    except Exception as e:
        print(f"Warning: Error detecting SVD backend: {e}")
        initial_backend = 'cpu'

    if initial_backend == 'gpu':
        try:
            import cupy as cp
            from cupyx.scipy import sparse as cp_sparse

            free_gb, total_gb = get_gpu_memory_info()
            print(f"GPU memory available: {free_gb:.2f} GB / {total_gb:.2f} GB")

            sparse_mem_gb = estimate_sparse_memory_gb(matrix)
            dense_mem_gb = estimate_dense_memory_gb(matrix)
            sparsity_pct = (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
            print(f"Sparse matrix memory: {sparse_mem_gb:.2f} GB "
                  f"(vs {dense_mem_gb:.2f} GB if dense)")
            print(f"Matrix sparsity: {sparsity_pct:.1f}% sparse")

            required_mem_gb = sparse_mem_gb * SPARSE_MEMORY_MULTIPLIER + MEMORY_BUFFER_GB
            if free_gb >= required_mem_gb:
                print("Using CuPy sparse SVD (supports sparse matrices on GPU)...")
                matrix_gpu = cp_sparse.csr_matrix(matrix, dtype=cp.float32)
                print(f"GPU sparse matrix created: {matrix_gpu.shape}, {matrix_gpu.nnz} non-zeros")
                print("Running GPU sparse SVD...")
                svd_u, svd_s, svd_vt = run_cupy_sparse_svd(matrix_gpu, n_components_max)
                explained_variance_ratio = compute_explained_variance_cupy(
                    svd_s, matrix_gpu, matrix.shape[0])
                # Sanity check: cupy_svds silently returns near-zero singular
                # values when k is too close to min(m, n) (Lanczos loses
                # orthogonality).
                total_explained = float(np.sum(explained_variance_ratio))
                if total_explained < 0.01:
                    raise RuntimeError(
                        f"CuPy sparse SVD returned degenerate output "
                        f"(total variance = {total_explained:.4f}). "
                    )
                use_cupy_sparse_svd = True
                print("GPU sparse SVD completed successfully.")
            else:
                print("Warning: Insufficient GPU memory for sparse operations.")
                print(f"Required: ~{required_mem_gb:.2f} GB, Available: {free_gb:.2f} GB")
                svd_cpu_model, explained_variance_ratio = fallback_to_cpu_svd(matrix, n_components_max)

        except (ImportError, MemoryError) as e:
            error_type = "ImportError" if isinstance(e, ImportError) else "Out of Memory"
            print(f"Warning: {error_type} during GPU SVD - {e}")
            svd_cpu_model, explained_variance_ratio = fallback_to_cpu_svd(matrix, n_components_max)
            use_cupy_sparse_svd = False
        except Exception as e:
            print(f"Warning: Unexpected error during GPU SVD - {e}")
            svd_cpu_model, explained_variance_ratio = fallback_to_cpu_svd(matrix, n_components_max)
            use_cupy_sparse_svd = False
    else:
        svd_cpu_model, explained_variance_ratio = fallback_to_cpu_svd(matrix, n_components_max)

    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    if np.any(cumulative_explained_variance >= target_variance):
        n_components_optimal = int(np.argmax(cumulative_explained_variance >= target_variance)) + 1
    else:
        n_components_optimal = n_components_max
    print(f"Number of components explaining {target_variance * 100:.0f}% variance: "
          f"{n_components_optimal}")

    if use_cupy_sparse_svd and matrix_gpu is not None:
        try:
            import cupy as cp
            # Slice the already-computed SVD down to n_components_optimal instead
            # of recomputing. The top-k truncation of a rank-m SVD is the first k
            # components, so this avoids a second SVD pass on the GPU. Mirrors
            # the CPU path's behavior.
            svd_u = svd_u[:, :n_components_optimal]
            svd_s = svd_s[:n_components_optimal]
            svd_vt = svd_vt[:n_components_optimal, :]
            print(f"Using top {n_components_optimal} components from GPU sparse SVD.")
            svd_embed = cp.asnumpy(svd_u @ cp.diag(svd_s))
            explained_variance_ratio_final = compute_explained_variance_cupy(
                svd_s, matrix_gpu, matrix.shape[0])
            explained_var_sum = float(np.sum(explained_variance_ratio_final))
            svd_transformer = _SVDTransformer(cp.asnumpy(svd_vt))
            print("GPU sparse SVD embedding computed successfully.")
        except Exception as e:
            print(f"Warning: GPU SVD transform failed: {e}")
            svd_model, _ = create_svd_model('sklearn', n_components_optimal, random_state=RANDOM_STATE)
            svd_embed = svd_model.fit_transform(matrix)
            explained_var_sum = float(svd_model.explained_variance_ratio_.sum())
            svd_transformer = _SVDTransformer(svd_model.components_)
    else:
        # svd_cpu_model was already fit with n_components_max components in
        # fallback_to_cpu_svd. The rank-k truncation of a rank-m SVD (m >= k) is the first
        # k components, so slice rather than refit.
        components = svd_cpu_model.components_[:n_components_optimal]
        svd_transformer = _SVDTransformer(components)
        svd_embed = svd_transformer.transform(matrix)
        explained_var_sum = float(
            svd_cpu_model.explained_variance_ratio_[:n_components_optimal].sum()
        )

    print(f"Explained variance by {n_components_optimal} components: {explained_var_sum:.3f}")
    return svd_embed, svd_transformer, n_components_optimal, explained_var_sum


# ============================================================================
# Output helpers
# ============================================================================

def create_empty_umap_output(key_col_name, umap_components, output_path):
    """Create an empty UMAP output file with proper headers."""
    schema = {key_col_name: pl.String}
    for i in range(umap_components):
        schema[f'UMAP{i + 1}'] = pl.Float64
    pl.DataFrame(schema=schema).write_csv(output_path, separator='\t')


def create_empty_skipped_summary(output_dir, reason):
    """Create an empty skipped-clonotypes summary file with a reason string."""
    skipped_summary_path = os.path.join(output_dir, 'skipped_clonotypes_summary.txt')
    with open(skipped_summary_path, 'w') as f:
        f.write(f"{reason}\n")


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    """Build the argparse parser and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute UMAP embeddings from sequences via k-mer counts and SVD.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('-i', '--input', required=False, default=None,
                        help='Input TSV file with sequence column(s). Required for kmer/pos-kmer '
                             'encodings; ignored for --encoding embedding (use --matrix).')
    parser.add_argument('-c', '--seq-col-start', default='sequence',
                        help='Prefix of input columns to treat as sequences (default: "sequence").')
    parser.add_argument('-u', '--umap-output', required=True,
                        help='Output TSV file for UMAP embeddings.')
    parser.add_argument('--alphabet', choices=['aminoacid', 'nucleotide'], default='aminoacid',
                        help='Sequence alphabet type (default: aminoacid).')
    parser.add_argument('--umap-components', type=int, default=2,
                        help='Number of UMAP dimensions (default: 2).')
    parser.add_argument('--umap-neighbors', type=int, default=15,
                        help='UMAP n_neighbors (default: 15).')
    parser.add_argument('--umap-min-dist', type=float, default=0.5,
                        help='UMAP min_dist (default: 0.5).')
    parser.add_argument('--k-mer-size', type=int, default=None,
                        help='Size of k-mers (default: 3 for aminoacid, 6 for nucleotide).')
    parser.add_argument('--encoding', choices=['kmer', 'pos-kmer', 'embedding'], default='kmer',
                        help='Feature encoding (default: kmer).\n'
                             '  kmer:      k-mer count vectors (position-agnostic, any length).\n'
                             '  pos-kmer:  position-tagged k-mers — each (position, k-mer) pair\n'
                             '             is a distinct feature. Variable-length sequences are\n'
                             '             right-padded (N-term aligned) with "_" / "N" up to the\n'
                             '             longest input. Use --k-mer-size 2 for short peptides.\n'
                             '  embedding: run UMAP on learned per-clonotype embedding vectors read\n'
                             '             from --matrix (parquet); skips k-mer counting and SVD,\n'
                             '             reduces with centered PCA-95%% then L2-normalizes and runs\n'
                             '             Euclidean UMAP (≡ cosine ranking).')
    # --- Embedding mode (--encoding embedding) ---
    parser.add_argument('--matrix', default=None,
                        help='Long-format embedding matrix (parquet) for --encoding embedding: one row '
                             'per (clonotype, embeddingDim) with the embedding value.')
    parser.add_argument('--key-col', default='clonotypeKey',
                        help='Clonotype-key column name in the embedding matrix (default: clonotypeKey).')
    parser.add_argument('--dim-col', default='embeddingDim',
                        help='Embedding-dimension column name in the embedding matrix (default: embeddingDim).')
    parser.add_argument('--value-col', default='value',
                        help='Embedding-value column name in the embedding matrix (default: value).')
    parser.add_argument('--embedding-model', default=None,
                        help='Embedding model identifier, logged to the processing log for provenance '
                             '(--encoding embedding). Stamped on the output column domain by the workflow.')
    parser.add_argument('--output-dir', default='.',
                        help='Directory for output files (default: current directory).')
    parser.add_argument('--svd-backend', type=str, default='auto',
                        choices=['auto', 'cuml', 'sklearn'],
                        help='SVD backend (default: auto).\n'
                             '  auto:    cuML if available, else scikit-learn.\n'
                             '  cuml:    force RAPIDS cuML (requires CUDA GPU).\n'
                             '  sklearn: force scikit-learn (CPU).')
    parser.add_argument('--umap-backend', type=str, default='auto',
                        choices=['auto', 'cuml', 'sklearn', 'parametric-umap'],
                        help='UMAP backend (default: auto).\n'
                             '  auto:    cuML if available, else umap-learn.\n'
                             '  cuml:    force RAPIDS cuML (requires CUDA GPU).\n'
                             '  sklearn: force umap-learn (CPU).')
    parser.add_argument('--store-models', action='store_true',
                        help='Save fitted models to --output-dir (parametric-umap only).')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel workers for k-mer counting (-1 = all CPUs, default: -1).')
    parser.add_argument('--max-sequences', type=int, default=200000,
                        help='Fit-sample size; SVD/UMAP fitted on this many randomly selected\n'
                             'sequences and ALL valid sequences transformed through fitted models.\n'
                             '0 = fit on all sequences (default: 200000).')
    return parser.parse_args()


def validate_args(args):
    """Validate parsed arguments. Exits with status 1 on invalid values."""
    if args.umap_components < 1:
        print("Error: Number of UMAP components must be at least 1")
        sys.exit(1)
    if args.umap_neighbors < 1:
        print("Error: UMAP neighbors must be at least 1")
        sys.exit(1)
    if not (0 <= args.umap_min_dist <= 1):
        print("Error: UMAP min_dist must be between 0 and 1")
        sys.exit(1)
    if args.k_mer_size is not None and args.k_mer_size < 1:
        print("Error: k-mer size must be at least 1")
        sys.exit(1)


# ============================================================================
# Input loading
# ============================================================================

def load_and_filter_input(args, output_path):
    """
    Load input TSV, build the combined sequence column, drop empty sequences,
    and split rows by alphabet validity.

    Exits with status 1 on unrecoverable issues (file missing, parse error, no valid
    sequences after filtering).

    Exits with status 0 (writing empty output files) for empty inputs.

    Returns:
        (df, df_valid, df_invalid, n_invalid)
    """
    try:
        print("Loading input file...")
        df = pl.read_csv(args.input, separator='\t', infer_schema=False, null_values=[''])
        print(f"Loaded {len(df)} sequences")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    if df.is_empty():
        print("Warning: Input file is empty — writing empty output.")
        first_col = df.columns[0] if df.columns else KEY_COL
        create_empty_umap_output(first_col, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir, "UMAP analysis skipped due to empty input file.")
        sys.exit(0)

    seq_col_list = sorted(c for c in df.columns if c.startswith(args.seq_col_start))
    if not seq_col_list:
        print(f"Error: Columns starting with '{args.seq_col_start}' not found. "
              f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    # Block workflow always provides 'clonotypeKey'; for standalone use we add a row index.
    if KEY_COL not in df.columns:
        df = df.with_row_index(KEY_COL).with_columns(pl.col(KEY_COL).cast(pl.String))

    df = df.with_columns(
        pl.concat_str([pl.col(c).fill_null('') for c in seq_col_list]).alias(SEQ_COL)
    )

    initial_count = len(df)
    df = df.filter(pl.col(SEQ_COL).str.strip_chars('_').str.len_chars() > 0)
    if len(df) < initial_count:
        print(f"Warning: Removed {initial_count - len(df)} empty or whitespace-only sequences.")

    if df.is_empty():
        print("Warning: No non-empty sequences after filtering — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir, "UMAP analysis skipped due to empty input file.")
        sys.exit(0)

    # Vectorized alphabet validation (no Python loop).
    if args.alphabet == 'aminoacid':
        valid_pattern = r'^[ACDEFGHIKLMNPQRSTVWY*X_]+$'
    else:
        valid_pattern = r'^[ACGTN_]+$'
    valid_expr = pl.col(SEQ_COL).str.to_uppercase().str.contains(valid_pattern)
    df_valid = df.filter(valid_expr)
    df_invalid = df.filter(valid_expr.not_())
    n_invalid = len(df_invalid)

    if n_invalid > 0:
        seq_type_name = "amino acid" if args.alphabet == 'aminoacid' else "nucleotide"
        print(f"Warning: Found {n_invalid} sequences with invalid {seq_type_name} characters. "
              f"These sequences will be skipped.")
        examples = df_invalid[SEQ_COL].head(5).to_list()
        print(f"Example invalid sequences (first 5): {examples}")

    if len(df_valid) == 0:
        print('Error: No valid sequences found after filtering. Exiting.')
        sys.exit(1)

    return df, df_valid, df_invalid, n_invalid


# ============================================================================
# Pipelines
# ============================================================================

def run_gpu_pipeline(args, sequences_all, umap_model):
    """
    Run the GPU-style pipeline: k-mer → SVD → UMAP fit_transform on all sequences.

    If GPU UMAP fails after SVD, falls back to CPU UMAP (fit on a sub-sample, then transform
    all). Note this fallback produces different output than a full CPU run (different SVD too).

    Returns:
        umap_embed_all (numpy array)
    """
    n_all = len(sequences_all)

    start_time_kmer = time.time()
    if args.encoding == 'pos-kmer':
        # Pin pos-kmer width to the global max so feature counts stay consistent.
        pos_kmer_max_len = max(map(len, sequences_all))
        matrix = position_tagged_kmer_vectors(sequences_all, k=args.k_mer_size,
                                              alphabet=args.alphabet, verbose=True,
                                              max_len=pos_kmer_max_len)
    else:
        matrix = kmer_count_vectors(sequences_all, k=args.k_mer_size, alphabet=args.alphabet,
                                    n_jobs=args.n_jobs, verbose=True)
    print(f"Encoding completed in {time.time() - start_time_kmer:.2f} seconds.\n")

    start_time_svd = time.time()
    print("Running Truncated SVD...")
    svd_embed, _, n_components_used, explained_var_sum = compute_svd_embedding(
        matrix=matrix,
        svd_backend=args.svd_backend,
        target_variance=SVD_TARGET_VARIANCE,
        max_components=SVD_MAX_COMPONENTS,
    )
    print(f"Truncated SVD completed in {time.time() - start_time_svd:.2f} seconds "
          f"({n_components_used} components, {explained_var_sum:.3f} variance).\n")

    start_time_umap = time.time()
    print("Running UMAP dimensionality reduction (fitting and transforming on GPU)...")
    try:
        umap_embed_all = umap_model.fit_transform(svd_embed)
        print(f"UMAP completed in {time.time() - start_time_umap:.2f} seconds.\n")
        return umap_embed_all
    except Exception as e:
        print(f"Warning: GPU UMAP failed - {e}. Falling back to CPU UMAP...")

    # CPU UMAP fallback path.
    cpu_umap_model, _ = create_umap_model(
        'sklearn', args.umap_components, args.umap_neighbors, args.umap_min_dist)
    np.random.seed(RANDOM_STATE)
    if n_all > UMAP_FIT_MAX_SAMPLE_SIZE:
        sample_indices = np.random.choice(n_all, size=UMAP_FIT_MAX_SAMPLE_SIZE, replace=False)
        cpu_umap_model.fit(svd_embed[sample_indices])
    else:
        cpu_umap_model.fit(svd_embed)
    umap_embed_all = cpu_umap_model.transform(svd_embed)
    print(f"UMAP (CPU fallback) completed in {time.time() - start_time_umap:.2f} seconds.\n")
    return umap_embed_all


def run_cpu_pipeline(args, df_valid, sequences_all, umap_model):
    """
    Run the two-phase CPU pipeline:
      Phase 1: build k-mer matrix on a fit-sample, fit SVD, fit UMAP.
      Phase 2: transform ALL valid sequences in chunks through the fitted models.

    Returns:
        umap_embed_all (numpy array)
    """
    n_all = len(sequences_all)
    num_total_sequences = len(df_valid)

    # Pin pos-kmer width to the global max so fit and chunk-transform calls
    # produce matrices with the same column count.
    pos_kmer_max_len = max(map(len, sequences_all)) if args.encoding == 'pos-kmer' else None

    # --- Build fit-sample ---
    fit_sample_size = args.max_sequences
    if fit_sample_size > 0 and num_total_sequences > fit_sample_size:
        print(f"Fit-sample: {num_total_sequences} valid sequences exceed "
              f"--max-sequences={fit_sample_size}.")
        print(f"Randomly sampling {fit_sample_size} sequences to fit SVD and UMAP models...")
        df_fit = df_valid.sample(n=fit_sample_size, seed=RANDOM_STATE)
        print(f"Fit sample: {len(df_fit)} sequences selected.\n")
    else:
        df_fit = df_valid

    sequences_fit = df_fit[SEQ_COL].str.to_uppercase().to_list()
    num_fit_sequences = len(sequences_fit)

    # --- Phase 1a: encode fit sample ---
    start_time_kmer = time.time()
    if args.encoding == 'pos-kmer':
        matrix_fit = position_tagged_kmer_vectors(sequences_fit, k=args.k_mer_size,
                                                  alphabet=args.alphabet, verbose=True,
                                                  max_len=pos_kmer_max_len)
    else:
        matrix_fit = kmer_count_vectors(sequences_fit, k=args.k_mer_size, alphabet=args.alphabet,
                                        n_jobs=args.n_jobs, verbose=True)
    print(f"Encoding completed in {time.time() - start_time_kmer:.2f} seconds.\n")

    # --- Phase 1b: fit SVD ---
    start_time_svd = time.time()
    print("Running Truncated SVD...")
    svd_embed_fit, svd_transformer, n_components_used, explained_var_sum = compute_svd_embedding(
        matrix=matrix_fit,
        svd_backend=args.svd_backend,
        target_variance=SVD_TARGET_VARIANCE,
        max_components=SVD_MAX_COMPONENTS,
    )
    print(f"Truncated SVD completed in {time.time() - start_time_svd:.2f} seconds "
          f"({n_components_used} components, {explained_var_sum:.3f} variance).\n")

    # --- Phase 1c: fit UMAP (sub-sample if fit-sample is large) ---
    if num_fit_sequences <= UMAP_FIT_MAX_SAMPLE_SIZE:
        print(f"Fit sample ({num_fit_sequences}) <= {UMAP_FIT_MAX_SAMPLE_SIZE}. "
              f"No sub-sampling — UMAP fits on all fit-sample sequences.")
        sampled_data_for_fit = svd_embed_fit
    else:
        print(f"Fit sample ({num_fit_sequences}) > {UMAP_FIT_MAX_SAMPLE_SIZE}. "
              f"Sampling {UMAP_FIT_MAX_SAMPLE_SIZE} sequences for UMAP fitting.")
        np.random.seed(RANDOM_STATE)
        sample_indices = np.random.choice(num_fit_sequences, size=UMAP_FIT_MAX_SAMPLE_SIZE,
                                          replace=False)
        sampled_data_for_fit = svd_embed_fit[sample_indices]

    start_time_umap_fit = time.time()
    print("Running UMAP dimensionality reduction (fitting model)...")
    umap_model.fit(sampled_data_for_fit)
    print(f"UMAP model fitting completed in {time.time() - start_time_umap_fit:.2f} seconds.\n")

    # --- Phase 2: transform ALL valid sequences in chunks ---
    n_chunks = (n_all + TRANSFORM_CHUNK_SIZE - 1) // TRANSFORM_CHUNK_SIZE
    print(f"Transforming all {n_all} valid sequences in {n_chunks} chunks of "
          f"{TRANSFORM_CHUNK_SIZE}...")
    start_time_transform = time.time()

    all_coords = []
    for chunk_idx, chunk_start in enumerate(range(0, n_all, TRANSFORM_CHUNK_SIZE)):
        chunk_end = min(chunk_start + TRANSFORM_CHUNK_SIZE, n_all)
        chunk_seqs = sequences_all[chunk_start:chunk_end]

        if args.encoding == 'pos-kmer':
            chunk_matrix = position_tagged_kmer_vectors(chunk_seqs, k=args.k_mer_size,
                                                        alphabet=args.alphabet, verbose=False,
                                                        max_len=pos_kmer_max_len)
        else:
            chunk_matrix = kmer_count_vectors(chunk_seqs, k=args.k_mer_size,
                                              alphabet=args.alphabet,
                                              n_jobs=args.n_jobs, verbose=False)
        chunk_svd = svd_transformer.transform(chunk_matrix)
        chunk_umap = umap_model.transform(chunk_svd)
        all_coords.append(chunk_umap)

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            elapsed = time.time() - start_time_transform
            print(f"  Chunk {chunk_idx + 1}/{n_chunks} — {chunk_end}/{n_all} sequences "
                  f"({elapsed:.0f}s elapsed)")

    print(f"Transform completed in {time.time() - start_time_transform:.2f} seconds.\n")
    return np.vstack(all_coords)


# ============================================================================
# Embedding feature path (--encoding embedding)
# ============================================================================
#
# UMAP on learned per-clonotype embedding vectors. Order is load-bearing:
#   load matrix → vector-dedup → centered PCA-95% → L2-normalize → Euclidean UMAP.
# Centered PCA removes ESM-2's large non-discriminative shared mean, so the post-PCA
# L2-normalize + Euclidean (≡ cosine ranking) then measures the mean-removed residuals.
# Reuses create_umap_model (CPU/GPU backends) but NOT run_cpu/gpu_pipeline — the matrix is already
# materialized, so there is no per-chunk feature recompute.

def load_embedding_matrix(path, key_col, dim_col, value_col):
    """Read the long-format embedding matrix (parquet) → (X float32, keys object array).

    The parquet has one row per (clonotype, embeddingDim). Rather than sorting the whole N*D-row frame
    (a full sorted copy — the dominant memory cost at scale), the matrix is built by SCATTER: factorize
    the clonotype key to an integer code, then place each value at X[code, dim]. This is order-
    independent (dim positions the column, code positions the row) and avoids the sort entirely. Casting
    the key to Categorical keeps only the unique keys + per-row codes instead of N*D expanded strings.

    Keys come back in physical-code (first-appearance) order; row order does not matter downstream
    (dedup re-sorts; the output pframe is keyed by clonotype value, not row position). Kept float32.

    Completeness (the ragged-matrix check, R-equivalent): a full, duplicate-free fill requires BOTH
    `len(vals) == n*D` (no extra/duplicate rows) AND no unfilled cell after the scatter (no missing dim).
    embeddingDim is the producer's 0..D-1 index, used directly as the column index; D is inferred as
    max(dim)+1."""
    df = pl.read_parquet(path, columns=[key_col, dim_col, value_col])
    if df.height == 0:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)

    cat = df.get_column(key_col).cast(pl.Categorical)
    keys = cat.cat.get_categories()                 # unique clonotype keys, in physical-code order
    n = keys.len()
    codes = cat.to_physical().to_numpy()            # per-row clonotype index 0..n-1 (aligns with `keys`)
    dims = df.get_column(dim_col).to_numpy()         # per-row embeddingDim index
    vals = df.get_column(value_col).cast(pl.Float32).to_numpy()
    del df, cat

    if dims.min() < 0:
        raise ValueError(f"embeddingDim must be a 0-based index; got a negative value {int(dims.min())}.")
    D = int(dims.max()) + 1

    if vals.shape[0] != n * D:
        raise ValueError(
            f"Ragged embedding matrix: {vals.shape[0]} rows != n*D ({n}*{D}) — a clonotype has extra "
            f"or duplicate (clonotype, dim) rows.")

    X = np.full((n, D), np.nan, dtype=np.float32)
    X[codes, dims] = vals                            # scatter: row = clonotype, col = embeddingDim
    if np.isnan(X).any():
        raise ValueError(
            "Ragged embedding matrix: a (clonotype, dim) cell is missing — a clonotype has a partial "
            "embedding-dimension set.")
    return X, keys.to_numpy().astype(object)


def l2_normalize(X, eps=1e-12):
    """Row-wise L2-normalize, eps-guarded so a zero row can't 0/0. Returns float32.
    Uses np.maximum(norm, eps) (not norm+eps) so valid vectors stay exactly unit-norm — only a
    near-zero row is clamped."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norm, eps)).astype(np.float32)


def _select_k_for_variance(explained_variance_ratio, target=SVD_TARGET_VARIANCE):
    """Smallest component count whose cumulative explained variance reaches `target`; full count if it
    never does."""
    cum = np.cumsum(explained_variance_ratio)
    if np.any(cum >= target):
        return int(np.searchsorted(cum, target) + 1)
    return len(explained_variance_ratio)


def _pca_reduce_sklearn(X_fit, X_all, ncomp):
    """Centered PCA on CPU (sklearn, svd_solver='full', deterministic). Fit on X_fit, transform X_all,
    slice to the 95%-variance k. Returns (Xr float32, k)."""
    from sklearn.decomposition import PCA
    print("Using CPU-based centered PCA (scikit-learn, svd_solver='full')...")
    pca = PCA(n_components=ncomp, svd_solver='full').fit(X_fit)  # centered by default
    k = min(_select_k_for_variance(pca.explained_variance_ratio_), ncomp)
    return pca.transform(X_all)[:, :k].astype(np.float32), k


def _pca_reduce_cuml(X_fit, X_all, ncomp):
    """Centered PCA on GPU (cuML, svd_solver='full'). Net-new vs the k-mer path's sparse SVD; wrapped by
    an OOM/error guard + CPU fallback in embedding_pca_reduce. Returns (Xr float32, k)."""
    import cuml  # noqa: F401
    from cuml.decomposition import PCA as cuPCA
    print("Using GPU-accelerated centered PCA (RAPIDS cuML, svd_solver='full')...")
    pca = cuPCA(n_components=ncomp, svd_solver='full').fit(X_fit)
    evr = pca.explained_variance_ratio_
    try:
        import cupy as cp
        evr = cp.asnumpy(evr)
        k = min(_select_k_for_variance(np.asarray(evr)), ncomp)
        Xr = cp.asnumpy(pca.transform(X_all))
    except ImportError:
        evr = np.asarray(evr)
        k = min(_select_k_for_variance(evr), ncomp)
        Xr = np.asarray(pca.transform(X_all))
    return Xr[:, :k].astype(np.float32), k


def _fit_sample(X, max_sequences, n):
    """Seeded fit-sample for the CPU PCA fit (the dense svd_solver='full' fit on all rows is a CPU
    bottleneck). Returns X unchanged when max_sequences is 0/disabled or n is already below it."""
    if max_sequences and max_sequences > 0 and n > max_sequences:
        rng = np.random.default_rng(RANDOM_STATE)
        print(f"PCA fit-sample (CPU): {n} unique vectors > --max-sequences={max_sequences}; "
              f"fitting on {max_sequences} sampled vectors, transforming all.")
        return X[rng.choice(n, size=max_sequences, replace=False)]
    return X


def embedding_pca_reduce(X, args):
    """Centered PCA → the 95%-variance component count (cap SVD_MAX_COMPONENTS); transform ALL rows.

    Per-backend fit policy mirrors the sequence path exactly:
      - GPU (cuML): fit on ALL unique rows, no cap — like run_gpu_pipeline (sequence GPU fits on all).
      - CPU (sklearn): fit on a seeded --max-sequences sample — like run_cpu_pipeline, since dense
        svd_solver='full' on all rows is a CPU bottleneck.
    A GPU OOM/error falls back to the (fit-sampled).
    Returns (Xr float32, k)."""
    n, D = X.shape
    cap = min(SVD_MAX_COMPONENTS, n - 1, D)
    if cap < 1:
        return X.astype(np.float32), D

    if args.svd_backend in ('cuml', 'auto'):
        try:
            ncomp = min(cap, n - 1, D)   # GPU: fit on ALL unique rows (no cap)
            return _pca_reduce_cuml(X, X, ncomp)
        except Exception as e:  # noqa: BLE001 — OOM/import/CUDA all fall back to CPU
            if args.svd_backend == 'cuml':
                print(f"Error: cuML PCA failed and backend forced to 'cuml': {e}")
                raise
            print(f"cuML PCA unavailable or failed ({e}); falling back to CPU PCA (fit-sampled).")

    # CPU: fit on a seeded --max-sequences sample (matches the sequence CPU path), transform all.
    X_fit = _fit_sample(X, args.max_sequences, n)
    ncomp = min(cap, X_fit.shape[0] - 1, D)
    return _pca_reduce_sklearn(X_fit, X, ncomp)


def run_embedding_umap(Xn, umap_model, run_type, args):
    """Fit/transform UMAP on the reduced, L2-normalized matrix (already materialized — no per-chunk
    feature recompute). GPU: single fit_transform on all rows. CPU: fit on a sub-sample
    (UMAP_FIT_MAX_SAMPLE_SIZE), transform all rows. Default metric is Euclidean on both backends, which
    on L2-normalized vectors is a monotonic function of cosine similarity. Returns (n, 2) coords."""
    n = Xn.shape[0]
    start = time.time()
    if run_type == 'gpu':
        try:
            coords = np.asarray(umap_model.fit_transform(Xn))
            print(f"UMAP (GPU) completed in {time.time() - start:.2f} seconds.\n")
            return coords
        except Exception as e:  # noqa: BLE001
            print(f"Warning: GPU UMAP failed - {e}. Falling back to CPU UMAP...")
            umap_model, run_type = create_umap_model(
                'sklearn', args.umap_components, args.umap_neighbors, args.umap_min_dist)

    # Isolated, seeded RNG (deterministic, doesn't touch global numpy state) — matches _fit_sample.
    rng = np.random.default_rng(RANDOM_STATE)
    if n > UMAP_FIT_MAX_SAMPLE_SIZE:
        print(f"Fit sample ({n}) > {UMAP_FIT_MAX_SAMPLE_SIZE}. "
              f"Sampling {UMAP_FIT_MAX_SAMPLE_SIZE} vectors for UMAP fitting.")
        idx = rng.choice(n, size=UMAP_FIT_MAX_SAMPLE_SIZE, replace=False)
        umap_model.fit(Xn[idx])
    else:
        umap_model.fit(Xn)
    coords = np.asarray(umap_model.transform(Xn))
    print(f"UMAP (CPU) completed in {time.time() - start:.2f} seconds.\n")
    return coords


def _peak_rss_gib():
    """Peak resident set size of this process, in GiB. ru_maxrss is in KB on Linux (where the block
    software runs) and in bytes on macOS (local dev)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return peak / (1024 ** 3)   # bytes -> GiB
    return peak / (1024 ** 2)       # KB -> GiB


def run_embedding_mode(args, output_path):
    """Embedding feature path: load matrix → vector-dedup → centered PCA-95% → L2-normalize (eps-guarded,
    degenerate rows excluded) → Euclidean UMAP → broadcast back to all clonotypes → write outputs.

    Empty / too-few-rows inputs write an empty output and return (the k-mer path's guards are TSV/
    sequence-specific, so embedding mode carries its own here). Clonotypes ABSENT from the embedding
    column never enter this input, so they are simply absent from the output (sparse), like the k-mer
    path's no-sequence drop — not the invalid-character null-coord path."""
    print(f"Loading embedding matrix {args.matrix} ...")
    X, keys = load_embedding_matrix(args.matrix, args.key_col, args.dim_col, args.value_col)
    n_clonotypes = X.shape[0]
    D = X.shape[1] if n_clonotypes else 0
    print(f"Loaded {n_clonotypes} clonotypes x {D} embedding dims.")
    print(f"Peak memory after matrix load: {_peak_rss_gib():.2f} GiB")

    if n_clonotypes == 0:
        print("Warning: Embedding matrix is empty — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to empty embedding matrix.")
        return

    min_required = max(args.umap_neighbors + 1, 4)

    # Vector-dedup (exact, bit-identical for same-model embeddings): collapse identical rows before the
    # fit, re-expand after via the inverse index.
    uniq, inv = np.unique(X, axis=0, return_inverse=True)
    inv = inv.ravel()
    n_unique = uniq.shape[0]
    if n_unique < n_clonotypes:
        print(f"Deduplicating vectors: {n_clonotypes} → {n_unique} unique "
              f"({n_clonotypes - n_unique} identical vectors collapsed for PCA/UMAP).")

    if n_unique < min_required:
        print(f"Warning: Not enough unique embedding vectors for UMAP "
              f"(required {min_required}, unique {n_unique}) — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient unique vectors.")
        return

    # Centered PCA → 95% variance (cap 500). GPU fits on all unique rows (no cap, like the sequence
    # GPU path); CPU fits on a seeded --max-sequences sample.
    start_pca = time.time()
    Xr, k_pca = embedding_pca_reduce(uniq, args)
    print(f"Centered PCA: {uniq.shape[1]} → {k_pca} components (95% variance, cap "
          f"{SVD_MAX_COMPONENTS}) in {time.time() - start_pca:.2f}s.")

    # Degenerate (near-zero post-PCA residual) detection BEFORE normalization: such a vector sits at the
    # dataset mean, so its normalized direction is numerical noise — exclude from the fit, null coords,
    # count. The eps-guarded L2-normalize itself can't 0/0; Euclidean never NaNs from a zero row.
    pre_norm = np.linalg.norm(Xr, axis=1)
    valid = pre_norm > DEGENERATE_NORM_THRESHOLD
    n_degenerate = int((~valid).sum())
    if n_degenerate:
        print(f"Warning: {n_degenerate} unique vector(s) near-zero norm post-PCA "
              f"(centered-PCA residual ≈ dataset mean) — excluded from the UMAP fit, null coords.")
    n_valid = int(valid.sum())
    if n_valid < min_required:
        print(f"Warning: Not enough non-degenerate vectors for UMAP "
              f"(required {min_required}, valid {n_valid}) — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient non-degenerate vectors.")
        return

    # Normalize only the valid (non-degenerate) rows we actually fit/transform — avoids dividing a
    # near-zero residual by eps and amplifying numerical noise into rows we'd then discard.
    Xn_valid = l2_normalize(Xr[valid])

    umap_model, run_type = create_umap_model(
        args.umap_backend, args.umap_components, args.umap_neighbors, args.umap_min_dist)

    print(f"Running UMAP on {n_valid} non-degenerate unique vectors "
          f"(L2-normalized, Euclidean ≡ cosine)...")
    coords_valid = run_embedding_umap(Xn_valid, umap_model, run_type, args)

    # Degenerate uniques get NaN coords; assemble the full unique-coord matrix, then broadcast to every
    # clonotype via the dedup inverse index. NaN rows surface as null coords downstream.
    coords_unique = np.full((n_unique, args.umap_components), np.nan, dtype=np.float64)
    coords_unique[valid] = coords_valid
    coords_all = coords_unique[inv]

    print(f"Embedding UMAP summary: model={args.embedding_model or '(unknown)'}, mode=embedding, "
          f"clonotypes={n_clonotypes}, unique={n_unique}, PCA k={k_pca}, "
          f"degenerate-excluded={n_degenerate}")

    write_embedding_outputs(args, keys, coords_all, output_path,
                            n_clonotypes=n_clonotypes, n_unique=n_unique, k_pca=k_pca,
                            n_degenerate=n_degenerate)

    # Peak memory across the whole embedding run (matrix load + dedup + PCA + UMAP). Reported so the
    # mem control can be sized to the actual N x D footprint, and to calibrate any backend cap.
    print(f"Peak memory (RSS) for embedding run: {_peak_rss_gib():.2f} GiB "
          f"(N={n_clonotypes}, unique={n_unique}, D={D}, PCA backend fit + UMAP).")


def write_embedding_outputs(args, keys, coords_all, output_path, n_clonotypes, n_unique, k_pca,
                            n_degenerate):
    """Write the UMAP coordinates TSV (clonotypeKey, UMAP1, UMAP2, …) and the processing-log summary.

    Degenerate rows carry NaN coordinates; NaN is converted to null so the TSV has empty cells (matching
    the k-mer path's null coords for excluded sequences), not the literal string "NaN"."""
    coord_dict = {KEY_COL: [str(k) for k in keys]}
    for i in range(args.umap_components):
        coord_dict[f'UMAP{i + 1}'] = coords_all[:, i].tolist()
    df = pl.DataFrame(coord_dict)
    df = df.with_columns([
        pl.when(pl.col(f'UMAP{i + 1}').is_nan())
          .then(None)
          .otherwise(pl.col(f'UMAP{i + 1}'))
          .alias(f'UMAP{i + 1}')
        for i in range(args.umap_components)
    ])
    df.write_csv(output_path, separator='\t', null_value='')

    # Saved as skipped_clonotypes_summary.txt to satisfy the exec's saveFile contract (shared with the
    # k-mer path). Clonotypes absent from the embedding column never enter this input → they are absent
    # from the output (sparse); that count is stated in the upstream embedding block.
    summary_path = os.path.join(args.output_dir, 'skipped_clonotypes_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Embedding-mode UMAP summary\n")
        f.write(f"Embedding model: {args.embedding_model or '(unknown)'}\n")
        f.write(f"Clonotypes embedded: {n_clonotypes}\n")
        f.write(f"Unique vectors (after dedup): {n_unique}\n")
        f.write(f"PCA components (95% variance): {k_pca}\n")
        f.write(f"Degenerate clonotypes excluded (null coords): {n_degenerate}\n")
    print(f"Embedding UMAP summary saved to {summary_path}")


# ============================================================================
# Output writing
# ============================================================================

def write_outputs(args, df, df_valid, df_invalid, umap_embed_all, keys_all, n_invalid,
                  output_path):
    """Write the UMAP coordinates TSV and the skipped-clonotypes summary."""
    # Build coordinates dataframe (valid sequences only) and left-join onto all input rows.
    # Invalid-character sequences end up with null coordinates.
    coord_dict = {KEY_COL: keys_all}
    for i in range(args.umap_components):
        coord_dict[f'UMAP{i + 1}'] = umap_embed_all[:, i].tolist()
    coords_df = pl.DataFrame(coord_dict)
    output_df = df.select(KEY_COL).join(coords_df, on=KEY_COL, how='left')
    output_df.write_csv(output_path, separator='\t', null_value='')

    seq_type_name = "amino acid" if args.alphabet == 'aminoacid' else "nucleotide"
    skipped_summary_path = os.path.join(args.output_dir, 'skipped_clonotypes_summary.txt')
    with open(skipped_summary_path, 'w') as f:
        f.write(f"Number of clonotypes skipped due to invalid {seq_type_name} sequences: "
                f"{n_invalid}\n")
        f.write(f"Total clonotypes processed: {len(df)}\n")
        f.write(f"Valid clonotypes: {len(df_valid)}\n")
        f.write(f"Skipped clonotypes: {n_invalid}\n\n")
        if n_invalid > 0:
            f.write("Skipped clonotypes:\n")
            for row in df_invalid.select([KEY_COL, SEQ_COL]).iter_rows():
                f.write(f"{row[0]}\t{row[1]}\n")
    print(f"Skipped clonotypes summary saved to {skipped_summary_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    args = parse_args()

    if args.k_mer_size is None:
        args.k_mer_size = 3 if args.alphabet == 'aminoacid' else 6

    os.makedirs(args.output_dir, exist_ok=True)

    # k-mer banner is sequence-specific; embedding mode prints its own banner in run_embedding_mode.
    if args.encoding != 'embedding':
        sequence_type = "amino acid" if args.alphabet == 'aminoacid' else "nucleotide"
        max_seq_str = str(args.max_sequences) if args.max_sequences > 0 else "disabled"
        print(f"Starting k-mer UMAP analysis for {sequence_type} sequences")
        print(f"Input file: {args.input}")
        print(f"Output file: {args.umap_output}")
        print(f"Parameters: alphabet={args.alphabet}, k-mer size={args.k_mer_size}, "
              f"UMAP components={args.umap_components}, "
              f"UMAP neighbors={args.umap_neighbors}, "
              f"UMAP min_dist={args.umap_min_dist}, "
              f"SVD Backend={args.svd_backend.upper()}, "
              f"UMAP Backend={args.umap_backend.upper()}, "
              f"max-sequences={max_seq_str}")

    validate_args(args)

    output_path = os.path.join(args.output_dir, args.umap_output)

    # Embedding feature path: read the parquet matrix and run centered PCA-95% →
    # L2-normalize → Euclidean UMAP. Branches BEFORE load_and_filter_input, which is TSV/sequence-
    # specific. Carries its own empty/min-rows guards inside run_embedding_mode.
    if args.encoding == 'embedding':
        if not args.matrix:
            print("Error: --encoding embedding requires --matrix (the embedding parquet file).")
            sys.exit(1)
        print("Embedding feature mode: running UMAP on learned embedding vectors.")
        start_time_emb = time.time()
        run_embedding_mode(args, output_path)
        print(f"\nTotal analysis completed in {time.time() - start_time_emb:.2f} seconds.")
        return

    if not args.input:
        print("Error: -i/--input is required for kmer/pos-kmer encodings.")
        sys.exit(1)

    start_time_load = time.time()
    df, df_valid, df_invalid, n_invalid = load_and_filter_input(args, output_path)
    print(f"Input loading and preprocessing completed in "
          f"{time.time() - start_time_load:.2f} seconds.\n")

    umap_model, run_type = create_umap_model(
        args.umap_backend, args.umap_components, args.umap_neighbors, args.umap_min_dist)

    min_required_sequences = max(args.umap_neighbors + 1, 4)
    if len(df_valid) < min_required_sequences:
        print(f"Warning: Not enough sequences for UMAP analysis "
              f"(required {min_required_sequences}, available {len(df_valid)}) — "
              f"writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient sequences.")
        sys.exit(0)

    n_sequences_all = len(df_valid)
    keys_all = df_valid[KEY_COL].to_list()

    # Make sure all sequences are in uppercase
    df_valid_upper = df_valid.with_columns(
        pl.col(SEQ_COL).str.to_uppercase().alias(SEQ_COL)
    )
    # Collapse exact duplicates before fitting (They can cause issues in GPU mode)
    df_unique = (
        df_valid_upper.unique(subset=[SEQ_COL], keep='first', maintain_order=True)
        .with_row_index('_uidx')
    )
    sequences_unique = df_unique[SEQ_COL].to_list()
    n_unique = len(sequences_unique)
    if n_unique < n_sequences_all:
        print(f"Deduplicating sequences: {n_sequences_all} valid → {n_unique} unique "
              f"({n_sequences_all - n_unique} duplicates collapsed for SVD/UMAP).")

    if n_unique < min_required_sequences:
        print(f"Warning: Not enough unique sequences for UMAP analysis "
              f"(required {min_required_sequences}, unique {n_unique}) — "
              f"writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient unique sequences.")
        sys.exit(0)

    if run_type == 'gpu':
        umap_embed_unique = run_gpu_pipeline(args, sequences_unique, umap_model)
    else:
        umap_embed_unique = run_cpu_pipeline(args, df_unique, sequences_unique, umap_model)

    # Broadcast unique-row embeddings back to every valid row
    index_map = (
        df_valid_upper.select(SEQ_COL)
        .join(df_unique.select([SEQ_COL, '_uidx']),
              on=SEQ_COL, how='left', maintain_order='left')
        ['_uidx'].to_numpy()
    )
    umap_embed_all = umap_embed_unique[index_map]

    start_time_save = time.time()
    write_outputs(args, df, df_valid, df_invalid, umap_embed_all, keys_all, n_invalid, output_path)
    print(f"UMAP embeddings saved to {output_path} in "
          f"{time.time() - start_time_save:.2f} seconds.")

    print(f"\nTotal analysis completed in {time.time() - start_time_load:.2f} seconds.")

    if args.store_models and args.umap_backend == 'parametric-umap':
        umap_model.save(args.output_dir)


if __name__ == '__main__':
    main()
