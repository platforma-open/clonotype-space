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
# Execution tracker
# ============================================================================

# Records which backend actually ran each compute stage. Populated by
# _mark_exec() at the exact point where the GPU or CPU library returned a
# result; the closing summary in main() reports these so the user has a
# ground-truth record, not just intent-of-use logs.
_EXEC = {'svd': None, 'umap': None}


def _mark_exec(stage, backend, detail=''):
    """Stamp a definitive execution marker once a compute stage actually completes."""
    _EXEC[stage] = backend
    bar = '=' * 64
    suffix = f' — {detail}' if detail else ''
    print(bar)
    print(f'>>> {stage.upper()} EXECUTED ON {backend}{suffix}')
    print(bar)


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


def log_gpu_status():
    """
    Print a single, prominent log line stating whether the GPU pipeline is
    usable in this run. Probes the same imports + CUDA handshake that
    create_svd_model / create_umap_model do later, so a "GPU IN USE" banner
    here is a reliable predictor of which code path runs.
    """
    try:
        import cuml  # noqa: F401
        import cupy as cp
    except Exception as e:
        print(f"GPU STATUS: NOT IN USE - cuML/CuPy import failed ({type(e).__name__}: {e}). "
              f"Running on CPU.")
        return

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        print(f"GPU STATUS: NOT IN USE - no CUDA device visible "
              f"({type(e).__name__}: {e}). Running on CPU.")
        return

    if device_count == 0:
        print("GPU STATUS: NOT IN USE - CUDA reports 0 devices. Running on CPU.")
        return

    try:
        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        free_gb, total_gb = dev.mem_info
        free_gb /= 1024 ** 3
        total_gb /= 1024 ** 3
        print(f"GPU STATUS: IN USE - {name} ({total_gb:.1f} GiB VRAM, {free_gb:.1f} GiB free). "
              f"Will use RAPIDS cuML SVD/UMAP path.")
    except Exception as e:
        print(f"GPU STATUS: NOT IN USE - CUDA device probe failed "
              f"({type(e).__name__}: {e}). Running on CPU.")


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
    _mark_exec('svd', 'CPU', 'sklearn TruncatedSVD.fit() returned')
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
                _mark_exec('svd', 'GPU', 'CuPy sparse svds() returned')
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
            # Overwrites the earlier GPU mark — the recovery path lands on sklearn.
            _mark_exec('svd', 'CPU', 'sklearn fit_transform() recovered after GPU transform failure')
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
    parser.add_argument('--dims', default=None,
                        help='Embedding vector length D (--encoding embedding), from the input column\'s '
                             'pl7.app/embedding/length annotation. Lets the streaming loader resolve D '
                             'without a scan; when omitted D is inferred as max(embeddingDim)+1 in the '
                             'first row group.')
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
        _mark_exec('umap', 'GPU', 'cuML UMAP.fit_transform() returned')
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
    _mark_exec('umap', 'CPU', 'umap-learn fallback after GPU UMAP failure')
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
    _mark_exec('umap', 'CPU', 'umap-learn UMAP.fit() returned')
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
#   centered PCA-95% → L2-normalize → Euclidean UMAP.
# Centered PCA removes ESM-2's large non-discriminative shared mean, so the post-PCA
# L2-normalize + Euclidean (≡ cosine ranking) then measures the mean-removed residuals.
#
# Memory handling for large inputs: the full N x D matrix is NEVER materialized. The CPU path
# STREAMS the long-format parquet twice — once to collect a bounded, seeded fit sample
# (~max_sequences x D), once to project every clonotype through the fitted PCA/UMAP in batches — so
# peak memory is bounded by the fit sample, independent of N. The GPU path (G2) streams the load
# straight into a device array (host stays bounded) and keeps cuML's fit-on-all behaviour unchanged.
# Exact-duplicate vectors are NOT de-duplicated before the fit to avoid memory scaling.

# Fixed, data-derived batch size for the streaming reads: 4M long-format rows per pyarrow batch.
# Deliberately NOT scaled by the allocated memory — a fixed batch keeps assembly deterministic and the
# per-batch footprint constant regardless of the RAM the backend granted.
EMBEDDING_STREAM_BATCH_ROWS = 4_000_000

# GPU IncrementalPCA fit batch: number of clonotypes per partial_fit call. Decoupled from the parquet
# stream granularity (a 4M-row read yields ~4M/D clonotypes) by buffering whole-clonotype blocks up to
# this target. Fixed (not memory-scaled) so the incremental-SVD result is reproducible across backends;
# must stay >= the PCA component count (n_components <= batch rows), which holds by a wide margin
# (100k >> SVD_MAX_COMPONENTS=500).
GPU_PCA_FIT_BATCH = 100_000


class _EmptyEmbeddingResult(Exception):
    """Signals that an embedding path already wrote the empty output files and processing should
    unwind to the caller (empty / too-few-vector inputs)."""


def _open_matrix(path, key_col, dim_col, value_col, dims):
    """Open the long-format parquet and resolve D without scanning the file. Validates the three
    columns exist; D comes from --dims (the pl7.app/embedding/length annotation) when given, else
    max(embeddingDim)+1 in the first row group. Returns (ParquetFile, D)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    for role, col in [("--key-col", key_col), ("--dim-col", dim_col), ("--value-col", value_col)]:
        if col not in names:
            raise ValueError(f"{role} {col!r} is not a column in {path}; available columns: {names}")
    if pf.metadata.num_rows == 0:            # empty input -> D irrelevant (caller writes empty output)
        return pf, int(dims) if dims else 0
    if dims:
        return pf, int(dims)
    D = int(pl.from_arrow(pf.read_row_group(0, columns=[dim_col])).get_column(dim_col).max()) + 1
    return pf, D


def _clonotype_count(pf, D):
    """Clonotype count = long rows / D, taken from the parquet footer (no scan)."""
    return 0 if D <= 0 else pf.metadata.num_rows // D


def _stream_clonotypes(pf, key_col, dim_col, value_col, D, batch_rows=EMBEDDING_STREAM_BATCH_ROWS):
    """Yield (keys, matrix) blocks of whole clonotypes, assembled from contiguous long-format rows and
    holding back the trailing clonotype that may continue in the next batch. Memory bounded to ~one
    batch. Copied from embedding-clustering; keep the guards in sync."""
    def build(ks, ds, vs):
        # np.unique -> distinct keys `uk` (sorted) + `inv` (per-row index into uk). Scattering
        # mat[inv, ds] = vs drops each value into its (clonotype, embeddingDim) cell, so row order
        # within the block does not matter.
        uk, inv = np.unique(ks, return_inverse=True)
        # Row-count guard: each clonotype must have exactly D contiguous rows.
        if (np.bincount(inv) != D).any():
            raise ValueError("a clonotype did not have exactly D contiguous rows -- the embedding "
                             "matrix is not clonotype-blocked (the streaming loader needs contiguous rows).")
        # Dimension-range guard: embeddingDim must be a 0..D-1 index.
        if ds.min() < 0 or ds.max() >= D:
            raise ValueError(f"embeddingDim out of range [0, {D}); saw {int(ds.min())}..{int(ds.max())} "
                             f"-- pass the correct --dims (the pl7.app/embedding/length annotation).")
        # Completeness: NaN-fill, scatter, require no cell left unset -- catches a duplicated dim (hence
        # a missing dim -> an unfilled hole) and any NaN in the value column.
        mat = np.full((uk.shape[0], D), np.nan, dtype=np.float32)
        mat[inv, ds] = vs
        if np.isnan(mat).any():
            raise ValueError("ragged embedding matrix: a (clonotype, dim) cell is missing, duplicated, "
                             "or NaN -- a clonotype has a partial/invalid embedding-dimension set.")
        return uk, mat

    leftover = None
    for b in pf.iter_batches(columns=[key_col, dim_col, value_col], batch_size=batch_rows):
        t = pl.from_arrow(b)
        k = t.get_column(key_col).to_numpy().astype(object)
        d = t.get_column(dim_col).to_numpy()
        v = t.get_column(value_col).cast(pl.Float32).to_numpy()
        # A clonotype's D rows can straddle a batch boundary, so prepend the previous batch's
        # carried-over trailing clonotype before assembling this batch.
        if leftover is not None:
            k = np.concatenate([leftover[0], k])
            d = np.concatenate([leftover[1], d])
            v = np.concatenate([leftover[2], v])
        # Hold back the LAST key's rows (they may continue in the next batch); assemble + emit the rest.
        tail = k == k[-1]
        leftover = (k[tail], d[tail], v[tail])
        keep = ~tail
        if keep.any():
            yield build(k[keep], d[keep], v[keep])
    # The final trailing clonotype has no continuation -- emit it.
    if leftover is not None:
        yield build(*leftover)


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


def _fit_pca_cpu(X_fit):
    """Centered PCA (sklearn, svd_solver='full', deterministic) fit on the fit sample. Returns
    (pca, k) with k the 95%-variance component count (cap SVD_MAX_COMPONENTS); any batch is projected
    as pca.transform(batch)[:, :k]. Returns (None, D) for a sample too small to fit (guarded upstream —
    the caller writes an empty output before this is reached in practice)."""
    from sklearn.decomposition import PCA
    n_fit, D = X_fit.shape
    ncomp = min(SVD_MAX_COMPONENTS, n_fit - 1, D)
    if ncomp < 1:
        return None, D
    print("Fitting centered PCA (scikit-learn, svd_solver='full') on the fit sample...")
    pca = PCA(n_components=ncomp, svd_solver='full').fit(X_fit)  # centered by default
    k = min(_select_k_for_variance(pca.explained_variance_ratio_), ncomp)
    _mark_exec('svd', 'CPU', 'sklearn PCA-95% (embedding mode, streamed)')
    return pca, k


def _fit_umap_cpu(umap_model, Xn_fit):
    """Fit umap-learn on the L2-normalized fit-sample reduction. Sub-samples to
    UMAP_FIT_MAX_SAMPLE_SIZE first (seeded, isolated RNG), matching the pre-streaming CPU UMAP fit
    size. Xn_fit is already L2-normalized (Euclidean ≡ cosine ranking)."""
    n = Xn_fit.shape[0]
    rng = np.random.default_rng(RANDOM_STATE)
    if n > UMAP_FIT_MAX_SAMPLE_SIZE:
        print(f"UMAP fit sample ({n}) > {UMAP_FIT_MAX_SAMPLE_SIZE}. "
              f"Sub-sampling {UMAP_FIT_MAX_SAMPLE_SIZE} vectors for the UMAP fit.")
        idx = rng.choice(n, size=UMAP_FIT_MAX_SAMPLE_SIZE, replace=False)
        umap_model.fit(Xn_fit[idx])
    else:
        umap_model.fit(Xn_fit)
    _mark_exec('umap', 'CPU', 'umap-learn UMAP.fit() returned (embedding mode, streamed)')


def _transform_batch_cpu(pca, k, umap_model, n_components, X_batch):
    """Project one clonotype block through the fitted PCA + UMAP (out-of-sample transform). Degenerate
    (near-zero post-PCA residual) rows get NaN coords (null downstream), matching the fit-side
    exclusion. Returns (coords float64 (b, n_components), n_degenerate)."""
    reduced = X_batch if pca is None else pca.transform(X_batch)[:, :k]
    coords = np.full((reduced.shape[0], n_components), np.nan, dtype=np.float64)
    valid = np.linalg.norm(reduced, axis=1) > DEGENERATE_NORM_THRESHOLD
    if valid.any():
        coords[valid] = np.asarray(umap_model.transform(l2_normalize(reduced[valid])))
    return coords, int((~valid).sum())


def _run_embedding_cpu(pf, D, n_clonotypes, min_required, args, output_path):
    """CPU two-pass streaming path. Pass A: stream once, gather a seeded fit sample bounded to
    ~max_sequences x D. Fit PCA + UMAP on it. Pass B: stream again, project every clonotype through the
    fitted models in batches, accumulating only the 2-D coordinates. Never holds the full N x D matrix.
    Returns (keys, coords, n_degenerate, k_pca); raises _EmptyEmbeddingResult when there are too few
    non-degenerate vectors (empty output already written)."""
    fit_size = args.max_sequences if (args.max_sequences and args.max_sequences > 0) else n_clonotypes
    fit_size = min(fit_size, n_clonotypes)

    # --- Pass A: gather the seeded fit sample. choice-over-N is the same sampling algorithm as the
    #     pre-streaming _fit_sample; collected in stream order via a global row counter. ---
    rng = np.random.default_rng(RANDOM_STATE)
    if fit_size < n_clonotypes:
        sel = np.sort(rng.choice(n_clonotypes, size=fit_size, replace=False))
        print(f"Fit sample: {n_clonotypes} clonotypes > --max-sequences={args.max_sequences}; "
              f"gathering {fit_size} sampled vectors while streaming (PCA/UMAP fit on the sample, all "
              f"clonotypes transformed).")
    else:
        sel = np.arange(n_clonotypes)
        print(f"Fit sample: {n_clonotypes} clonotypes <= --max-sequences; fitting on all.")

    X_fit = np.empty((fit_size, D), dtype=np.float32)
    g = 0
    for _, mat in _stream_clonotypes(pf, args.key_col, args.dim_col, args.value_col, D):
        b = mat.shape[0]
        lo = int(np.searchsorted(sel, g))
        hi = int(np.searchsorted(sel, g + b))
        if hi > lo:
            X_fit[lo:hi] = mat[sel[lo:hi] - g]
        g += b
    print(f"Fit sample gathered: {X_fit.shape[0]} x {D} (peak RSS {_peak_rss_gib():.2f} GiB).")

    # --- Fit centered PCA on the sample ---
    start_pca = time.time()
    pca, k_pca = _fit_pca_cpu(X_fit)
    Xr_fit = X_fit if pca is None else pca.transform(X_fit)[:, :k_pca]
    print(f"Centered PCA: {D} → {k_pca} components (95% variance, cap {SVD_MAX_COMPONENTS}) "
          f"in {time.time() - start_pca:.2f}s.")

    # --- Degenerate check on the fit sample; fit UMAP on its non-degenerate rows ---
    valid_fit = np.linalg.norm(Xr_fit, axis=1) > DEGENERATE_NORM_THRESHOLD
    n_valid_fit = int(valid_fit.sum())
    if n_valid_fit < min_required:
        print(f"Warning: Not enough non-degenerate vectors in the fit sample for UMAP "
              f"(required {min_required}, valid {n_valid_fit}) — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient non-degenerate vectors.")
        raise _EmptyEmbeddingResult()

    umap_model, _ = create_umap_model('sklearn', args.umap_components, args.umap_neighbors,
                                      args.umap_min_dist)
    print(f"Fitting UMAP on {n_valid_fit} non-degenerate fit-sample vectors "
          f"(L2-normalized, Euclidean ≡ cosine)...")
    start_umap = time.time()
    _fit_umap_cpu(umap_model, l2_normalize(Xr_fit[valid_fit]))
    print(f"UMAP fit completed in {time.time() - start_umap:.2f}s.")

    # --- Pass B: project every clonotype in batches (out-of-sample transform) ---
    start_tr = time.time()
    keys_parts, coords_parts, n_degenerate = [], [], 0
    for ks, mat in _stream_clonotypes(pf, args.key_col, args.dim_col, args.value_col, D):
        c, nd = _transform_batch_cpu(pca, k_pca, umap_model, args.umap_components, mat)
        keys_parts.append(ks)
        coords_parts.append(c)
        n_degenerate += nd
    keys = np.concatenate(keys_parts) if keys_parts else np.array([], dtype=object)
    coords = np.vstack(coords_parts) if coords_parts else np.empty((0, args.umap_components))
    print(f"Transformed all {keys.shape[0]} clonotypes in {time.time() - start_tr:.2f}s "
          f"(degenerate-excluded={n_degenerate}).")
    return keys, coords, n_degenerate, k_pca


def _run_embedding_gpu(pf, D, n_clonotypes, min_required, args, output_path):
    """GPU path: stream the long-format parquet through cuML IncrementalPCA so the full N x D matrix is
    NEVER resident on the device (the old fit-on-all cuPCA held it whole and OOM'd at scale). Two passes
    over the parquet: pass 1 partial_fit IncrementalPCA on ~GPU_PCA_FIT_BATCH-clonotype buffers (device
    memory bounded to one batch + the components); pass 2 re-stream and transform every clonotype into
    the reduced N x k on device. Then cuML UMAP fit_transform on all non-degenerate reduced points —
    still fit-on-all, but on the small reduced matrix rather than the raw embeddings. Returns
    (keys, coords, n_degenerate, k_pca). Raises _EmptyEmbeddingResult for too-few valid vectors; raises
    on any GPU/CUDA error so the caller can fall back to the CPU streaming path."""
    import cupy as cp
    import cuml  # noqa: F401
    from cuml.decomposition import IncrementalPCA as cuIncrementalPCA

    ncomp = min(SVD_MAX_COMPONENTS, n_clonotypes - 1, D)

    # --- Pass 1: streaming centered IncrementalPCA. Whole-clonotype blocks are buffered on the host to
    #     ~GPU_PCA_FIT_BATCH, then moved to the device one batch at a time for partial_fit. One full
    #     batch is held back so the FINAL partial_fit is never a sub-batch remainder (guaranteeing every
    #     batch has >= ncomp rows); an input below one batch is fit in a single call. ---
    ipca = cuIncrementalPCA(n_components=ncomp)
    print(f"Fitting centered IncrementalPCA (RAPIDS cuML) on ALL clonotypes, streaming in "
          f"~{GPU_PCA_FIT_BATCH}-clonotype batches (GPU)...")
    start_pca = time.time()
    n_batches = 0
    prev, buf, buf_rows = None, [], 0
    for _, mat in _stream_clonotypes(pf, args.key_col, args.dim_col, args.value_col, D):
        buf.append(mat)
        buf_rows += mat.shape[0]
        if buf_rows >= GPU_PCA_FIT_BATCH:
            block = np.concatenate(buf)
            if prev is not None:
                ipca.partial_fit(cp.asarray(prev))
                n_batches += 1
            prev, buf, buf_rows = block, [], 0
    remainder = np.concatenate(buf) if buf else None
    tail = remainder if prev is None else (prev if remainder is None
                                           else np.concatenate([prev, remainder]))
    if tail is not None:
        ipca.partial_fit(cp.asarray(tail))
        n_batches += 1

    k_pca = min(_select_k_for_variance(cp.asnumpy(ipca.explained_variance_ratio_)), ncomp)
    _mark_exec('svd', 'GPU', 'cuML IncrementalPCA-95% (embedding mode, streamed fit)')
    print(f"Centered IncrementalPCA: {D} → {k_pca} components (95% variance, cap {SVD_MAX_COMPONENTS}) "
          f"over {n_batches} batch(es) in {time.time() - start_pca:.2f}s "
          f"(host peak RSS {_peak_rss_gib():.2f} GiB).")

    # --- Pass 2: re-stream and transform every clonotype into the reduced N x k on device. This N x k
    #     matrix (k <= SVD_MAX_COMPONENTS) is the only large device array now — never the raw N x D. ---
    Xr = cp.empty((n_clonotypes, k_pca), dtype=cp.float32)
    keys = np.empty(n_clonotypes, dtype=object)
    g = 0
    for ks, mat in _stream_clonotypes(pf, args.key_col, args.dim_col, args.value_col, D):
        b = mat.shape[0]
        Xr[g:g + b] = ipca.transform(cp.asarray(mat))[:, :k_pca]
        keys[g:g + b] = ks
        g += b
    print(f"Transformed {g} clonotypes to the reduced {k_pca}-d space "
          f"({Xr.nbytes / 1024**3:.2f} GiB on device).")

    # Degenerate detection + eps-guarded L2-normalize on device (mirrors l2_normalize).
    pre_norm = cp.linalg.norm(Xr, axis=1)
    valid = pre_norm > DEGENERATE_NORM_THRESHOLD
    n_valid = int(valid.sum())
    n_degenerate = int(n_clonotypes - n_valid)
    if n_degenerate:
        print(f"Warning: {n_degenerate} vector(s) near-zero norm post-PCA — excluded, null coords.")
    if n_valid < min_required:
        print(f"Warning: Not enough non-degenerate vectors for UMAP "
              f"(required {min_required}, valid {n_valid}) — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient non-degenerate vectors.")
        raise _EmptyEmbeddingResult()
    Xn = Xr[valid] / cp.maximum(pre_norm[valid][:, None], 1e-12)
    del Xr, pre_norm   # free the reduced matrix before dedup + UMAP

    # Collapse EXACT-duplicate reduced vectors before the UMAP fit. Identical embeddings map to an
    # identical PCA value (PCA is a deterministic linear map), and cuML's GPU UMAP places zero-distance
    # duplicate points erratically (scattered artefacts).
    Xn_host = cp.asnumpy(Xn)
    del Xn
    uniq, inv = np.unique(Xn_host, axis=0, return_inverse=True)
    inv = inv.ravel()   # numpy 2.x can return a 2-D inverse for axis=0; flatten to (n_valid,)
    n_unique = int(uniq.shape[0])
    if n_unique < Xn_host.shape[0]:
        print(f"Collapsing {Xn_host.shape[0]} → {n_unique} unique reduced vectors "
              f"({Xn_host.shape[0] - n_unique} exact duplicates) before the GPU UMAP fit.")
    if n_unique < min_required:
        print(f"Warning: Not enough unique non-degenerate vectors for UMAP "
              f"(required {min_required}, unique {n_unique}) — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient unique vectors.")
        raise _EmptyEmbeddingResult()

    umap_model, run_type = create_umap_model(args.umap_backend, args.umap_components,
                                             args.umap_neighbors, args.umap_min_dist)
    if run_type != 'gpu':
        raise RuntimeError("cuML UMAP unavailable despite a usable GPU PCA — deferring to CPU path.")
    print(f"Running UMAP (GPU, fit_transform on {n_unique} unique non-degenerate vectors)...")
    start_umap = time.time()
    uniq_coords = cp.asnumpy(umap_model.fit_transform(cp.asarray(uniq)))
    _mark_exec('umap', 'GPU', 'cuML UMAP.fit_transform() returned (embedding mode, dedup + streamed fit)')
    print(f"UMAP (GPU) completed in {time.time() - start_umap:.2f}s.")
    coords_valid = uniq_coords[inv]   # broadcast each unique's coords back to every valid clonotype

    coords = np.full((n_clonotypes, args.umap_components), np.nan, dtype=np.float64)
    coords[cp.asnumpy(valid)] = coords_valid
    return keys, coords, n_degenerate, k_pca


def _peak_rss_gib():
    """Peak resident set size of this process, in GiB. ru_maxrss is in KB on Linux (where the block
    software runs) and in bytes on macOS (local dev)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return peak / (1024 ** 3)   # bytes -> GiB
    return peak / (1024 ** 2)       # KB -> GiB


def run_embedding_mode(args, output_path):
    """Embedding feature path: stream matrix → centered PCA-95% → L2-normalize (degenerate rows
    excluded, null coords) → Euclidean UMAP → write outputs. The full N x D matrix is never
    materialized (see the module comment above). Empty / too-few-clonotype inputs write an empty
    output. Clonotypes ABSENT from the embedding column never enter this input, so they are simply
    absent from the output (sparse) — not the invalid-character null-coord path."""
    print(f"Opening embedding matrix {args.matrix} ...")
    pf, D = _open_matrix(args.matrix, args.key_col, args.dim_col, args.value_col, args.dims)
    n_clonotypes = _clonotype_count(pf, D)
    print(f"Embedding matrix: {n_clonotypes} clonotypes x {D} embedding dims "
          f"(from parquet footer; no full load).")

    if n_clonotypes == 0:
        print("Warning: Embedding matrix is empty — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to empty embedding matrix.")
        return

    min_required = max(args.umap_neighbors + 1, 4)
    if n_clonotypes < min_required:
        print(f"Warning: Not enough clonotypes for UMAP (required {min_required}, "
              f"have {n_clonotypes}) — writing empty output.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir,
                                     "UMAP analysis skipped due to insufficient clonotypes.")
        return

    # Backend: try the GPU path unless a backend is pinned to sklearn; fall back to CPU streaming
    # on any GPU/CUDA failure. Neither path holds the full N x D matrix in host RAM.
    try_gpu = args.umap_backend in ('cuml', 'auto') and args.svd_backend != 'sklearn'
    used_gpu = False
    result = None
    if try_gpu:
        try:
            result = _run_embedding_gpu(pf, D, n_clonotypes, min_required, args, output_path)
            used_gpu = True
        except _EmptyEmbeddingResult:
            return
        except Exception as e:  # noqa: BLE001 — OOM/import/CUDA all fall back to CPU streaming
            if args.umap_backend == 'cuml':
                print(f"Error: GPU embedding path failed and --umap-backend forced to 'cuml': {e}")
                raise
            print(f"GPU embedding path unavailable or failed ({e}); falling back to CPU streaming.")

    if result is None:
        try:
            result = _run_embedding_cpu(pf, D, n_clonotypes, min_required, args, output_path)
        except _EmptyEmbeddingResult:
            return

    keys, coords, n_degenerate, k_pca = result
    print(f"Embedding UMAP summary: model={args.embedding_model or '(unknown)'}, mode=embedding, "
          f"clonotypes={n_clonotypes}, PCA k={k_pca}, degenerate-excluded={n_degenerate}")
    write_embedding_outputs(args, keys, coords, output_path,
                            n_clonotypes=n_clonotypes, k_pca=k_pca, n_degenerate=n_degenerate)

    # Peak memory across the whole embedding run. Reported so the mem control can be calibrated to the
    # actual (now N-independent) footprint — a fit-sample-bounded peak, not the old N x D load.
    print(f"Peak memory (RSS) for embedding run: {_peak_rss_gib():.2f} GiB "
          f"(N={n_clonotypes}, D={D}, "
          f"{'GPU fit-on-all + streamed load' if used_gpu else 'CPU streamed sample + batched transform'}).")


def write_embedding_outputs(args, keys, coords_all, output_path, n_clonotypes, k_pca,
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

    # Banner-style GPU/CPU pipeline indicator, printed before any heavy work so
    # the user can confirm at a glance which path will run. Honors --svd-backend
    # and --umap-backend overrides: if either is pinned to 'sklearn', the GPU
    # never runs regardless of hardware.
    if args.svd_backend == 'sklearn' and args.umap_backend == 'sklearn':
        print("GPU STATUS: NOT IN USE - both backends pinned to sklearn (CPU). "
              "Pass --svd-backend auto / --umap-backend auto to allow GPU.")
    else:
        log_gpu_status()

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

    # Ground-truth summary: reports the backend that actually ran each stage,
    # populated by _mark_exec() at the exact return site of the compute call.
    svd_actual = _EXEC['svd'] or 'UNKNOWN'
    umap_actual = _EXEC['umap'] or 'UNKNOWN'
    bar = '=' * 64
    print('\n' + bar)
    print(f'COMPUTATION SUMMARY: SVD={svd_actual}, UMAP={umap_actual}')
    print(bar)

    if args.store_models and args.umap_backend == 'parametric-umap':
        umap_model.save(args.output_dir)


if __name__ == '__main__':
    main()
