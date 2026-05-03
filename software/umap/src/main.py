#!/usr/bin/env python3
"""
kmer_umap.py: Command-line tool to convert amino acid sequences from a TSV file into k-mer count
vectors, perform truncated SVD, and output UMAP embeddings.

Usage:
    python kmer_umap.py \
        -i input.tsv -c sequence_column \
        -u umap.csv \
        [--umap-components 2] \
        [--umap-neighbors 15] [--umap-min-dist 0.5] \
        [--k-mer-size 3] \
        [--max-sequences 200000] \
        [--svd-backend auto|cuml|sklearn] \
        [--umap-backend auto|cuml|sklearn] \
        [--output-dir .]

Inputs:
  - A TSV file (`-i`/`--input`) with at least one column of amino acid sequences.
  - Specify the sequence column starting string with `-c`/`--seq-col-start` (default: "sequence").

Outputs:
  - A TSV file (`-u`/`--umap-output`) containing the UMAP embeddings for each sequence.
    Columns will be named UMAP1, UMAP2, etc. Only sequences with invalid characters receive NA.

Options:
  --umap-components Number of UMAP dimensions (default: 2)
  --umap-neighbors  UMAP n_neighbors (default: 15)
  --umap-min-dist   UMAP min_dist (default: 0.5)
  --k-mer-size      Size of k-mers (default: 3 for aminoacid, 6 for nucleotide)
  --max-sequences   Fit-sample size (default: 200,000). SVD and UMAP are fitted on this many
                    randomly selected sequences; all sequences are then transformed through the
                    fitted models. Set 0 to fit on all sequences.
  --n-jobs          Parallel workers for k-mer counting (-1 = all CPUs, default: -1)
  --svd-backend     'auto' (default), 'cuml', or 'sklearn'
  --umap-backend    'auto' (default), 'cuml', or 'sklearn'
  --output-dir      Directory for output files (default: current directory)

Two-phase pipeline (internal logic):
  Phase 1 — fit models on a sample:
  - If total sequences > max_sequences (default 200,000): randomly sample max_sequences sequences.
  - Build k-mer matrix, fit TruncatedSVD, fit UMAP on this sample.
  - UMAP is fitted on min(sample_size, 100,000) sequences (sub-sample for UMAP fitting only).

  Phase 2 — transform all sequences:
  - All valid sequences are processed in chunks of TRANSFORM_CHUNK_SIZE.
  - Each chunk: k-mer count → svd.transform() → umap.transform().
  - Every valid sequence receives coordinates. Only invalid-character sequences get NA.
"""

import warnings
# Suppress UMAP spectral initialization warnings that are noisy but harmless
warnings.filterwarnings("ignore", message="Spectral initialisation failed!")
warnings.filterwarnings("ignore", message="Falling back to random initialisation!")
# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
# Suppress UMAP n_jobs override warnings
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to .* by setting random_state")

import argparse
import itertools
import numpy as np
import polars as pl
import sys
import os
from scipy import sparse
import time

# Maximum number of sequences used to fit the UMAP model (CPU path).
# When the fit sample exceeds this, a random subset is used for UMAP fitting
# and all sequences in the fit sample are transformed.
FIXED_SAMPLE_SIZE_LARGE_DATA = 100000

# Chunk size for Phase 2 transform (k-mer → svd.transform → umap.transform).
# Chosen to keep per-chunk memory bounded while minimising Python loop overhead.
TRANSFORM_CHUNK_SIZE = 50000


class _SVDTransformer:
    """Minimal wrapper that provides a .transform() interface for both CPU and GPU SVD paths.

    CPU path: wrap a fitted sklearn TruncatedSVD model directly.
    GPU path: store VT components as a numpy array; transform is X @ components.T.
    """
    def __init__(self, components):
        # components shape: (n_components, n_features)
        self.components_ = components

    def transform(self, X):
        return X @ self.components_.T


def _process_sequence_chunk(args):
    """
    Worker function for parallel k-mer counting.

    Args:
        args: Tuple of (sequences_chunk, start_idx, k, kmer_to_index)

    Returns:
        Tuple of (rows, cols) numpy int32 arrays for COO matrix construction
    """
    sequences_chunk, start_idx, k, kmer_to_index = args

    rows = []
    cols = []

    for local_idx, seq in enumerate(sequences_chunk):
        global_idx = start_idx + local_idx
        seq_upper = str(seq).upper()

        # Extract all k-mers from this sequence
        for pos in range(len(seq_upper) - k + 1):
            kmer = seq_upper[pos:pos + k]
            kmer_idx = kmer_to_index.get(kmer)
            if kmer_idx is not None:
                rows.append(global_idx)
                cols.append(kmer_idx)

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


def kmer_count_vectors(sequences, k=3, alphabet='aminoacid', n_jobs=-1, verbose=True):
    """
    Convert sequences to k-mer count vectors using parallel processing.

    Args:
        sequences (list): List of sequences (already uppercase)
        k (int): Size of k-mers to count
        alphabet (str): Sequence alphabet type ('aminoacid' or 'nucleotide')
        n_jobs (int): Number of parallel workers. -1 uses all available CPUs (default: -1)
        verbose (bool): Print progress messages (default: True). Set False for chunk transforms.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix of k-mer counts (CSR format)
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import os

    if verbose:
        print(f"Generating {k}-mer count vectors for {alphabet} sequences...")

    # Define alphabet characters based on sequence type
    if alphabet == 'aminoacid':
        alphabet_chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', '*', '_']
    else:  # nucleotide
        alphabet_chars = ['A', 'C', 'G', 'T', 'N']

    # Generate all possible k-mers and create lookup dictionary
    all_kmers = [''.join(p) for p in itertools.product(alphabet_chars, repeat=k)]
    kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}

    num_seqs = len(sequences)
    num_kmers = len(all_kmers)

    # Use spawn context to avoid fork issues on macOS and with GPU libraries
    mp_context = get_context("spawn")

    # Determine number of workers
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    # For small datasets, use single process to avoid overhead
    if num_seqs < 5000:
        n_jobs = 1

    if n_jobs == 1:
        # Single-threaded processing (no multiprocessing overhead)
        if verbose:
            print(f"Processing {num_seqs} sequences (single-threaded)...")
        rows_arr, cols_arr = _process_sequence_chunk((sequences, 0, k, kmer_to_index))
    else:
        # Split sequences into chunks for parallel processing
        # Use 4 chunks per worker for better load balancing
        chunk_size = max(1000, num_seqs // (n_jobs * 4))
        chunks = []

        for i in range(0, num_seqs, chunk_size):
            chunk_end = min(i + chunk_size, num_seqs)
            chunks.append((sequences[i:chunk_end], i, k, kmer_to_index))

        if verbose:
            print(f"Processing {num_seqs} sequences using {n_jobs} parallel workers ({len(chunks)} chunks)...")

        # Process chunks in parallel using ProcessPoolExecutor with spawn context
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_context) as executor:
            results = list(executor.map(_process_sequence_chunk, chunks))

        # Merge numpy arrays from workers — avoids Python object overhead of list.extend()
        if verbose:
            print("Merging results from parallel workers...")
        rows_parts, cols_parts = [], []
        for chunk_rows, chunk_cols in results:
            rows_parts.append(chunk_rows)
            cols_parts.append(chunk_cols)

        rows_arr = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.int32)
        cols_arr = np.concatenate(cols_parts) if cols_parts else np.array([], dtype=np.int32)

    # Build sparse matrix from (row, col) index arrays.
    # scipy's tocsr() sums duplicate (row, col) entries natively in C —
    # no Python Counter needed, and memory stays in compact int32 arrays.
    if verbose:
        print(f"Building sparse matrix from {len(rows_arr)} k-mer occurrences...")
    data_arr = np.ones(len(rows_arr), dtype=np.int32)
    matrix = sparse.coo_matrix(
        (data_arr, (rows_arr, cols_arr)),
        shape=(num_seqs, num_kmers),
        dtype=np.int32
    ).tocsr()

    if verbose:
        print(f"Sparse matrix created: {matrix.shape}, {matrix.nnz} non-zero entries")

    return matrix


def create_svd_model(backend, n_components, random_state=42):
    """Create TruncatedSVD model with specified parameters."""
    if backend == 'cuml' or (backend == 'auto'):
        try:
            import cuml
            from cuml.decomposition import TruncatedSVD as cuML_TruncatedSVD
            print("Using GPU-accelerated TruncatedSVD (RAPIDS cuML)...")
            return cuML_TruncatedSVD(n_components=n_components, random_state=random_state), 'gpu'
        except (ImportError, Exception) as e:
            if backend == 'cuml':
                print(f"Error: RAPIDS cuML not available or CUDA error: {e}")
                raise
            else:
                print(f"RAPIDS cuML not available, falling back to CPU-based TruncatedSVD: {e}")

    # Default to CPU-based TruncatedSVD
    from sklearn.decomposition import TruncatedSVD
    print("Using CPU-based TruncatedSVD (scikit-learn)...")
    return TruncatedSVD(n_components=n_components, random_state=random_state), 'cpu'


def create_umap_model(backend, components, neighbors, min_dist):
    """Create UMAP model with specified parameters."""
    common_params = {
        'n_components': components,
        'n_neighbors': neighbors,
        'min_dist': min_dist,
        'random_state': 42
    }

    if backend == 'cuml' or (backend == 'auto'):
        try:
            import cuml
            import cuml.manifold.umap as cuml_umap
            print("Using GPU-accelerated UMAP (RAPIDS cuML)...\n")
            return cuml_umap.UMAP(**common_params, init='spectral', n_epochs=2000), 'gpu'
        except (ImportError, Exception) as e:
            print(f"RAPIDS cuML not available or CUDA error: {e}")
    if backend == 'parametric-umap':
        from umap.parametric_umap import ParametricUMAP
        return ParametricUMAP(n_components=common_params['n_components']), 'gpu'

    # Default to CPU-based UMAP
    import umap as umap_learn
    print("Using CPU-based UMAP (umap-learn)...\n")
    return umap_learn.UMAP(n_jobs=-1, **common_params), 'cpu'


# This is a hack to make the print statements flush to the console immediately
# https://stackoverflow.com/questions/107705/disable-output-buffering
_orig_print = print


def print(*args, **kwargs):
    _orig_print(*args, flush=True, **kwargs)


def create_empty_umap_output(key_col_name, umap_components, output_path):
    """Create empty UMAP output file with proper headers."""
    schema = {key_col_name: pl.String}
    for i in range(umap_components):
        schema[f'UMAP{i+1}'] = pl.Float64
    pl.DataFrame(schema=schema).write_csv(output_path, separator='\t')


def create_empty_skipped_summary(output_dir, reason):
    """Create skipped clonotypes summary file."""
    skipped_summary_path = os.path.join(output_dir, 'skipped_clonotypes_summary.txt')
    with open(skipped_summary_path, 'w') as f:
        f.write(f"{reason}\n")


def estimate_sparse_memory_gb(matrix):
    """Estimate GPU memory required for sparse matrix in GB."""
    nnz = matrix.nnz
    sparse_memory_bytes = (nnz * 4 + nnz * 4 + (matrix.shape[0] + 1) * 4)
    return sparse_memory_bytes / (1024**3)


def estimate_dense_memory_gb(matrix):
    """Estimate GPU memory required for dense matrix in GB."""
    return (matrix.shape[0] * matrix.shape[1] * 4) / (1024**3)


def get_gpu_memory_info():
    """Get GPU memory information in GB. Returns (free_gb, total_gb)."""
    import cupy as cp
    device = cp.cuda.Device()
    device_memory = device.mem_info
    free_gb = device_memory[0] / (1024**3)
    total_gb = device_memory[1] / (1024**3)
    return free_gb, total_gb


def compute_explained_variance_cupy(singular_values, matrix_gpu, n_samples):
    """Compute explained variance ratio from CuPy SVD singular values."""
    import cupy as cp
    explained_variance = (singular_values ** 2) / (n_samples - 1)
    total_sum_squares = float(matrix_gpu.power(2).sum())
    total_variance = total_sum_squares / (n_samples - 1)
    if total_variance == 0:
        return cp.zeros_like(explained_variance)
    return cp.asnumpy(explained_variance / total_variance)


def run_cupy_sparse_svd(matrix_gpu, n_components, random_seed=42):
    """Run CuPy sparse SVD and return results in standard order."""
    import cupy as cp
    from cupyx.scipy.sparse.linalg import svds as cupy_svds
    cp.random.seed(random_seed)
    u, s, vt = cupy_svds(matrix_gpu, k=n_components)
    return u[:, ::-1], s[::-1], vt[::-1, :]


def fallback_to_cpu_svd(matrix, n_components):
    """Fallback to CPU-based SVD. Returns (svd_model, explained_variance_ratio)."""
    print("Falling back to CPU-based SVD...")
    svd_model, _ = create_svd_model('sklearn', n_components, random_state=42)
    svd_model.fit(matrix)
    return svd_model, svd_model.explained_variance_ratio_


def compute_svd_embedding(matrix, svd_backend='auto', target_variance=0.95, max_components=500):
    """
    Fit SVD on matrix and return embedding + a transformer for future .transform() calls.

    Tries GPU-accelerated sparse SVD first (memory efficient), falls back to CPU if needed.
    Automatically determines optimal number of components based on explained variance.

    Returns:
        tuple: (svd_embed, svd_transformer, n_components, explained_var_sum)
            - svd_embed: numpy array of reduced dimensions for the input matrix
            - svd_transformer: _SVDTransformer with .transform() for new data
            - n_components: number of components used
            - explained_var_sum: total explained variance
    """
    SPARSE_MEMORY_MULTIPLIER = 3.0
    MEMORY_BUFFER_GB = 2.0

    n_components_max = min(matrix.shape[0] - 1, matrix.shape[1], max_components)
    print(f"Computing SVD with up to {n_components_max} components...")

    matrix_gpu = None
    use_cupy_sparse_svd = False
    svd_u, svd_s, svd_vt = None, None, None
    explained_variance_ratio = None
    svd_cpu_model = None

    try:
        _, initial_backend = create_svd_model(svd_backend, n_components_max, random_state=42)
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
                    svd_s, matrix_gpu, matrix.shape[0]
                )
                use_cupy_sparse_svd = True
                print("GPU sparse SVD completed successfully.")
            else:
                print(f"Warning: Insufficient GPU memory for sparse operations.")
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
    n_components_optimal = (np.argmax(cumulative_explained_variance >= target_variance) + 1
                            if np.any(cumulative_explained_variance >= target_variance)
                            else n_components_max)
    print(f"Number of components explaining {target_variance*100:.0f}% variance: {n_components_optimal}")

    if use_cupy_sparse_svd and matrix_gpu is not None:
        try:
            import cupy as cp
            if n_components_optimal < n_components_max:
                print(f"Recomputing GPU sparse SVD with {n_components_optimal} components...")
                svd_u, svd_s, svd_vt = run_cupy_sparse_svd(matrix_gpu, n_components_optimal)
            else:
                print(f"Reusing GPU sparse SVD results with {n_components_max} components.")
            svd_embed = cp.asnumpy(svd_u @ cp.diag(svd_s))
            explained_variance_ratio_final = compute_explained_variance_cupy(
                svd_s, matrix_gpu, matrix.shape[0]
            )
            explained_var_sum = sum(explained_variance_ratio_final)
            # Store VT as numpy for CPU-side transform calls
            svd_transformer = _SVDTransformer(cp.asnumpy(svd_vt[:n_components_optimal]))
            print("GPU sparse SVD embedding computed successfully.")
        except (MemoryError, Exception) as e:
            print(f"Warning: GPU SVD transform failed: {e}")
            svd_model, _ = create_svd_model('sklearn', n_components_optimal, random_state=42)
            svd_embed = svd_model.fit_transform(matrix)
            explained_var_sum = sum(svd_model.explained_variance_ratio_)
            svd_transformer = _SVDTransformer(svd_model.components_)
    else:
        # svd_cpu_model was already fit with n_components_max components in
        # fallback_to_cpu_svd. The rank-k truncation of a rank-m SVD (m >= k) is
        # the first k components, so slicing avoids recomputing the SVD. For
        # sklearn's randomized solver this is not bitwise identical to a fresh
        # k-component fit but yields equally good (or better, due to higher
        # oversampling) top components.
        components = svd_cpu_model.components_[:n_components_optimal]
        svd_transformer = _SVDTransformer(components)
        svd_embed = svd_transformer.transform(matrix)
        explained_var_sum = float(
            svd_cpu_model.explained_variance_ratio_[:n_components_optimal].sum()
        )

    print(f"Explained variance by {n_components_optimal} components: {explained_var_sum:.3f}")

    return svd_embed, svd_transformer, n_components_optimal, explained_var_sum


def main():
    parser = argparse.ArgumentParser(
        description='Compute UMAP embeddings from amino acid sequences via k-mer counts and SVD.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input TSV file with sequence column')
    parser.add_argument('-c', '--seq-col-start', default='sequence',
                        help='Starting string of the column containing sequences (default: "sequence")')
    parser.add_argument('-u', '--umap-output', required=True,
                        help='Output TSV file for UMAP embeddings')
    parser.add_argument('--alphabet', choices=['aminoacid', 'nucleotide'], default='aminoacid',
                        help='Sequence alphabet type (default: aminoacid)')
    parser.add_argument('--umap-components', type=int, default=2,
                        help='Number of UMAP dimensions (default: 2)')
    parser.add_argument('--umap-neighbors', type=int, default=15,
                        help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.5,
                        help='UMAP min_dist (default: 0.5)')
    parser.add_argument('--k-mer-size', type=int, default=None,
                        help='Size of k-mers (default: 3 for aminoacid, 6 for nucleotide)')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--svd-backend', type=str, default='auto',
                        choices=['auto', 'cuml', 'sklearn'],
                        help='Choose TruncatedSVD implementation:\n'
                             '  "auto" (default): Tries cuML first, falls back to scikit-learn.\n'
                             '  "cuml": Forces RAPIDS cuML (requires CUDA GPU).\n'
                             '  "sklearn": Forces scikit-learn (CPU-based).')
    parser.add_argument('--umap-backend', type=str, default='auto',
                        choices=['auto', 'cuml', 'sklearn', 'parametric-umap'],
                        help='Choose UMAP implementation:\n'
                             '  "auto" (default): Tries cuml.UMAP first, falls back to umap-learn.\n'
                             '  "cuml": Forces RAPIDS cuml.UMAP (requires CUDA GPU).\n'
                             '  "sklearn": Forces umap-learn (CPU-based).')
    parser.add_argument('--store-models', action='store_true',
                        help='Store SVD and UMAP models to disk (default: False)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel workers for k-mer counting.\n'
                             '  -1 (default): Use all available CPU cores.\n'
                             '  1: Single-threaded.\n'
                             '  N: Use N workers.')
    parser.add_argument('--max-sequences', type=int, default=200000,
                        help='Fit-sample size. SVD and UMAP models are fitted on this many randomly\n'
                             'selected sequences; ALL valid sequences are then transformed through\n'
                             'the fitted models and receive coordinates in the output.\n'
                             '  0: Fit on all sequences (no sub-sampling).\n'
                             '  N: Fit on N sequences (default: 200,000).')

    args = parser.parse_args()

    if args.k_mer_size is None:
        args.k_mer_size = 3 if args.alphabet == 'aminoacid' else 6

    os.makedirs(args.output_dir, exist_ok=True)

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

    if args.umap_components < 1:
        print("Error: Number of UMAP components must be at least 1")
        sys.exit(1)
    if args.umap_neighbors < 1:
        print("Error: UMAP neighbors must be at least 1")
        sys.exit(1)
    if not (0 <= args.umap_min_dist <= 1):
        print("Error: UMAP min_dist must be between 0 and 1")
        sys.exit(1)
    if args.k_mer_size < 1:
        print("Error: k-mer size must be at least 1")
        sys.exit(1)

    output_path = os.path.join(args.output_dir, args.umap_output)

    # --- Load input with polars (multithreaded Rust CSV parser) ---
    start_time_load = time.time()
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
        print("Error: Input file is empty")
        first_col = df.columns[0] if df.columns else 'clonotypeKey'
        create_empty_umap_output(first_col, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir, "UMAP analysis skipped due to empty input file.")
        sys.exit(0)

    # Find sequence columns
    seq_col_list = sorted([c for c in df.columns if c.startswith(args.seq_col_start)])
    if not seq_col_list:
        print(f"Error: Columns starting with '{args.seq_col_start}' not found. "
              f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    # Ensure there is a row-identity column for the output join.
    # The block workflow always provides 'clonotypeKey'; for standalone use we add a row index.
    KEY_COL = 'clonotypeKey'
    if KEY_COL not in df.columns:
        df = df.with_row_index(KEY_COL).with_columns(pl.col(KEY_COL).cast(pl.String))

    SEQ_COL = 'combined_sequence'

    # Combine selected sequence columns (concatenate, treating nulls as empty string)
    df = df.with_columns(
        pl.concat_str([pl.col(c).fill_null('') for c in seq_col_list]).alias(SEQ_COL)
    )

    # Drop rows where the combined sequence is empty or only underscores
    initial_count = len(df)
    df = df.filter(pl.col(SEQ_COL).str.strip_chars('_').str.len_chars() > 0)
    if len(df) < initial_count:
        print(f"Warning: Removed {initial_count - len(df)} empty or whitespace-only sequences.")

    if df.is_empty():
        print("Error: Input file is empty")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir, "UMAP analysis skipped due to empty input file.")
        sys.exit(0)

    # --- Invalid character filtering (vectorized polars regex — no Python loop) ---
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

    num_total_sequences = len(df_valid)
    if num_total_sequences == 0:
        print('Error: No valid sequences found after filtering. Exiting.')
        sys.exit(1)

    end_time_load = time.time()
    print(f"Input loading and preprocessing completed in {end_time_load - start_time_load:.2f} seconds.\n")

    # Detect UMAP backend early — determines CPU vs GPU execution path
    if args.umap_backend in ['cuml', 'auto', 'sklearn', 'parametric-umap']:
        umap_model, run_type = create_umap_model(args.umap_backend, args.umap_components,
                                                 args.umap_neighbors, args.umap_min_dist)
    else:
        print(f"Error: Unknown UMAP backend '{args.umap_backend}'. Exiting.")
        sys.exit(1)

    # Check minimum sequences for UMAP
    min_required_sequences = max(args.umap_neighbors + 1, 4)
    if num_total_sequences < min_required_sequences:
        print("Warning: Not enough sequences for UMAP analysis.")
        print(f"Required: {min_required_sequences}, Available: {num_total_sequences}.")
        create_empty_umap_output(KEY_COL, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir, "UMAP analysis skipped due to insufficient sequences.")
        sys.exit(0)

    sequences_all = df_valid[SEQ_COL].str.to_uppercase().to_list()
    keys_all = df_valid[KEY_COL].to_list()
    n_all = len(sequences_all)

    if run_type == 'gpu':
        # -----------------------------------------------------------------------
        # GPU PATH — process all sequences at once (legacy behavior)
        # k-mer → SVD fit_transform → UMAP fit_transform, all on full dataset
        # -----------------------------------------------------------------------

        # --- K-mer counting on all sequences ---
        start_time_kmer = time.time()
        matrix = kmer_count_vectors(sequences_all, k=args.k_mer_size, alphabet=args.alphabet,
                                    n_jobs=args.n_jobs, verbose=True)
        end_time_kmer = time.time()
        print(f"K-mer counting completed in {end_time_kmer - start_time_kmer:.2f} seconds.\n")

        # --- Truncated SVD on all sequences ---
        start_time_svd = time.time()
        print("Running Truncated SVD...")
        svd_embed, _, n_components_used, explained_var_sum = compute_svd_embedding(
            matrix=matrix,
            svd_backend=args.svd_backend,
            target_variance=0.95,
            max_components=500
        )
        end_time_svd = time.time()
        print(f"Truncated SVD completed in {end_time_svd - start_time_svd:.2f} seconds.\n")

        # --- UMAP fit_transform on all sequences ---
        start_time_umap = time.time()
        print("Running UMAP dimensionality reduction (fitting and transforming on GPU)...")
        try:
            umap_embed_all = umap_model.fit_transform(svd_embed)
            print(f"UMAP completed in {time.time() - start_time_umap:.2f} seconds.\n")
        except Exception as e:
            print(f"Warning: GPU UMAP failed - {e}. Falling back to CPU two-phase approach...")
            umap_model, run_type = create_umap_model('sklearn', args.umap_components,
                                                     args.umap_neighbors, args.umap_min_dist)
            # Fall through to CPU path below using already-computed svd_embed
            np.random.seed(42)
            if n_all > FIXED_SAMPLE_SIZE_LARGE_DATA:
                sample_indices = np.random.choice(n_all, size=FIXED_SAMPLE_SIZE_LARGE_DATA, replace=False)
                umap_model.fit(svd_embed[sample_indices])
            else:
                umap_model.fit(svd_embed)
            umap_embed_all = umap_model.transform(svd_embed)
            print(f"UMAP (CPU fallback) completed in {time.time() - start_time_umap:.2f} seconds.\n")

    else:
        # -----------------------------------------------------------------------
        # CPU PATH — two-phase: fit models on a sample, transform all in chunks
        # -----------------------------------------------------------------------

        # --- Phase 1: fit SVD and UMAP on a sample ---
        fit_sample_size = args.max_sequences
        if fit_sample_size > 0 and num_total_sequences > fit_sample_size:
            print(f"Fit-sample: {num_total_sequences} valid sequences exceed "
                  f"--max-sequences={fit_sample_size}.")
            print(f"Randomly sampling {fit_sample_size} sequences to fit SVD and UMAP models...")
            df_fit = df_valid.sample(n=fit_sample_size, seed=42)
            print(f"Fit sample: {len(df_fit)} sequences selected.\n")
        else:
            df_fit = df_valid

        sequences_fit = df_fit[SEQ_COL].str.to_uppercase().to_list()
        num_fit_sequences = len(sequences_fit)

        start_time_kmer = time.time()
        matrix_fit = kmer_count_vectors(sequences_fit, k=args.k_mer_size, alphabet=args.alphabet,
                                        n_jobs=args.n_jobs, verbose=True)
        end_time_kmer = time.time()
        print(f"K-mer counting completed in {end_time_kmer - start_time_kmer:.2f} seconds.\n")

        start_time_svd = time.time()
        print("Running Truncated SVD...")
        svd_embed_fit, svd_transformer, n_components_used, explained_var_sum = compute_svd_embedding(
            matrix=matrix_fit,
            svd_backend=args.svd_backend,
            target_variance=0.95,
            max_components=500
        )
        end_time_svd = time.time()
        print(f"Truncated SVD completed in {end_time_svd - start_time_svd:.2f} seconds.\n")

        if num_fit_sequences <= FIXED_SAMPLE_SIZE_LARGE_DATA:
            print(f"Fit sample ({num_fit_sequences}) <= {FIXED_SAMPLE_SIZE_LARGE_DATA}. "
                  f"No sub-sampling — UMAP fits on all fit-sample sequences.")
            sampled_data_for_fit = svd_embed_fit
        else:
            print(f"Fit sample ({num_fit_sequences}) > {FIXED_SAMPLE_SIZE_LARGE_DATA}. "
                  f"Sampling {FIXED_SAMPLE_SIZE_LARGE_DATA} sequences for UMAP fitting.")
            np.random.seed(42)
            sample_indices = np.random.choice(num_fit_sequences, size=FIXED_SAMPLE_SIZE_LARGE_DATA,
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

            chunk_matrix = kmer_count_vectors(chunk_seqs, k=args.k_mer_size,
                                              alphabet=args.alphabet, n_jobs=args.n_jobs,
                                              verbose=False)
            chunk_svd = svd_transformer.transform(chunk_matrix)
            chunk_umap = umap_model.transform(chunk_svd)
            all_coords.append(chunk_umap)

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                elapsed = time.time() - start_time_transform
                print(f"  Chunk {chunk_idx + 1}/{n_chunks} — {chunk_end}/{n_all} sequences "
                      f"({elapsed:.0f}s elapsed)")

        umap_embed_all = np.vstack(all_coords)
        print(f"Transform completed in {time.time() - start_time_transform:.2f} seconds.\n")

    # --- Build and write output ---
    start_time_save = time.time()

    coord_dict = {KEY_COL: keys_all}
    for i in range(args.umap_components):
        coord_dict[f'UMAP{i+1}'] = umap_embed_all[:, i].tolist()
    coords_df = pl.DataFrame(coord_dict)

    # Left-join onto ALL input rows: invalid-char sequences get null coordinates.
    output_df = df.select(KEY_COL).join(coords_df, on=KEY_COL, how='left')
    output_df.write_csv(output_path, separator='\t', null_value='')

    # --- Skipped clonotypes summary ---
    skipped_summary_path = os.path.join(args.output_dir, 'skipped_clonotypes_summary.txt')
    with open(skipped_summary_path, 'w') as f:
        f.write(f"Number of clonotypes skipped due to invalid amino acid sequences: {n_invalid}\n")
        f.write(f"Total clonotypes processed: {len(df)}\n")
        f.write(f"Valid clonotypes: {len(df_valid)}\n")
        f.write(f"Skipped clonotypes: {n_invalid}\n\n")
        if n_invalid > 0:
            f.write("Skipped clonotypes:\n")
            for row in df_invalid.select([KEY_COL, SEQ_COL]).iter_rows():
                f.write(f"{row[0]}\t{row[1]}\n")
    print(f"Skipped clonotypes summary saved to {skipped_summary_path}")

    end_time_save = time.time()
    print(f'UMAP embeddings saved to {output_path} in {end_time_save - start_time_save:.2f} seconds.')

    total_run_time = end_time_save - start_time_load
    print(f"\nTotal analysis completed in {total_run_time:.2f} seconds.")

    if args.store_models:
        if args.umap_backend == 'parametric-umap':
            umap_model.save(args.output_dir)


if __name__ == '__main__':
    main()
