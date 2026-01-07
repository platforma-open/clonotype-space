#!/usr/bin/env python3
"""
kmer_umap.py: Command-line tool to convert amino acid sequences from a TSV file into 6-mer count vectors,
compute pairwise Euclidean distances, perform PCA, and output only the final UMAP embeddings.

Usage: 
    python kmer_umap.py \
        -i input.tsv -c sequence_column \
        -u umap.csv \
        [--dr-components 5] [--umap-components 2] \
        [--umap-neighbors 8] [--umap-min-dist 0.05] \
        [--k-mer-size 3] \
        [--svd-backend auto|cuml|sklearn] \
        [--umap-backend auto|cuml|sklearn] \
        [--output-dir .]

Inputs:
  - A TSV file (`-i`/`--input`) with at least one column of amino acid sequences.
  - Specify the sequence column starting string with `-c`/`--seq-col-start` (default: "aaSequence").

Outputs:
  - A CSV file (`-u`/`--umap-output`) containing the UMAP embeddings for each sequence.
    Columns will be named UMAP1, UMAP2, etc.

Options:
  --dr-components   Number of dimensionality reduction components (TruncatedSVD) before UMAP (default: 5)
  --umap-components Number of UMAP dimensions (default: 2)
  --umap-neighbors  UMAP n_neighbors (default: 8)
  --umap-min-dist   UMAP min_dist (default: 0.05)
  --k-mer-size      Size of k-mers to use for sequence analysis (default: 3 for amino acids)
  --n-jobs          Number of parallel workers for k-mer counting: -1 (default, use all CPUs),
                    1 (single-threaded), or N (use N workers). Parallelization provides 4-8x speedup.
  --svd-backend     Choose TruncatedSVD implementation: 'auto' (default, tries cuml then sklearn),
                    'cuml' (RAPIDS cuML TruncatedSVD, requires GPU), 'sklearn' (scikit-learn, CPU-based).
  --umap-backend    Choose UMAP implementation: 'auto' (default, tries cuml then sklearn),
                    'cuml' (RAPIDS cuml.UMAP, requires GPU), 'sklearn' (umap-learn, CPU-based).
  --output-dir      Directory to save output files (default: current directory)

Sampling Strategy (Internal Logic):
  - If total sequences < 50,000: No sampling.
  - If total sequences between 50,000 and 200,000 (inclusive of 50k, exclusive of 200k): Sample 50% of total sequences for UMAP fitting.
  - If total sequences >= 200,000: Sample 100,000 sequences for UMAP fitting.
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
import pandas as pd
import sys
import os
from scipy import sparse
import time
# import pickle

# Constants for sampling thresholds
SAMPLING_THRESHOLD_1 = 50000
SAMPLING_THRESHOLD_2 = 200000
FIXED_SAMPLE_SIZE_LARGE_DATA = 100000

def _process_sequence_chunk(args):
    """
    Worker function for parallel k-mer counting.
    
    Args:
        args: Tuple of (sequences_chunk, start_idx, k, kmer_to_index)
    
    Returns:
        Tuple of (rows, cols) lists for COO matrix construction
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
    
    return rows, cols

def kmer_count_vectors(sequences, k=3, alphabet='aminoacid', n_jobs=-1):
    """
    Convert sequences to k-mer count vectors using parallel processing.
    
    Args:
        sequences (list): List of sequences
        k (int): Size of k-mers to count
        alphabet (str): Sequence alphabet type ('aminoacid' or 'nucleotide')
        n_jobs (int): Number of parallel workers. -1 uses all available CPUs (default: -1)
        
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix of k-mer counts (CSR format)
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from collections import Counter
    import os
    
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
        print(f"Processing {num_seqs} sequences (single-threaded)...")
        rows, cols = _process_sequence_chunk((sequences, 0, k, kmer_to_index))
    else:
        # Split sequences into chunks for parallel processing
        # Use 4 chunks per worker for better load balancing
        chunk_size = max(1000, num_seqs // (n_jobs * 4))
        chunks = []
        
        for i in range(0, num_seqs, chunk_size):
            chunk_end = min(i + chunk_size, num_seqs)
            chunks.append((sequences[i:chunk_end], i, k, kmer_to_index))
        
        print(f"Processing {num_seqs} sequences using {n_jobs} parallel workers ({len(chunks)} chunks)...")
        
        # Process chunks in parallel using ProcessPoolExecutor with spawn context
        # Context manager ensures proper shutdown and cleanup
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_context) as executor:
            results = list(executor.map(_process_sequence_chunk, chunks))
        
        # Merge results from all workers
        print("Merging results from parallel workers...")
        rows = []
        cols = []
        for chunk_rows, chunk_cols in results:
            rows.extend(chunk_rows)
            cols.extend(chunk_cols)
    
    # Build sparse matrix from collected (row, col) pairs
    # Use Counter to aggregate k-mer counts
    print(f"Building sparse matrix from {len(rows)} k-mer occurrences...")
    data_dict = Counter(zip(rows, cols))
    
    rows_final = np.array([r for r, c in data_dict.keys()], dtype=np.int32)
    cols_final = np.array([c for r, c in data_dict.keys()], dtype=np.int32)
    data_final = np.array(list(data_dict.values()), dtype=np.int32)
    
    # Create COO matrix and convert to CSR for efficient operations
    matrix = sparse.coo_matrix(
        (data_final, (rows_final, cols_final)), 
        shape=(num_seqs, num_kmers), 
        dtype=np.int32
    )
    
    print(f"Sparse matrix created: {matrix.shape}, {matrix.nnz} non-zero entries")
    
    return matrix.tocsr()

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
            return cuml_umap.UMAP(**common_params, init = 'spectral', n_epochs = 2000), 'gpu'
          
        except (ImportError, Exception) as e:
            print(f"RAPIDS cuML not available or CUDA error: {e}")
    if backend == 'parametric-umap':
        from umap.parametric_umap import ParametricUMAP
        return ParametricUMAP(n_components = common_params['n_components']), 'gpu'

    # Default to CPU-based UMAP
    import umap as umap_learn
    print("Using CPU-based UMAP (umap-learn)...\n")
    return umap_learn.UMAP(n_jobs=-1, **common_params), 'cpu'

# This is a hack to make the print statements flush to the console immediately
# https://stackoverflow.com/questions/107705/disable-output-buffering
_orig_print = print

def print(*args, **kwargs):
    _orig_print(*args, flush=True, **kwargs)

def create_empty_umap_output(df_input, umap_components, output_path):
    """Create empty UMAP output file with proper headers."""
    umap_df = pd.DataFrame(
        columns=[df_input.columns[0]] + [f'UMAP{i+1}' for i in range(umap_components)]
    )
    umap_df.to_csv(output_path, index=False, sep='\t')

def create_empty_skipped_summary(output_dir, reason):
    """Create skipped clonotypes summary file."""
    skipped_summary_path = os.path.join(output_dir, 'skipped_clonotypes_summary.txt')
    with open(skipped_summary_path, 'w') as f:
        f.write(f"{reason}\n")

def estimate_sparse_memory_gb(matrix):
    """Estimate GPU memory required for sparse matrix in GB."""
    nnz = matrix.nnz
    # CSR format: data (float32) + indices (int32) + indptr (int32)
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
    """Compute explained variance ratio from CuPy SVD singular values.
    
    Matches sklearn's TruncatedSVD behavior for sparse matrices.
    For SVD of non-centered data X: X = U @ diag(S) @ V^T
    - explained_variance = S^2 / (n-1)
    - total_variance = Frobenius norm squared / (n-1) for non-centered data
    """
    import cupy as cp
    explained_variance = (singular_values ** 2) / (n_samples - 1)
    
    # For TruncatedSVD on non-centered data, sklearn uses Frobenius norm
    # Total variance = ||X||_F^2 / (n-1) where ||X||_F^2 = sum of all squared elements
    # This is because for non-centered data, the "total variance" in TruncatedSVD context
    # is actually the sum of squared entries divided by (n-1)
    
    # For sparse matrix: compute Frobenius norm squared
    total_sum_squares = float(matrix_gpu.power(2).sum())
    total_variance = total_sum_squares / (n_samples - 1)

    # Handle case where total_variance is zero to avoid division by zero
    if total_variance == 0:
        return cp.zeros_like(explained_variance)
    
    return cp.asnumpy(explained_variance / total_variance)

def run_cupy_sparse_svd(matrix_gpu, n_components, random_seed=42):
    """
    Run CuPy sparse SVD and return results in standard order.
    
    Returns:
        tuple: (u, s, vt) where singular values are in descending order
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import svds as cupy_svds
    
    # Set random seed for reproducibility
    cp.random.seed(random_seed)
    
    # Perform SVD (returns ascending order)
    u, s, vt = cupy_svds(matrix_gpu, k=n_components)
    
    # Reverse to descending order (largest singular values first)
    return u[:, ::-1], s[::-1], vt[::-1, :]

def fallback_to_cpu_svd(matrix, n_components):
    """Fallback to CPU-based SVD. Returns (svd_model, explained_variance_ratio)."""
    print("Falling back to CPU-based SVD...")
    svd_model, _ = create_svd_model('sklearn', n_components, random_state=42)
    svd_model.fit(matrix)
    return svd_model, svd_model.explained_variance_ratio_

def compute_svd_embedding(matrix, svd_backend='auto', target_variance=0.95, max_components=500):
    """
    Compute SVD dimensionality reduction with automatic backend selection.
    
    Tries GPU-accelerated sparse SVD first (memory efficient), falls back to CPU if needed.
    Automatically determines optimal number of components based on explained variance.
    
    Args:
        matrix: scipy.sparse matrix (CSR format) containing k-mer count vectors
        svd_backend: 'auto', 'gpu', or 'cpu' - which SVD implementation to use
        target_variance: target explained variance ratio (default: 0.95 for 95%)
        max_components: maximum number of components to compute (default: 500)
    
    Returns:
        tuple: (svd_embed, n_components, explained_variance_sum)
            - svd_embed: numpy array of reduced dimensions
            - n_components: number of components used
            - explained_variance_sum: total explained variance by selected components
    """
    # Memory safety margins for GPU operations
    SPARSE_MEMORY_MULTIPLIER = 3.0  # 3x sparse memory for intermediate operations
    MEMORY_BUFFER_GB = 2.0  # Additional buffer for CUDA operations
    
    # Calculate maximum feasible number of components
    n_components_max = min(matrix.shape[0] - 1, matrix.shape[1], max_components)
    print(f"Computing SVD with up to {n_components_max} components...")
    
    # State variables for SVD computation
    matrix_gpu = None
    use_cupy_sparse_svd = False
    svd_u, svd_s, svd_vt = None, None, None
    explained_variance_ratio = None
    svd_cpu_model = None
    
    # Determine which SVD backend to use
    try:
        _, initial_backend = create_svd_model(svd_backend, n_components_max, random_state=42)
    except Exception as e:
        print(f"Warning: Error detecting SVD backend: {e}")
        initial_backend = 'cpu'
    
    # Try GPU-accelerated sparse SVD if GPU backend is available
    if initial_backend == 'gpu':
        try:
            import cupy as cp
            from cupyx.scipy import sparse as cp_sparse
            
            # Get GPU memory information
            free_gb, total_gb = get_gpu_memory_info()
            print(f"GPU memory available: {free_gb:.2f} GB / {total_gb:.2f} GB")
            
            # Estimate memory requirements
            sparse_mem_gb = estimate_sparse_memory_gb(matrix)
            dense_mem_gb = estimate_dense_memory_gb(matrix)
            sparsity_pct = (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
            
            print(f"Sparse matrix memory: {sparse_mem_gb:.2f} GB "
                  f"(vs {dense_mem_gb:.2f} GB if dense)")
            print(f"Matrix sparsity: {sparsity_pct:.1f}% sparse")
            
            # Check if we have enough GPU memory for sparse operations
            required_mem_gb = sparse_mem_gb * SPARSE_MEMORY_MULTIPLIER + MEMORY_BUFFER_GB
            
            if free_gb >= required_mem_gb:
                print("Using CuPy sparse SVD (supports sparse matrices on GPU)...")
                
                # Convert to GPU sparse matrix
                matrix_gpu = cp_sparse.csr_matrix(matrix, dtype=cp.float32)
                print(f"GPU sparse matrix created: {matrix_gpu.shape}, {matrix_gpu.nnz} non-zeros")
                
                # Perform GPU sparse SVD
                print("Running GPU sparse SVD...")
                svd_u, svd_s, svd_vt = run_cupy_sparse_svd(matrix_gpu, n_components_max)
                
                # Calculate explained variance
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
        # CPU-based SVD
        svd_cpu_model, explained_variance_ratio = fallback_to_cpu_svd(matrix, n_components_max)
    
    # Find optimal number of components for target explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    n_components_optimal = (np.argmax(cumulative_explained_variance >= target_variance) + 1 
                           if np.any(cumulative_explained_variance >= target_variance) 
                           else n_components_max)
    print(f"Number of components explaining {target_variance*100:.0f}% variance: {n_components_optimal}")
    
    # Compute final SVD embedding with optimal number of components
    if use_cupy_sparse_svd and matrix_gpu is not None:
        # GPU sparse SVD path
        try:
            import cupy as cp
            
            # Recompute only if we need fewer components
            if n_components_optimal < n_components_max:
                print(f"Recomputing GPU sparse SVD with {n_components_optimal} components...")
                svd_u, svd_s, svd_vt = run_cupy_sparse_svd(matrix_gpu, n_components_optimal)
            else:
                print(f"Reusing GPU sparse SVD results with {n_components_max} components.")
            
            # Compute embedding: X_reduced = U * S
            svd_embed = cp.asnumpy(svd_u @ cp.diag(svd_s))
            
            # Calculate explained variance for selected components
            explained_variance_ratio_final = compute_explained_variance_cupy(
                svd_s, matrix_gpu, matrix.shape[0]
            )
            explained_var_sum = sum(explained_variance_ratio_final)
            
            print("GPU sparse SVD embedding computed successfully.")
            
        except (MemoryError, Exception) as e:
            print(f"Warning: GPU SVD transform failed: {e}")
            svd_model, _ = create_svd_model('sklearn', n_components_optimal, random_state=42)
            svd_embed = svd_model.fit_transform(matrix)
            explained_var_sum = sum(svd_model.explained_variance_ratio_)
    else:
        # CPU-based SVD path
        if n_components_optimal < n_components_max or svd_cpu_model is None:
            # Create new model with optimal number of components
            svd_model, _ = create_svd_model('sklearn', n_components_optimal, random_state=42)
            svd_embed = svd_model.fit_transform(matrix)
        else:
            # Reuse the full model if we're using all components
            svd_embed = svd_cpu_model.fit_transform(matrix)
            svd_model = svd_cpu_model
        
        explained_var_sum = sum(svd_model.explained_variance_ratio_)
    
    print(f"Explained variance by {n_components_optimal} components: {explained_var_sum:.3f}")
    
    return svd_embed, n_components_optimal, explained_var_sum

def main():
    parser = argparse.ArgumentParser(
        description='Compute UMAP embeddings from amino acid sequences via k-mer counts and PCA.',
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
                        help='Size of k-mers to use for sequence analysis (default: 3 for aminoacid, 6 for nucleotide)')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--svd-backend', type=str, default='auto',
                        choices=['auto', 'cuml', 'sklearn'],
                        help='Choose TruncatedSVD implementation:\n'
                             '  "auto" (default): Tries cuML TruncatedSVD first, falls back to scikit-learn.\n'
                             '  "cuml": Forces RAPIDS cuML TruncatedSVD (requires CUDA-enabled GPU and cuml installed).\n'
                             '  "sklearn": Forces scikit-learn TruncatedSVD (CPU-based, no GPU required).')
    parser.add_argument('--umap-backend', type=str, default='auto',
                        choices=['auto', 'cuml', 'sklearn', 'parametric-umap'],
                        help='Choose UMAP implementation:\n'
                             '  "auto" (default): Tries cuml.UMAP first, falls back to umap-learn.\n'
                             '  "cuml": Forces RAPIDS cuml.UMAP (requires CUDA-enabled GPU and cuml installed).\n'
                             '  "sklearn": Forces umap-learn (CPU-based, no GPU required).')
    parser.add_argument('--store-models', action='store_true',
                        help='Set to True to store SVD and UMAP models (default: False)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel workers for k-mer counting.\n'
                             '  -1 (default): Use all available CPU cores.\n'
                             '  1: Single-threaded processing (no parallelization).\n'
                             '  N: Use N parallel workers.')

    args = parser.parse_args()

    # Auto-adjust k-mer size based on alphabet if not explicitly provided
    if args.k_mer_size is None:
        args.k_mer_size = 3 if args.alphabet == 'aminoacid' else 6

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    sequence_type = "amino acid" if args.alphabet == 'aminoacid' else "nucleotide"
    print(f"Starting k-mer UMAP analysis for {sequence_type} sequences")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.umap_output}")
    print(f"Parameters: alphabet={args.alphabet}, k-mer size={args.k_mer_size}, "
          f"UMAP components={args.umap_components}, "
          f"UMAP neighbors={args.umap_neighbors}, "
          f"UMAP min_dist={args.umap_min_dist}, "
          f"SVD Backend={args.svd_backend.upper()}, "
          f"UMAP Backend={args.umap_backend.upper()}")

    # Validate input parameters
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

    # --- Start Timing: Load Input ---
    start_time_load = time.time()
    try:
        print("Loading input file...")
        df_input = pd.read_csv(args.input, sep='\t', dtype=str)
        print(f"Loaded {len(df_input)} sequences")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Before continuing, we check if input files is empty
    output_path = os.path.join(args.output_dir, args.umap_output)
    if df_input.empty:
        print("Error: Input file is empty")
        create_empty_umap_output(df_input, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir, "UMAP analysis skipped due to empty input file.")
        sys.exit(0)

    seq_col_list = sorted([c for c in df_input.columns 
                        if c.startswith(args.seq_col_start)])
    if len(seq_col_list) == 0:
        print(f"Error: Columns starting with '{args.seq_col_start}' not found in input TSV. Available columns: {', '.join(df_input.columns)}")
        sys.exit(1)

    seq_col = "combined_sequence"
    # Fill NaN values with empty string and convert to string type before concatenating
    df_input[seq_col] = df_input[seq_col_list].fillna('').astype(str).agg(''.join, axis=1)
    
    initial_seq_count = len(df_input)
    df_input = df_input[df_input[seq_col].notna() & (df_input[seq_col].str.strip('_').str.len() > 0)]
    if len(df_input) < initial_seq_count:
        print(f"Warning: Removed {initial_seq_count - len(df_input)} empty or invalid sequences.")

    sequences = df_input[seq_col].tolist()
    num_total_sequences = len(sequences)

    if not sequences:
        print('Error: No valid sequences found after filtering. Exiting.')
        sys.exit(1)
    
    # Define valid characters based on alphabet type
    if args.alphabet == 'aminoacid':
        valid_chars = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*', 'X', '_'}
    else:  # nucleotide
        valid_chars = {'A', 'C', 'G', 'T', 'N', '_'}
    
    invalid_seq_indices = []
    for i, seq in enumerate(sequences):
        clean_seq = str(seq).upper()  # Keep underscores in middle, allow them as valid characters
        if not all(char in valid_chars for char in clean_seq):
            invalid_seq_indices.append(i)
    
    if invalid_seq_indices:
        seq_type_name = "amino acid" if args.alphabet == 'aminoacid' else "nucleotide"
        print(f"Warning: Found {len(invalid_seq_indices)} sequences with invalid {seq_type_name} characters. These sequences will be skipped.")
        print(f"Example invalid sequence (first 5): {[sequences[i] for i in invalid_seq_indices[:5]]}")
        
        # Create list of valid sequences and their indices
        valid_sequences = []
        valid_indices = []
        for i, seq in enumerate(sequences):
            if i not in invalid_seq_indices:
                valid_sequences.append(seq)
                valid_indices.append(i)
        
        # Update sequences and count
        sequences = valid_sequences
        num_total_sequences = len(sequences)
        
        if num_total_sequences == 0:
            print('Error: No valid sequences found after filtering. Exiting.')
            sys.exit(1)
    else:
        valid_indices = list(range(len(sequences)))

    end_time_load = time.time()
    print(f"Input loading and preprocessing completed in {end_time_load - start_time_load:.2f} seconds.\n")

    # --- UMAP Backend Selection ---
    umap_model = None
    # Determine which UMAP backend to use
    if args.umap_backend in ['cuml', 'auto', 'sklearn', 'parametric-umap']:
        umap_model, run_type = create_umap_model(args.umap_backend, args.umap_components, 
                                    args.umap_neighbors, args.umap_min_dist)
    else:
        print(f"Error: Unknown UMAP backend '{args.umap_backend}'. Exiting.")
        sys.exit(1)

    # --- Start Timing: K-mer Counting ---
    start_time_kmer = time.time()
    matrix = kmer_count_vectors(sequences, k=args.k_mer_size, alphabet=args.alphabet, n_jobs=args.n_jobs)
    end_time_kmer = time.time()
    print(f"K-mer counting completed in {end_time_kmer - start_time_kmer:.2f} seconds.\n")
    
    # --- Start Timing: Truncated SVD ---
    start_time_svd = time.time()
    print("Running Truncated SVD...")
    
    # Compute SVD dimensionality reduction
    svd_embed, n_components_used, explained_var_sum = compute_svd_embedding(
        matrix=matrix,
        svd_backend=args.svd_backend,
        target_variance=0.95,
        max_components=500
    )
    
    end_time_svd = time.time()
    print(f"Truncated SVD completed in {end_time_svd - start_time_svd:.2f} seconds.\n")

    # -- Subsample dataset if working with CPU --
    if (run_type == 'gpu'):
        print(f"Total sequences ({num_total_sequences}). No sampling used for UMAP fitting on GPU.")
    elif num_total_sequences < SAMPLING_THRESHOLD_1:
        print(f"Total sequences ({num_total_sequences}) < {SAMPLING_THRESHOLD_1}. No sampling for UMAP fitting.")
        sampled_data_for_fit = svd_embed # Default to full data
    else:
        if num_total_sequences < SAMPLING_THRESHOLD_2:
            actual_sample_size = int(num_total_sequences * 0.50)
            print(f"Total sequences ({num_total_sequences}) between {SAMPLING_THRESHOLD_1} and {SAMPLING_THRESHOLD_2}. "
                f"Sampling {actual_sample_size} sequences (50%) for UMAP fitting.")
        else:
            # num_total_sequences >= 200k
            actual_sample_size = FIXED_SAMPLE_SIZE_LARGE_DATA
            print(f"Total sequences ({num_total_sequences}) >= {SAMPLING_THRESHOLD_2}. "
                f"Sampling fixed {actual_sample_size} sequences for UMAP fitting.")
        np.random.seed(42)
        sample_indices = np.random.choice(num_total_sequences, size=actual_sample_size, replace=False)
        sampled_data_for_fit = svd_embed[sample_indices]

    # --- UMAP computation and Sampling Logic ---
    # Check if we have enough sequences for UMAP
    # UMAP requires at least n_neighbors + 1 sequences; also n_neighbors must be > 1 -> minimum 3 sequences
    # Also enough data for spectral layout is needed, you need at least 2 more samples than
    # the number of dimensions (2 by default) -> at least 4 sequences or n_neighbors + 1
    # https://github.com/lmcinnes/umap/issues/201
    min_required_sequences = max(args.umap_neighbors + 1, 4)
    if num_total_sequences < min_required_sequences:
        print("Warning: Not enough sequences for UMAP analysis.")
        print("UMAP requires at least n_neighbors + 1 sequences.")
        print(f"Required: {min_required_sequences}, Available: {num_total_sequences}.")
        print("If possible, consider reducing the number of neighbors.")
        
        create_empty_umap_output(df_input, args.umap_components, output_path)
        create_empty_skipped_summary(args.output_dir, "UMAP analysis skipped due to insufficient sequences.")
        
        sys.exit(0)

    # --- Start Timing: UMAP Fitting ---
    start_time_umap_fit = time.time()

    if run_type == 'cpu':
        print("Running UMAP dimensionality reduction (fitting model)...")
        umap_model.fit(sampled_data_for_fit)
        end_time_umap_fit = time.time()
        print(f"UMAP model fitting completed in {end_time_umap_fit - start_time_umap_fit:.2f} seconds.\n")

        # --- Start Timing: UMAP Transforming ---
        start_time_umap_transform = time.time()
        print("Transforming all sequences using the fitted UMAP model...")
        umap_embed = umap_model.transform(svd_embed)
        end_time_umap_transform = time.time()
        print(f"UMAP transformation completed in {end_time_umap_transform - start_time_umap_transform:.2f} seconds.\n")

    else:
        print("Running UMAP dimensionality reduction (fitting and transforming)...")
        umap_embed = umap_model.fit_transform(svd_embed)
        end_time_umap_fit_transform = time.time()
        print(f"UMAP model fitting completed in {end_time_umap_fit_transform - start_time_umap_fit:.2f} seconds.\n")

    # --- Start Timing: Save Output ---
    start_time_save = time.time()
    
    if 'clonotypeKey' in df_input.columns:
        output_index = df_input['clonotypeKey']
    else:
        output_index = df_input.index
    
    # Create complete UMAP dataframe with NA values for skipped sequences
    complete_umap_df = pd.DataFrame(
        index=output_index,
        columns=[f'UMAP{i+1}' for i in range(args.umap_components)]
    )
    
    # Fill with UMAP embeddings for valid sequences
    for i, valid_idx in enumerate(valid_indices):
        complete_umap_df.iloc[valid_idx] = umap_embed[i]
    
    complete_umap_df.to_csv(output_path, index=True, sep='\t')
    
    # Save summary of skipped clonotypes
    skipped_summary_path = os.path.join(args.output_dir, 'skipped_clonotypes_summary.txt')
    with open(skipped_summary_path, 'w') as f:
        f.write(f"Number of clonotypes skipped due to invalid amino acid sequences: {len(invalid_seq_indices)}\n")
        f.write(f"Total clonotypes processed: {len(output_index)}\n")
        f.write(f"Valid clonotypes: {len(valid_indices)}\n")
        f.write(f"Skipped clonotypes: {len(invalid_seq_indices)}\n\n")
        if invalid_seq_indices:
            f.write("Skipped clonotypes:\n")
            for invalid_idx in invalid_seq_indices:
                clonotype_key = output_index.iloc[invalid_idx] if hasattr(output_index, 'iloc') else output_index[invalid_idx]
                sequence = df_input.iloc[invalid_idx][seq_col]
                f.write(f"{clonotype_key}\t{sequence}\n")
        else:
            f.write("No clonotypes were skipped.\n")
    print(f"Skipped clonotypes summary saved to {skipped_summary_path}")
    
    end_time_save = time.time()
    print(f'UMAP embeddings saved to {output_path} in {end_time_save - start_time_save:.2f} seconds.')
    
    total_run_time = end_time_save - start_time_load # Total time from start of loading to end of saving
    print(f"\nTotal analysis completed in {total_run_time:.2f} seconds.")

    if args.store_models:
        # with open(f"{args.output_dir}/svd_model.pickle", "wb") as f:
        #     pickle.dump(svd, f)
        if args.umap_backend == 'parametric-umap':
            # model.pkl
            umap_model.save(args.output_dir)
        # with open("umap_model.pickle", "wb") as f:
        #     pickle.dump(umap_model, f)

if __name__ == '__main__':
    main()