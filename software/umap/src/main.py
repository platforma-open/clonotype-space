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
  --umap-backend    Choose UMAP implementation: 'auto' (default, tries cuml then sklearn),
                    'cuml' (RAPIDS cuml.UMAP, requires GPU), 'sklearn' (umap-learn, CPU-based).
  --output-dir      Directory to save output files (default: current directory)

Sampling Strategy (Internal Logic):
  - If total sequences < 50,000: No sampling.
  - If total sequences between 50,000 and 200,000 (inclusive of 50k, exclusive of 200k): Sample 50% of total sequences for UMAP fitting.
  - If total sequences >= 200,000: Sample 100,000 sequences for UMAP fitting.
"""

import argparse
import itertools
import numpy as np
import pandas as pd
import sys
import os
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import time
import gc
# import pickle

# Constants for sampling thresholds
SAMPLING_THRESHOLD_1 = 50000
SAMPLING_THRESHOLD_2 = 200000
FIXED_SAMPLE_SIZE_LARGE_DATA = 100000

def kmer_count_vectors(sequences, k=3):
    """
    Convert amino acid sequences to k-mer count vectors.
    
    Args:
        sequences (list): List of amino acid sequences
        k (int): Size of k-mers to count
        
    Returns:
        numpy.ndarray: Sparse matrix of k-mer counts (CSR format)
    """
    print(f"Generating {k}-mer count vectors...")
    # Standard amino acid alphabet (excluding '*' for stop codons if not desired in kmers)
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Generate all possible k-mers
    all_kmers = [''.join(p) for p in itertools.product(amino_acids, repeat=k)]
    kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}

    num_seqs = len(sequences)
    num_kmers = len(all_kmers)
    
    # Use LIL matrix for efficient assignment, convert to CSR for computation
    matrix = sparse.lil_matrix((num_seqs, num_kmers), dtype=np.int32)

    # Process sequences in batches for progress feedback
    batch_size = 10000 
    for i in range(0, num_seqs, batch_size):
        batch_end = min(i + batch_size, num_seqs)
        print(f"Processing sequences {i+1} to {batch_end} of {num_seqs}...")
        
        for j in range(i, batch_end):
            seq = str(sequences[j]).upper().strip("_")
            for pos in range(len(seq) - k + 1):
                kmer = seq[pos:pos + k]
                idx = kmer_to_index.get(kmer)
                if idx is not None:
                    matrix[j, idx] += 1
    
    return matrix.tocsr()

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
            import cuml.manifold.umap as cuml_umap
            import torch
            if torch.cuda.is_available():
                print("Using GPU-accelerated UMAP (RAPIDS cuML)...")
                return cuml_umap.UMAP(**common_params, init = 'spectral', n_epochs = 2000), 'gpu'
        except ImportError:
            print("RAPIDS cuML not available. Using CPU-based UMAP...")
    if backend == 'parametric-umap':
        from umap.parametric_umap import ParametricUMAP
        return ParametricUMAP(n_components = common_params['n_components']), 'gpu'

    # Default to CPU-based UMAP
    import umap as umap_learn
    print("Using CPU-based UMAP (umap-learn)...")
    return umap_learn.UMAP(n_jobs=-1, **common_params), 'cpu'

def main():
    parser = argparse.ArgumentParser(
        description='Compute UMAP embeddings from amino acid sequences via k-mer counts and PCA.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input TSV file with sequence column')
    parser.add_argument('-c', '--seq-col-start', default='aaSequence',
                        help='Starting string of the column containing amino acid sequences (default: "aaSequence")')
    parser.add_argument('-u', '--umap-output', required=True,
                        help='Output TSV file for UMAP embeddings')
    parser.add_argument('--dr-components', type=int, default=5,
                        help='Number of dimensionality reduction components (TruncatedSVD) before UMAP (default: 5)')
    parser.add_argument('--umap-components', type=int, default=2,
                        help='Number of UMAP dimensions (default: 2)')
    parser.add_argument('--umap-neighbors', type=int, default=8,
                        help='UMAP n_neighbors (default: 8)')
    parser.add_argument('--umap-min-dist', type=float, default=0.05,
                        help='UMAP min_dist (default: 0.05)')
    parser.add_argument('--k-mer-size', type=int, default=3,
                        help='Size of k-mers to use for sequence analysis (default: 3 for amino acids)')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--umap-backend', type=str, default='auto',
                        choices=['auto', 'cuml', 'sklearn'],
                        help='Choose UMAP implementation:\n'
                             '  "auto" (default): Tries cuml.UMAP first, falls back to umap-learn.\n'
                             '  "cuml": Forces RAPIDS cuml.UMAP (requires CUDA-enabled GPU and cuml installed).\n'
                             '  "sklearn": Forces umap-learn (CPU-based, no GPU required).')
    parser.add_argument('--store-models', type=bool, default=False,
                        help='Set to True to store SVD and UMAP models (default: False)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting k-mer UMAP analysis for amino acid sequences")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.umap_output}")
    print(f"Parameters: k-mer size={args.k_mer_size}, "
          f"DR components={args.dr_components}, "
          f"UMAP components={args.umap_components}, "
          f"UMAP neighbors={args.umap_neighbors}, "
          f"UMAP min_dist={args.umap_min_dist}, "
          f"UMAP Backend={args.umap_backend.upper()}")

    # Validate input parameters
    if args.dr_components < 1:
        print("Error: Number of dimensionality reduction components must be at least 1")
        sys.exit(1)
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

    seq_col_list = sorted([c for c in df_input.columns 
                        if c.startswith(args.seq_col_start)])
    if len(seq_col_list) == 0:
        print(f"Error: Columns starting with '{args.seq_col_start}' not found in input TSV. Available columns: {', '.join(df_input.columns)}")
        sys.exit(1)

    seq_col = "combined_sequence"
    df_input[seq_col] = df_input[seq_col_list].agg(''.join, axis=1)
    
    initial_seq_count = len(df_input)
    df_input = df_input[df_input[seq_col].notna() & (df_input[seq_col].str.strip('_').str.len() > 0)]
    if len(df_input) < initial_seq_count:
        print(f"Warning: Removed {initial_seq_count - len(df_input)} empty or invalid sequences.")

    sequences = df_input[seq_col].tolist()
    num_total_sequences = len(sequences)

    if not sequences:
        print('Error: No valid sequences found after filtering. Exiting.')
        sys.exit(1)
    
    valid_aas = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*'} 
    
    invalid_seq_indices = []
    for i, seq in enumerate(sequences):
        clean_seq = str(seq).upper().strip("_")
        if not all(aa in valid_aas for aa in clean_seq):
            invalid_seq_indices.append(i)
    
    if invalid_seq_indices:
        print(f"Warning: Found {len(invalid_seq_indices)} sequences with invalid amino acid characters. These sequences will be processed, but results for them might be noisy.")
        print(f"Example invalid sequence (first 5): {[sequences[i] for i in invalid_seq_indices[:5]]}")
    
    end_time_load = time.time()
    print(f"Input loading and preprocessing completed in {end_time_load - start_time_load:.2f} seconds.\n")

    # --- Start Timing: K-mer Counting ---
    start_time_kmer = time.time()
    matrix = kmer_count_vectors(sequences, k=args.k_mer_size)
    end_time_kmer = time.time()
    print(f"K-mer counting completed in {end_time_kmer - start_time_kmer:.2f} seconds.\n")
    
    # --- Start Timing: Truncated SVD ---
    start_time_svd = time.time()
    print("Running Truncated SVD...")
    svd = TruncatedSVD(n_components=args.dr_components, random_state=42)
    svd_embed = svd.fit_transform(matrix)
    print(f"Explained variance ratio by {args.dr_components} components: {sum(svd.explained_variance_ratio_):.3f}")
    end_time_svd = time.time()
    print(f"Truncated SVD completed in {end_time_svd - start_time_svd:.2f} seconds.\n")
    
    # -- Subsample dataset if working with CPU --
    if num_total_sequences < SAMPLING_THRESHOLD_1:
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

    # --- UMAP Backend Selection and Sampling Logic ---
    umap_model = None
    
    # Determine which UMAP backend to use
    if args.umap_backend in ['cuml', 'auto', 'sklearn', 'parametric-umap']:
        umap_model, run_type = create_umap_model(args.umap_backend, args.umap_components, 
                                    args.umap_neighbors, args.umap_min_dist)
    else:
        print(f"Error: Unknown UMAP backend '{args.umap_backend}'. Exiting.")
        sys.exit(1)

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
        
        # Clean GPU memory
        v = 1
        while v != 0:
            v = gc.collect()

    # --- Start Timing: Save Output ---
    start_time_save = time.time()
    output_path = os.path.join(args.output_dir, args.umap_output)
    
    if 'clonotypeKey' in df_input.columns:
        output_index = df_input['clonotypeKey']
    else:
        output_index = df_input.index
    
    umap_df = pd.DataFrame(
        umap_embed,
        index=output_index,
        columns=[f'UMAP{i+1}' for i in range(args.umap_components)]
    )
    umap_df.to_csv(output_path, index=True, sep='\t')
    end_time_save = time.time()
    print(f'UMAP embeddings saved to {output_path} in {end_time_save - start_time_save:.2f} seconds.')
    
    total_run_time = end_time_save - start_time_load # Total time from start of loading to end of saving
    print(f"\nTotal analysis completed in {total_run_time:.2f} seconds.")
    time.sleep(1)

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