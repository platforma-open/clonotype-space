"""Exact identical-row detection without sorting the matrix.

Each row's bytes are hashed to one 64-bit number (FNV-1a over the row's raw bytes, in parallel -- no per-row
Python objects), only that 1-D hash array is sorted, and rows sharing a hash are then byte-compared, so two
different rows that happen to collide are never merged. identical_row_groups never exposes the hashes: the grouping is
exact and the representative is the lowest row index whatever the hash values are. Extra memory ~ N * 24
bytes, versus the several full N x D copies np.unique(axis=0) makes.

Copied from embedding-clustering (software/src/hash_dedup.py); keep the two in sync. Adds row_hashes() and
hash_groups() so the streamed embedding path can fingerprint rows batch by batch and find the candidate
copies that fall in different batches.
"""
import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def _row_hashes(U):
    """FNV-1a over each row of U, a (N, k) uint32 view of a float32 matrix."""
    n, k = U.shape
    h = np.empty(n, dtype=np.uint64)
    for i in numba.prange(n):                           # rows are independent: hashed in parallel
        x = np.uint64(14695981039346656037)             # FNV-1a 64-bit offset basis (fixed start value)
        for j in range(k):
            # mix in one value: XOR it in, then multiply by the FNV prime (uint64 overflow wraps on purpose)
            x = (x ^ np.uint64(U[i, j])) * np.uint64(1099511628211)
        h[i] = x
    return h


@numba.njit(cache=True)
def _resolve(order, sorted_hashes, U):
    """rep[i] = the row every byte-identical copy of row i maps to. `order` is a STABLE argsort of the hashes,
    so rows sharing a hash are visited in ascending row index and the representative is the lowest index."""
    n = order.shape[0]
    rep = np.arange(n)                                  # start: every row is its own representative
    i = 0
    while i < n:
        # [i, j) = the run of rows (in hash order) that share one hash value
        j = i + 1
        while j < n and sorted_hashes[j] == sorted_hashes[i]:
            j += 1
        if j - i > 1:                                   # several rows share this hash: byte-compare them
            for a in range(i, j):
                ra = order[a]
                if rep[ra] != ra:                       # already claimed as a copy of an earlier row
                    continue
                # ra is the first unclaimed row of the run: claim every later row identical to it
                for b in range(a + 1, j):
                    rb = order[b]
                    if rep[rb] == rb:
                        same = True
                        for c in range(U.shape[1]):     # exact check: every value must match
                            if U[ra, c] != U[rb, c]:
                                same = False            # same hash, different row: a collision
                                break
                        if same:
                            rep[rb] = ra
        i = j                                           # next run
    return rep


def identical_row_groups(X):
    """Find the groups of byte-identical rows in a float32 matrix X (N x k). X is only read, never modified.

    Returns one array, rep: a NumPy int64 array with one entry per row of X, so length N. rep[i] is the row
    number of the first row (the lowest index) whose vector is byte-identical to row i:
      - a unique row, or the first row of its group, points to itself: rep[i] == i
      - a later copy points back to the first row of its group:        rep[i] < i
    e.g. rows 0, 2 and 4 identical, rows 1 and 3 unique -> rep = [0, 1, 0, 3, 0]."""
    U = np.ascontiguousarray(X, dtype=np.float32).view(np.uint32)   # same bytes read as integers: exact compare
    h = _row_hashes(U)                                  # 1. one 64-bit fingerprint per row
    order = np.argsort(h, kind="stable")                # 2. sort only the fingerprints; equal ones become
                                                        #    neighbours, in ascending row order (stable)
    return _resolve(order, h[order], U)                 # 3. confirm candidates byte by byte -> rep


def row_hashes(X):
    """64-bit FNV-1a fingerprint of every row of a float32 matrix X (N x k), as a uint64 array of length N.
    Identical rows always share a fingerprint; different rows almost never do, so a shared fingerprint only
    marks CANDIDATE copies -- confirm them byte by byte before merging."""
    return _row_hashes(np.ascontiguousarray(X, dtype=np.float32).view(np.uint32))


@numba.njit(cache=True)
def _runs(order, hashes):
    """For each run of `order` (an argsort of `hashes`) sharing one hash value: every row's rep = the run's
    lowest row index, and last[rep] = the run's highest row index."""
    n = order.shape[0]
    rep = np.empty(n, dtype=np.int64)
    last = np.full(n, -1, dtype=np.int64)
    i = 0
    while i < n:
        # [i, j) = the run of rows (in hash order) that share one hash value; lo/hi = its lowest/highest row
        h = hashes[order[i]]
        lo = order[i]
        hi = order[i]
        j = i + 1
        while j < n and hashes[order[j]] == h:
            if order[j] < lo:
                lo = order[j]
            if order[j] > hi:
                hi = order[j]
            j += 1
        for t in range(i, j):
            rep[order[t]] = lo
        last[lo] = hi
        i = j
    return rep, last


def hash_groups(hashes):
    """Group rows by fingerprint `hashes` (N,). Returns two int64 arrays of length N:
      rep[i]   the lowest row index with the same fingerprint (i itself for a unique one)
      last[j]  the highest row index whose rep is j (j itself when it has no copy, -1 when j is nobody's rep)
    Rows sharing a fingerprint are only CANDIDATE copies: byte-compare them before merging. Sorts only the
    1-D fingerprint array, in place; extra memory ~ 24 bytes per row."""
    return _runs(np.argsort(hashes), hashes)
