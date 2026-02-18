# Practice 2.2 — MPI Sparse Matrix Multiplication (mD8K)

MPI parallelization of the OpenMP version from Practice 1.2. The program multiplies large sparse matrices and
collects the final results on **rank 0**, using MPI communication primitives.

## Goal
Parallelize the original sparse matrix multiplication code using **MPI** (distributed memory), ensuring that:
- Work is split across processes as evenly as possible.
- The final output is assembled and printed only by **process 0** (root). :contentReference[oaicite:1]{index=1}

## Problem / Data model
- Matrix size: `N` (large, e.g., 8000 in the original mD8K code family).
- Number of sparse non-zero entries: `ND`.
- Sparse entries are stored as triplets `(i, j, v)` (row, col, value). :contentReference[oaicite:2]{index=2}

## Parallel strategy (high level)
1. **MPI init**: `MPI_Init`, `MPI_Comm_size`, `MPI_Comm_rank`. :contentReference[oaicite:3]{index=3}
2. **Balanced partitioning of rows/columns**
   - Each rank gets a contiguous block, with remainders distributed to keep the load balanced.
   - Arrays like `iterations`, `elements`, and `offsets` are used to manage the chunk each rank computes. :contentReference[oaicite:4]{index=4}
3. **Local computation**
   - Each process computes only its assigned subrange of the result.
4. **Collect results on root**
   - Dense result parts (e.g., `C1`) are assembled using `MPI_Gatherv`. :contentReference[oaicite:5]{index=5}
   - For the sparse compressed output (`CD`), each rank computes its local number of non-zeros, and root:
     - sums totals with `MPI_Reduce`,
     - gathers per-rank counts with `MPI_Gather`,
     - computes displacement offsets,
     - merges all sparse triplets with `MPI_Gatherv`. :contentReference[oaicite:6]{index=6}
5. **Print only on rank 0**
   - Avoids duplicated output and guarantees data is collected correctly. :contentReference[oaicite:7]{index=7}

## Results (summary)
Measured speedups increase up to ~**32 processes**, after which performance drops due to synchronization/communication overhead (e.g., 64 and 128 processes slower than 32). :contentReference[oaicite:8]{index=8}

## Files
- `mD8K_mpi_par.c` — MPI parallel implementation
- `Documentació_P2.2_CPM.pdf` — report (design + results + pseudocode) :contentReference[oaicite:9]{index=9}

## Build
```bash
mpicc -O2 mD8K_mpi_par.c -o mD8K_mpi_par
