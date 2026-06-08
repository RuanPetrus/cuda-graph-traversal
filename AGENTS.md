# AGENTS.md

Instructions for agents working in this Graph500 repository.

## Project Layout

- `src/` contains the NVSHMEM traversal sources.
- `src/traversal.cu` contains the NVSHMEM BFS and SSSP kernels.
- `src/graph500_runner.c` contains the benchmark runner `main` function.
- `Makefile` at the repository root builds binaries into `build/`.
- `src/graph_generator.cu` contains the consolidated Graph500 graph generator implementation used by the NVSHMEM binaries.
- Host-side graph conversion and validation use direct MPI messaging.

## Compile
From the repository root:

```sh
make
```

Expected binaries:
- `build/graph500_runner`: runs the NVSHMEM BFS and custom NVSHMEM SSSP kernels.

Clean generated binaries with:

```sh
make clean
```

## Run The Benchmark

The binary takes `SCALE` and optional `edgefactor` arguments:

```sh
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP="mpi" mpirun --allow-run-as-root -np <ranks> ./build/graph500_runner <SCALE> [edgefactor]
```

Examples for a small local smoke test:

```sh
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP="mpi" mpirun --allow-run-as-root -n 1 -np 4 ./build/graph500_runner 14 16
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP="mpi" SKIP_BFS=1 mpirun --allow-run-as-root -n 1 -np 4 ./build/graph500_runner 14 16
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP="mpi" SKIP_SSSP=1 mpirun --allow-run-as-root -n 1 -np 4 ./build/graph500_runner 14 16
```

`SCALE` is `log2(number_of_vertices)`. `edgefactor` defaults to `16` when omitted.

## Check Correctness

Correctness validation is enabled by default. A valid run prints Graph500 result fields such as `SCALE`, `NBFS`, phase TEPS, and validation timing fields for whichever phases are enabled.

If validation fails, rank 0 prints a message like `Validation failed for this BFS root; skipping rest.` or `Validation failed for this SSSP root; skipping rest.`, and the final output contains:

```text
No results printed for invalid run.
```

Do not set `SKIP_VALIDATION=1` when checking correctness. Use it only for performance-only runs where correctness has already been established.

## Useful Environment Variables

- `SKIP_VALIDATION=1`: disables BFS/SSSP validation. Do not use for correctness checks.
- `SKIP_BFS=1`: skips the BFS phase.
- `SKIP_SSSP=1`: skips the SSSP phase.
- `TMPFILE=<filename>`: stores generated graph data using MPI file I/O instead of memory.
- `REUSEFILE=1`: keeps/reuses files named by `TMPFILE`; the runner also uses `<filename>.weights`.
- `VERBOSE=1`: enables some extra diagnostic messages while opening/reusing graph files.

## Notes For Agents

- Prefer small scales such as `12` or `14` for smoke tests; benchmark-scale runs can require substantial memory and time.
- The code assumes power-of-two MPI sizing by default. See `README` and `src/README` before changing process counts or related macros.
- Preserve validation unless the task is explicitly about performance-only benchmarking.

## Benchmarking
Set `SKIP_VALIDATION=1` when running benchmark, `22` is the biggest scale to run local. The benchmark value to look for is `bfs median_TEPS:`.
```
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP="mpi" SKIP_SSSP=1 SKIP_VALIDATION=1 mpirun --allow-run-as-root -n 1 -np 1 ./build/graph500_runner 22 16
```

For nvshmem SSSP-only benchmarks, set both `SKIP_BFS=1` and `SKIP_VALIDATION=1`. The benchmark value to look for is `sssp median_TEPS:`.

```sh
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP="mpi" SKIP_BFS=1 SKIP_VALIDATION=1 mpirun --allow-run-as-root -n 1 -np 1 ./build/graph500_runner 22 16
```
