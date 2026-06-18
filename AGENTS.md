# Agents.md

Guidance for coding agents working in this repository.

## Project Context

This project started as a Graph500/NVSHMEM graph traversal implementation. The original benchmark-oriented code is kept in `src/`. The active refactor toward a reusable library is in `src-refactor/`.

Treat `src/` as reference code unless the task explicitly asks to modify the original implementation. New library-facing work should normally happen in `src-refactor/` and the top-level `Makefile`.

## Repository Layout

- `src/`: original Graph500-derived implementation with MPI/NVSHMEM traversal, validation, file-backed tuple graph support, global state, and benchmark runner logic.
- `src-refactor/`: current refactor into smaller modules: worker/runtime setup, arena allocation, graph generation, tuple-to-CSR conversion, and visualization.
- `Makefile`: builds the refactored binary by default and the original binary with `make old`.
- `dot/`: generated DOT/SVG visualization output. Avoid treating generated visualizations as source unless the task is specifically about examples or docs.
- `graph-*.bin` and `*.weights`: generated graph dump files. Do not depend on them as canonical inputs unless the task says so.
- `docs/`: thesis/reference material.

## Build And Run

Build the refactored runner:

```sh
make
```

Build the original runner:

```sh
make old
```

Run a small single-rank smoke test when CUDA, MPI, and NVSHMEM are available:

```sh
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP=mpi mpirun --allow-run-as-root -np 1 ./build/graph500_runner 12 16
```

Generate DOT files for small graphs only:

```sh
CUDA_VISIBLE_DEVICES=0 NVSHMEM_BOOTSTRAP=mpi WRITE_DOT=1 mpirun --allow-run-as-root -np 1 ./build/graph500_runner 4 2
make dots-svg
```

Important environment variables from the original runner include `SKIP_BFS=1`, `SKIP_SSSP=1`, `SKIP_VALIDATION=1`, `TMPFILE=<filename>`, and `REUSEFILE=1`. The refactored runner currently uses `WRITE_DOT` and always calls `tuple_graph_dump()` in `main.cu`.

## Refactor Architecture

The current refactor is C-style CUDA/C++ and is organized around these modules:

- `base.{h,cu}`: fixed-width typedefs, utility macros, CUDA error checking, and arena allocation. GPU arenas use `nvshmem_malloc()` and are symmetric across PEs.
- `worker.{h,cu}`: initializes MPI, selects a CUDA device by rank, initializes NVSHMEM with `MPI_COMM_WORLD`, creates the GPU arena, and provides rank-symmetric arena allocation helpers.
- `graph_generation.{h,cu}`: generates Graph500 Kronecker tuple edges and SSSP weights on the GPU. It also has temporary dump/load helpers using hard-coded `graph-refactor.bin` filenames.
- `mrg_transitions.cu`: generated transition table included directly by `graph_generation.cu`.
- `traversal.{h,cu}`: converts a distributed tuple graph into a cyclic 1D CSR-like `Oned_Graph` using NVSHMEM atomics.
- `visualization.{h,cu}`: copies graph data to host and writes per-PE/global DOT files for tuple and 1D graph layouts.
- `main.cu`: demo/runner entry point, not a stable library API.

## Original Code Reference Points

When checking behavior against the original implementation:

- Original Graph500 types, distribution macros, tuple graph storage, and iteration macros are in `src/common.h`.
- Original runner, graph generation orchestration, validation, BFS root selection, and benchmark flow are in `src/graph500_runner.c`.
- Original Kronecker generator is in `src/graph_generator.cu`.
- Original NVSHMEM BFS/SSSP implementation is in `src/traversal.cu`.
- Original CPU CSR conversion and validation helpers are in `src/csr_reference.*`, `src/bitmap_reference.h`, and `src/validate.c`.

Use `src/` to preserve algorithmic compatibility, generated edge ordering, weight generation, distribution semantics, and validation behavior.

## Coding Guidelines

- Prefer small, direct changes. The refactor is still in progress, so avoid large abstractions unless they remove concrete duplication or expose a needed library boundary.
- Keep the refactored code C-style unless a task explicitly asks for modern C++ APIs.
- Use the fixed-width aliases from `base.h` in `src-refactor/` code.
- Keep MPI/NVSHMEM calls collective where the surrounding code expects collectives. Do not introduce rank-conditional early returns around collective operations.
- Preserve cyclic vertex ownership semantics: owner is `vertex % rank_size`, local index is `vertex / rank_size`.
- GPU arena allocations through `worker_arena_push_array()` are sized to the maximum count across ranks. This is required for symmetric NVSHMEM addressing.
- `Oned_Graph` memory currently belongs to the worker GPU arena; `oned_graph_free()` only clears the struct and does not reclaim arena memory.
- Avoid adding hidden global state to `src-refactor/`. The original code has substantial global state; the refactor should move away from that pattern.
- Do not modify generated graph binaries, DOT/SVG files, or build outputs unless the task explicitly asks for generated artifacts.

## Known Refactor Gaps

- There is not yet a clean public library target; the default build still links a runner from `main.cu`.
- BFS/SSSP traversal APIs from the original benchmark have not been fully ported into `src-refactor/`.
- Validation and BFS root selection are still only in the original `src/` flow.
- `tuple_graph_dump()` and `tuple_graph_load()` use hard-coded filenames and host staging buffers.
- Some implementation comments mark code as generated or not fully reviewed, especially in graph dump/load and visualization.
- `compute_rowstarts_kernel()` is a single-thread prefix sum placeholder.
- Error handling is mostly abort-oriented and not yet library-friendly.
- The README still references some original paths and behavior; verify current code before relying on it.

## Verification Expectations

For non-trivial changes, prefer this order:

1. Build the refactored target with `make`.
2. If compatibility with the original is relevant, build `make old`.
3. Run a small MPI/NVSHMEM smoke test if the environment has CUDA/NVSHMEM.
4. For graph generation changes, compare `graph-refactor.bin` and `graph-refactor.bin.weights` against `graph-old.bin` and `graph-old.bin.weights` when reproducibility is the goal.
5. For visualization changes, run with a very small scale and `WRITE_DOT=1`, then optionally run `make dots-svg` if Graphviz is installed.

If CUDA, MPI, or NVSHMEM are unavailable in the environment, state that verification was limited to static inspection or host-side checks.

## Safety Notes

- Do not run destructive git commands such as `git reset --hard` or `git checkout --` unless explicitly requested.
- Do not remove original `src/` code during the refactor unless the user explicitly asks for that cleanup.
- Do not assume generated files in `build/`, `dot/`, or `graph-*.bin*` are disposable user-unimportant files; ask before deleting them.
- Be careful with large scales. Graph generation and DOT output can consume substantial GPU memory, host memory, disk, and time.
