# matrix_multiply_cuda

Square integer matrix multiplication on the GPU with CUDA, written for a parallel-processing course
assignment (assignment 4; the task sheet is `Assignment No. 4.pdf` and the write-up, in Persian, is
`report.pdf`).

Every variant fills two `N x N` matrices with random integers in `[0, 100)`, multiplies them on the GPU
using a **single thread block**, and copies the result back. The four `.cu` files are standalone programs
that differ in how threads are laid out and how much work each thread does:

| File | Block layout | Work per thread | Arguments |
|------|--------------|-----------------|-----------|
| `1D_threads(1st_phase).cu` | 1D block of `N*N` threads | one output element (`threadIdx.x` is split into row/column) | `<n> <m> <p>` |
| `2D_threads(1st_phase).cu` | 2D block of `N x N` threads | one output element (`threadIdx.y`, `threadIdx.x`) | `<n> <m> <p>` |
| `1D_threads(2nd_phase).cu` | 1D block of `thrN` threads | a `(N/sqrt(thrN))^2` tile of the output | `<N> <thrN>` |
| `2D_threads(2nd_phase).cu` | 2D block of `sqrt(thrN) x sqrt(thrN)` threads (capped at 32x32) | a tile of the output | `<N> <thrN>` |

Notes on the inputs:

- The code only uses `m` as the matrix width, so the 1st-phase programs effectively assume `n == m == p`.
  Because a single block is used, `N*N` must not exceed the device's block-size limit (1024 threads).
- In the 2nd-phase programs `thrN` must be a perfect square and `sqrt(thrN)` must divide `N`; otherwise the
  program prints `INAPPROPRIATE NUMBER OF THREADS / MATRICE SIZE !`.
- If the arguments are missing, the program prompts for them on stdin.

The 1st-phase programs print the matrices A, B and C. The 2nd-phase programs run the GPU multiplication
100 times and a serial CPU multiplication 100 times, print the average elapsed time of each (wall-clock
seconds, measured with `time()`), the speedup, and optionally the matrices (answer `d` at the prompt).

## Building

The sources were written with Visual Studio / the CUDA toolkit on Windows: they call `scanf_s` and
`system("PAUSE")`. The `Makefile` builds all four programs with `nvcc` and, on non-Windows hosts, maps
`scanf_s` to `scanf` so they compile unchanged (`system("PAUSE")` just prints a harmless error there).

```sh
make                    # builds 1D_phase1, 2D_phase1, 1D_phase2, 2D_phase2
./2D_phase2 64 256      # 64x64 matrices, 256 threads (16x16 block)
./1D_phase1 16 16 16    # 16x16 matrices, one thread per element, prints A, B, C
make clean
```

Set `NVCC`/`NVCCFLAGS` on the `make` command line to point at a specific toolkit or add `-arch=sm_XX`.
