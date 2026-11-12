# Results summary

Made by `viewer/plots.py` from the CSV files in this folder. Times are medians in milliseconds.

## Environment

```text
=== Environment ===
Run started        : 2026-11-12 21:46:50 (mode: all)
CPU                : Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
CPU logical cores  : 6
OpenMP             : on, 6 threads for the all-cores CPU version
Host compiler      : MSVC 195136260 (_MSC_VER 1951)
Build              : Release (optimized)
GPU                : NVIDIA GeForce GTX 1660 SUPER
GPU details        : compute capability 7.5, 22 SMs, 6.0 GB
CUDA toolkit       : 13.5 (compiled with), runtime 13.5, driver supports CUDA 13.5
Managed prefetch   : not available on this system (concurrent managed access)
GPU a*b+c merging  : off (--fmad=false is working)
GPU one-time setup : 184.45 ms (start-up + cudaMalloc, not part of any timing)
CPU a*b+c merging  : off
Math               : 32-bit float on CPU and GPU, same functions (terrain.h)
Seeds              : 3762183612631 (main), 42 (second test seed)
Timing             : 1 warm-up run (thrown away) + 30 timed runs per case
Baseline block     : 16 x 16 (256 threads per block)
Speedup            : CPU median time / GPU median time
GPU total time     : copy table in + kernel + copy heights back
```

## RQ1: world size (time in ms, speedup = CPU time / GPU total time)

**plains** (2 octaves)

| World | CPU 1 core | CPU all cores | GPU kernel | GPU total | vs 1 core | vs all cores | Faster |
|---|---:|---:|---:|---:|---:|---:|---|
| 16 x 16 | 0.0107 | 0.0063 | 0.0043 | 0.0568 | 0.2x | 0.1x | CPU |
| 32 x 32 | 0.0411 | 0.0154 | 0.0037 | 0.0525 | 0.8x | 0.3x | CPU |
| 48 x 48 | 0.0948 | 0.0334 | 0.0038 | 0.0503 | 1.9x | 0.7x | GPU vs 1 core only |
| 64 x 64 | 0.16 | 0.0553 | 0.0037 | 0.0887 | 1.8x | 0.6x | GPU vs 1 core only |
| 256 x 256 | 2.66 | 0.48 | 0.0074 | 0.12 | 22.5x | 4.1x | GPU |
| 1024 x 1024 | 43.6 | 11.1 | 0.0655 | 1.01 | 43.3x | 11.1x | GPU |
| 4096 x 4096 | 703 | 183 | 0.97 | 14.8 | 47.6x | 12.4x | GPU |

**hilly** (4 octaves)

| World | CPU 1 core | CPU all cores | GPU kernel | GPU total | vs 1 core | vs all cores | Faster |
|---|---:|---:|---:|---:|---:|---:|---|
| 16 x 16 | 0.0201 | 0.0047 | 0.0047 | 0.0531 | 0.4x | 0.1x | CPU |
| 32 x 32 | 0.0784 | 0.0156 | 0.0049 | 0.0585 | 1.3x | 0.3x | GPU vs 1 core only |
| 48 x 48 | 0.18 | 0.0320 | 0.0050 | 0.0542 | 3.3x | 0.6x | GPU vs 1 core only |
| 64 x 64 | 0.32 | 0.10 | 0.0049 | 0.0538 | 5.9x | 1.9x | GPU |
| 256 x 256 | 5.06 | 1.44 | 0.0109 | 0.13 | 39.9x | 11.4x | GPU |
| 1024 x 1024 | 83.5 | 24.2 | 0.12 | 0.95 | 87.6x | 25.4x | GPU |
| 4096 x 4096 | 1353 | 388 | 1.87 | 15.9 | 84.8x | 24.3x | GPU |

**mountains** (6 octaves)

| World | CPU 1 core | CPU all cores | GPU kernel | GPU total | vs 1 core | vs all cores | Faster |
|---|---:|---:|---:|---:|---:|---:|---|
| 16 x 16 | 0.0288 | 0.0066 | 0.0060 | 0.0516 | 0.6x | 0.1x | CPU |
| 32 x 32 | 0.12 | 0.0325 | 0.0061 | 0.0892 | 1.3x | 0.4x | GPU vs 1 core only |
| 48 x 48 | 0.27 | 0.0727 | 0.0061 | 0.0556 | 4.9x | 1.3x | GPU |
| 64 x 64 | 0.48 | 0.0890 | 0.0061 | 0.0554 | 8.7x | 1.6x | GPU |
| 256 x 256 | 7.87 | 2.17 | 0.0156 | 0.13 | 59.9x | 16.5x | GPU |
| 1024 x 1024 | 130 | 37.3 | 0.18 | 1.07 | 122.2x | 35.0x | GPU |
| 4096 x 4096 | 2112 | 583 | 2.83 | 16.9 | 124.7x | 34.4x | GPU |

## RQ4: cost of copying data (GPU, ms)

**plains**

| World | Bytes copied back | Copy table in | Kernel | Copy heights back | Total | Copying share |
|---|---:|---:|---:|---:|---:|---:|
| 16 x 16 | 1,024 | 0.0277 | 0.0043 | 0.0229 | 0.0568 | 91.9% |
| 32 x 32 | 4,096 | 0.0244 | 0.0037 | 0.0246 | 0.0525 | 93.0% |
| 48 x 48 | 9,216 | 0.0235 | 0.0038 | 0.0230 | 0.0503 | 93.6% |
| 64 x 64 | 16,384 | 0.0484 | 0.0037 | 0.0369 | 0.0887 | 96.4% |
| 256 x 256 | 262,144 | 0.0239 | 0.0074 | 0.0873 | 0.12 | 94.0% |
| 1024 x 1024 | 4,194,304 | 0.0535 | 0.0655 | 0.89 | 1.01 | 93.5% |
| 4096 x 4096 | 67,108,864 | 0.0659 | 0.97 | 13.8 | 14.8 | 93.5% |

**hilly**

| World | Bytes copied back | Copy table in | Kernel | Copy heights back | Total | Copying share |
|---|---:|---:|---:|---:|---:|---:|
| 16 x 16 | 1,024 | 0.0251 | 0.0047 | 0.0230 | 0.0531 | 91.1% |
| 32 x 32 | 4,096 | 0.0271 | 0.0049 | 0.0268 | 0.0585 | 91.7% |
| 48 x 48 | 9,216 | 0.0249 | 0.0050 | 0.0240 | 0.0542 | 90.0% |
| 64 x 64 | 16,384 | 0.0246 | 0.0049 | 0.0241 | 0.0538 | 91.4% |
| 256 x 256 | 262,144 | 0.0258 | 0.0109 | 0.0896 | 0.13 | 90.8% |
| 1024 x 1024 | 4,194,304 | 0.0261 | 0.12 | 0.81 | 0.95 | 87.3% |
| 4096 x 4096 | 67,108,864 | 0.12 | 1.87 | 14.0 | 15.9 | 88.3% |

**mountains**

| World | Bytes copied back | Copy table in | Kernel | Copy heights back | Total | Copying share |
|---|---:|---:|---:|---:|---:|---:|
| 16 x 16 | 1,024 | 0.0239 | 0.0060 | 0.0209 | 0.0516 | 89.3% |
| 32 x 32 | 4,096 | 0.0490 | 0.0061 | 0.0340 | 0.0892 | 93.0% |
| 48 x 48 | 9,216 | 0.0245 | 0.0061 | 0.0255 | 0.0556 | 89.4% |
| 64 x 64 | 16,384 | 0.0245 | 0.0061 | 0.0249 | 0.0554 | 88.9% |
| 256 x 256 | 262,144 | 0.0257 | 0.0156 | 0.0904 | 0.13 | 88.2% |
| 1024 x 1024 | 4,194,304 | 0.0323 | 0.18 | 0.83 | 1.07 | 82.8% |
| 4096 x 4096 | 67,108,864 | 0.12 | 2.83 | 14.0 | 16.9 | 83.1% |

## RQ2: octaves (hilly, 1024 x 1024, ms)

| Octaves | CPU 1 core | CPU all cores | GPU kernel | GPU total | vs 1 core | vs all cores | Copying share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.0 | 5.34 | 0.0411 | 0.92 | 28.2x | 5.8x | 95.5% |
| 2 | 43.5 | 11.5 | 0.0656 | 0.92 | 47.4x | 12.6x | 92.8% |
| 4 | 83.5 | 24.2 | 0.12 | 1.07 | 77.9x | 22.6x | 88.6% |
| 6 | 145 | 39.9 | 0.20 | 1.13 | 128.3x | 35.3x | 82.5% |
| 8 | 246 | 63.1 | 0.31 | 1.37 | 179.5x | 46.1x | 77.2% |

## RQ3 (first half): threads per block (hilly, GPU, ms)

| World | Block | Threads per block | Kernel | Total | vs 1 core (total) |
|---|---|---:|---:|---:|---:|
| 1024 x 1024 | 8 x 8 | 64 | 0.12 | 0.98 | 85.3x |
| 1024 x 1024 | 16 x 16 | 256 | 0.12 | 1.27 | 66.0x |
| 1024 x 1024 | 32 x 16 | 512 | 0.13 | 1.05 | 79.5x |
| 1024 x 1024 | 32 x 32 | 1024 | 0.14 | 1.01 | 83.1x |
| 4096 x 4096 | 8 x 8 | 64 | 1.84 | 16.3 | 82.9x |
| 4096 x 4096 | 16 x 16 | 256 | 1.89 | 16.1 | 83.8x |
| 4096 x 4096 | 32 x 16 | 512 | 2.03 | 17.4 | 77.8x |
| 4096 x 4096 | 32 x 32 | 1024 | 2.32 | 16.3 | 83.2x |

## Correctness: CPU vs. GPU on the edge-case sizes

**162 of 162 cases passed.** Each size is run with 2 seeds, 3 origins and 3 terrain types. "Idle threads" were launched but had no column, so the bounds check stopped them.

| World | Grid of blocks | Threads launched | Idle threads | Cases | Passed | Largest height difference | Columns that differ |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 x 1 | 1 x 1 | 256 | 255 | 18 | 18 | 0 | 0 |
| 3 x 3 | 1 x 1 | 256 | 247 | 18 | 18 | 0 | 0 |
| 2 x 100 | 1 x 7 | 1,792 | 1,592 | 18 | 18 | 0 | 0 |
| 100 x 2 | 7 x 1 | 1,792 | 1,592 | 18 | 18 | 0 | 0 |
| 15 x 17 | 1 x 2 | 512 | 257 | 18 | 18 | 0 | 0 |
| 17 x 5 | 2 x 1 | 512 | 427 | 18 | 18 | 0 | 0 |
| 255 x 257 | 16 x 17 | 69,632 | 4,097 | 18 | 18 | 0 | 0 |
| 1001 x 777 | 63 x 49 | 790,272 | 12,495 | 18 | 18 | 0 | 0 |
| 4097 x 4097 | 257 x 257 | 16,908,544 | 123,135 | 18 | 18 | 0 | 0 |

## Terrain stats (1024 x 1024, main seed)

| Type | Octaves | Made by | Matches CPU | Lowest | Highest | Average | Steepest step | Limit | Smooth | Clamped columns |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| plains | 2 | gpu_global | yes | 59 | 68 | 63.18 | 1 | 2 | pass | 0 |
| hilly | 4 | gpu_global | yes | 59 | 84 | 71.29 | 1 | 4 | pass | 0 |
| mountains | 6 | gpu_global | yes | 93 | 188 | 148.29 | 3 | 8 | pass | 0 |

