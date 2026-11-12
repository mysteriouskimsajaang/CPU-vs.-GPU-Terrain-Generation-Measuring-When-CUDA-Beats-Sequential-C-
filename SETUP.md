# How to build and run (Windows, Visual Studio + CUDA)

## Files

| File | What it is |
|---|---|
| `src/terrain.h` | The terrain math (lookup table, Perlin noise, octaves, height). Shared by the CPU and GPU. |
| `src/main.cu` | CPU version, GPU version (global memory), correctness tests, timing, CSV and picture output |
| `viewer/plots.py` | Makes the graphs and `results/summary.md` from the CSV files |
| `viewer/viewer.py` | Rotatable 3D viewer for the saved heights |
| `results/` | Everything the program and `plots.py` produce. This folder is committed so the whole group can read it. |

## One-time setup

Install in this order: **Visual Studio** (with "Desktop development with C++"), then the **CUDA Toolkit**.

1. Visual Studio → **Create a new project** → search "CUDA" → **CUDA 13.x Runtime** → name it `TerrainGen`.
2. In **Solution Explorer**, delete the sample `kernel.cu`. Then right-click the project → **Add → Existing Item** → pick `src/main.cu` and `src/terrain.h`.
3. In the toolbar, set **Release** and **x64**.
4. Right-click the project → **Properties** (with Configuration: Release, Platform: x64):
   - **CUDA C/C++ → Device → Code Generation**: `compute_75,sm_75` (GTX 1660 Super)
   - **CUDA C/C++ → Command Line → Additional Options**: `--fmad=false`
     (keeps the GPU's rounding the same as the CPU's; required by our proposal)
   - **CUDA C/C++ → Host → Additional Compiler Options**: `/openmp`
     (turns on the all-cores CPU version)
5. Press **Ctrl+F5** to build and run.

## Running

With no arguments it runs everything (about 5 to 8 minutes; close other programs first so the timings are clean).
To run one part only, set **Properties → Debugging → Command Arguments** to one of these:

| Argument | What it does | Research question |
|---|---|---|
| `maps` | Heightmaps and terrain stats for the three terrain types | |
| `test` | CPU vs. GPU correctness on every edge-case size | |
| `bench` | Timing on the seven world sizes, all three terrain types | RQ1, RQ4 |
| `octaves` | Timing with 1, 2, 4, 6 and 8 octaves (hilly, 1024 × 1024) | RQ2 |
| `blocks` | Timing with 64, 256, 512 and 1024 threads per block (hilly) | RQ3 |

Then make the graphs:

```
py -m pip install numpy matplotlib
py viewer\plots.py
```

## What ends up in `results/`

| File | What it holds |
|---|---|
| `environment.txt` | CPU, GPU, CUDA and compiler versions, settings, and a check that `--fmad=false` is working |
| `log_<mode>.txt` | A copy of everything the program printed |
| `stats.csv` | Lowest, highest and average height and smoothness of each terrain type |
| `correctness.csv` | One row per test case: CPU vs. GPU difference, repeat check, overrun check, smoothness |
| `correctness_summary.csv` | One row per world size: threads launched, idle threads, cases passed |
| `benchmark_sizes.csv`, `benchmark_octaves.csv`, `benchmark_blocks.csv` | Median, average, minimum and standard deviation of each version, the copy-in / kernel / copy-back split, and the speedups |
| `raw_sizes.csv`, `raw_octaves.csv`, `raw_blocks.csv` | Every single timed run |
| `<type>_1024x1024.bin` and `.bmp` | Heights for the 3D viewer and a grayscale picture |
| `plots/*.png` | The graphs and PNG heightmaps (from `plots.py`) |
| `summary.md` | All the main numbers as Markdown tables, ready to paste into the README (from `plots.py`) |

### Reading the timing columns

- `median_ms` for a GPU row is the **total**: copy table in + kernel + copy heights back.
- `h2d_median_ms` = copying the 512-byte table to the GPU. `d2h_median_ms` = copying all the heights back.
- `transfer_pct` = share of the GPU total spent copying.
- `speedup_vs_1core_total` and `speedup_vs_allcores_total` = CPU median / GPU total median. The `_kernel` versions leave the copies out.

## Sharing results with the group

After a run, commit the results folder so groupmates can see it on GitHub:

```
git add results src viewer SETUP.md .gitignore
git commit -m "Mid-milestone results"
git push
```

## 3D viewer

```
py -m pip install numpy pyvista matplotlib
py viewer\viewer.py results\hilly_1024x1024.bin
```

Drag to rotate, scroll to zoom. Blue is low, red is high.

## Adding a new GPU version (Week 13)

The steps are listed at the top of `src/main.cu`: add a name to `GpuVariant`, write the kernel, add one `case` in `launchTerrain()`, and add the name to `TEST_VARIANTS` and `BENCH_VARIANTS`. The tests, timing, CSV files and graphs pick it up automatically.

## If something goes wrong

- **"CUDA 13.x Runtime" template missing**: CUDA was installed before Visual Studio. Re-run the CUDA installer.
- **"OpenMP : OFF" in `environment.txt`**: the `/openmp` option in step 4 is missing.
- **"GPU a\*b+c merging : ON"** in `environment.txt`: the `--fmad=false` option in step 4 is missing.
- **"Build : DEBUG"** in `environment.txt`: switch the toolbar to Release and rebuild.
- Anything else: copy the full error from the **Output** window and send it.
