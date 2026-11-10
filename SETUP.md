# How to build and run (Windows, Visual Studio + CUDA)

## Files

| File | What it is |
|---|---|
| `src/terrain.h` | The terrain math (lookup table, Perlin noise, octaves, height). Shared by the CPU and GPU. |
| `src/main.cu` | CPU version, GPU version (global memory), correctness tests, timing, CSV and picture output |
| `viewer/viewer.py` | Rotatable 3D viewer for the saved heights |

## One-time setup

Install in this order: **Visual Studio 2022** (with "Desktop development with C++"), then the **CUDA Toolkit**.

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

With no arguments it runs everything (maps, tests, benchmarks, about 1–2 minutes). To run one part only, set
**Properties → Debugging → Command Arguments** to `maps`, `test` or `bench`.

Output goes to a `results` folder next to the project:

- `correctness.csv`: CPU vs. GPU on every edge-case size, seed, origin and terrain type
- `benchmark.csv`: timings (CPU 1 core, CPU all cores, GPU kernel, GPU total)
- `plains/hilly/mountains_1024x1024.bmp`: grayscale heightmaps (open in Photos)
- `plains/hilly/mountains_1024x1024.bin`: heights for the 3D viewer

The first lines printed show your GPU and whether "Concurrent managed access" is available (expected **No** on Windows, which is why we dropped the prefetch tests).

## 3D viewer

```
py -m pip install numpy pyvista matplotlib
py viewer\viewer.py results\hilly_1024x1024.bin
```

Drag to rotate, scroll to zoom. Blue is low, red is high.

## If something goes wrong

- **"CUDA 13.x Runtime" template missing**: CUDA was installed before Visual Studio. Re-run the CUDA installer.
- **`std::filesystem` errors**: Properties → **CUDA C/C++ → Common → C++ Dialect** → C++17 (or add `-std=c++17` to Additional Options).
- **"CPU all-cores threads: 1"**: the `/openmp` option in step 4 is missing.
- Anything else: copy the full error from the **Output** window and send it.
