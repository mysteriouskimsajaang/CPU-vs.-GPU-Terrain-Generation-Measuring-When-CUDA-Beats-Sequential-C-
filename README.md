# CPU vs. GPU Terrain Generation: Measuring When CUDA Beats Sequential C++

**CEPARCO Integrating Project: Project Proposal (Revised Milestone #1)**

**Group members:** Matthew Fabregas, Kane Ibanez, Bowen Li, Jat Teoxon

## Abstract

In Minecraft, the world is made from a seed. For every (x, z) spot, or column, the game uses layered noise (similar to Perlin noise) to decide how high the ground is. Each column only needs its own coordinates, the seed, and a small lookup table, so every column can be worked out on its own. That is what makes it a good fit for parallel computing.

We will build a simple version of this that only makes the height of each spot: no caves, structures or villages. We will write it twice and run both on the same PC. One is a normal C++ program that goes through the spots one at a time, and the other is a CUDA program that gives each spot to its own GPU thread. We will check that both give the same heights, then time them on different world sizes and terrain types (plains, hilly and mountains) to find out when the GPU actually becomes faster, and why.

## 1. Project Description

Minecraft generates terrain one 16 × 16 chunk (256 columns) at a time, and performance mods such as C2ME spread this work over several CPU cores. A chunk is a small job for a GPU, because launching a kernel and copying data has a fixed cost. But if we generate a large world at once, the GPU may end up faster than the CPU. We will not copy Minecraft's generator exactly. We use it as a reference to build our own heightmap generator and measure how it performs.

Our research questions are below. They may be adjusted as we learn more during our project creation.

1. **RQ1:** At what world size (16 × 16 up to 4096 × 4096 columns) does the GPU become faster than the CPU?
2. **RQ2:** How does the speedup change with the number of noise octaves and the terrain type?
3. **RQ3:** Does it matter where we keep the lookup table (global, constant or shared memory), and how many threads we put in each block?
4. **RQ4:** How much of the GPU's total time is just copying data to and from the GPU?

We also check that the terrain is smooth, meaning no sudden cliffs between neighboring columns.

### Inputs

Nothing is loaded from a file. Everything is set by numbers we type in:

| Input | Values |
|---|---|
| Seed | 64-bit integer. Default `3762183612631`. A second seed, `42`, is used in correctness tests. |
| World size (W × H columns) | See the size tables below. |
| World origin (x0, z0) | (0, 0) by default. Tests also use (-5000, 3000) and (1000000, -1000000) for negative and far-away coordinates. |
| Terrain type | Plains, hilly, or mountains (see section 2). |
| Octaves | Default per terrain type. For RQ2 we also try 1, 2, 4, 6 and 8 octaves on the hilly type. |
| Block size | 16 × 16 (256 threads) by default. For RQ3 we also try 8 × 8, 32 × 16 and 32 × 32. |

**A note on the last two rows.** More octaves adds more small details to the terrain, like rocks popping out of the hills. Fewer octaves gives smoother terrain, like the Chocolate Hills. Block size doesn't change the terrain at all. It only changes how the threads are grouped: many small groups or a few big ones.

**Benchmark sizes** (used for timing, from one Minecraft chunk up to a Large world):

| Test | World size | Columns | Chunks (16 × 16) |
|---|---:|---:|---:|
| One chunk | 16 × 16 | 256 | 1 |
| Tiny | 64 × 64 | 4,096 | 16 |
| Small | 256 × 256 | 65,536 | 256 |
| Medium | 1024 × 1024 | 1,048,576 | 4,096 |
| Large | 4096 × 4096 | 16,777,216 | 65,536 |

**Edge-case sizes** (used for correctness and bounds checking). All the benchmark sizes are multiples of 16, so they never leave a partly filled block. These test whether our bounds check actually works:

| Size | Why |
|---|---|
| 1 × 1, 3 × 3 | Smaller than one block. A 1 × 1 world has no neighbors. |
| 2 × 100, 100 × 2 | Very thin: one side smaller than a block, the other very long. |
| 15 × 17, 17 × 5 | One side smaller than a block and one side larger. |
| 255 × 257 | Just under and just over a multiple of 16. |
| 1001 × 777 | An odd-sized world that isn't square. |
| 4097 × 4097 | One past the largest benchmark size. |

### Outputs

- A saved heightmap for each run, plus a grayscale picture of it
- A rotatable 3D terrain viewer (Python, using the saved heights shrunk to about 512 × 512), colored from blue (low) to red (high). It is not part of any timing.
- The run time of the CPU version and the GPU version (GPU kernel time, and total GPU time including copying data)
- A check that the CPU and GPU outputs match, and a smoothness report per terrain type
- The speedup, saved in a CSV file and shown in graphs for RQ1 to RQ4

## 2. What we will parallelize and how

This part describes which parts of the program can run in parallel, and how we split the work between many workers instead of one doing all the calculations.

| Step | CPU (sequential) | GPU (CUDA) |
|---|---|---|
| 1. Height of every column | Nested loop over all columns | One kernel, one thread per column |
| 2. Max, min and sum of heights | One loop over the height array | Reduction kernel (atomics, and a shared-memory version) |
| 3. Smoothness check | One loop comparing each column to its right and lower neighbor | One kernel, then the same reduction |

In step 1, each column's result does not depend on any other column, so they can all be done at the same time. 

Step 2 finds the highest point, the lowest point, and the total of all the heights. If every thread writes to the same result at the same time, they have to wait in line for each other, which is slow (this is the atomics version). So we also try a shared-memory version: the threads are split into groups (blocks), each group finds its own highest, lowest and total, and then we combine the group answers.

Step 3 checks if the terrain is smooth by comparing each column to its neighbors. Each column needs its neighbors' heights to already exist, so this only starts after all the heights are generated.

### Our plan for the GPU version

- **One thread per column.** Each thread works out which column it owns from its block number and its position in the block. We launch enough blocks to cover the whole world, rounding up. Some threads in the last blocks fall outside the world, so each thread first checks `if (x < W && z < H)` and does nothing if it is outside. The odd edge-case sizes test this. Basically, each thread computes exactly one height, and a thread that has no column to compute (like in the edge-case sizes) simply does nothing.
- **Same function on CPU and GPU.** We write the height function once with `__host__ __device__`, so both versions run the same math. 
- **Permutation table.** We build it once on the CPU from the seed (512 bytes) and copy it to the GPU. We test three places to keep it: global, constant and shared memory. Constant memory is fast when all threads read the same spot, but our threads read different spots, so we treat it as something to test, not as a sure speedup. Both versions use this same table, which is built on the CPU, so their outputs match and the comparison is fair.
- **Block size.** We test 64, 256, 512 and 1024 threads per block (8 × 8, 16 × 16, 32 × 16, 32 × 32).
- **Max, min and sum.** First version: every thread uses `atomicMax`, `atomicMin` and `atomicAdd` on one value. Second version: each block combines its own values in shared memory (waiting with `__syncthreads()` until the whole block is done), then writes one result with one atomic. Our lecture says many atomics on one spot slow things down, so we expect the second version to be faster, which is why we will test that.
- **Data transfer.** We compare copying the data ourselves (`cudaMalloc` and `cudaMemcpy`) with letting CUDA do it (`cudaMallocManaged`, with and without prefetch). The heights stay on the GPU while steps 2 and 3 run. We copy the full height array back only to compare with the CPU or to save it.
- We follow the CUDA thread, block and memory model from the CUDA C++ Programming Guide (NVIDIA, n.d.) and our lecture.

### The terrain formulas

The CPU and GPU versions use the same formulas and numbers. Only the way they run is different: the CPU goes through the columns one at a time, and the GPU gives each column its own thread.

**A. Seed and permutation table.** The seed is a 64-bit number. We use it to start a pseudo-random generator, then use that to shuffle the numbers 0 to 255 into a lookup table (the permutation table). The same seed always gives the same table. The table is built once on the CPU and copied to the GPU, so both versions use the identical table. We repeat it to 512 entries so lookups don't need to wrap around.

```text
rng = mt19937_64(seed)
p = [0, 1, 2, ..., 255]
for i = 255 down to 1:  j = rng() mod (i + 1);  swap(p[i], p[j])
perm[i] = p[i mod 256] for i = 0..511
```

We use a Fisher-Yates shuffle. We start from the end of the list and swap each number with a randomly chosen number from the part of the list before it. The random numbers come from `mt19937_64`, a standard C++ generator that gives the same numbers for the same seed on any computer. This way, if other group members run the code, they get the exact same terrain, so our results can be compared fairly. 

**B. 2D Perlin noise (adapted from the improved version, Perlin 2002).**

Imagine graph paper where every crossing has an arrow chosen from the permutation table. To get a value for a point, we look at the four crossings (corners) around the square it sits in. Each one pushes the value a little depending on how the point lines up with its arrow, then we blend the four pushes smoothly into one number. The fade curve keeps the blending smooth, so there are no sharp edges between squares. More octaves add small bumps on top, like rocky features. The result is a value between about -1 and 1. Nearby points give nearby values, so neighboring columns get similar heights.

```text
fade(t)       = 6t^5 - 15t^4 + 10t^3
lerp(t, a, b) = a + t * (b - a)
grad(h, x, z), using h mod 8:
  0: x     1: -x     2: z     3: -z
  4: x+z   5: -x+z   6: x-z   7: -x-z

noise(x, z):
  xi = floor(x); zi = floor(z)
  X = xi AND 255; Z = zi AND 255
  xf = x - xi; zf = z - zi
  u = fade(xf); v = fade(zf)
  aa = perm[perm[X] + Z]       ba = perm[perm[X+1] + Z]
  ab = perm[perm[X] + Z + 1]   bb = perm[perm[X+1] + Z + 1]
  return lerp(v, lerp(u, grad(aa, xf, zf),   grad(ba, xf-1, zf)),
                 lerp(u, grad(ab, xf, zf-1), grad(bb, xf-1, zf-1)))
```

The result is a smooth value between about -1 and 1. Nearby inputs give nearby outputs, so neighboring columns get similar heights.

**C. Octaves and terrain height.** Think of an octave as a zoom level. From far away you see one big smooth shape, closer you see hills on the slopes, and really close you see the rocks. The first layer is the big smooth shape, and each layer after it has twice the detail (lacunarity 2.0) and half the strength (persistence 0.5). We add the layers together, so we get a big hill with small bumps on it. Mountains also use **ridged** noise, which flips each layer so the places where the noise is near 0 become sharp ridges. With a low base frequency, we expect this to give long mountain ranges instead of random spikes.

```text
fbm(xw, zw):
  sum = 0; amp = 1; freq = f0
  for o = 0 .. N-1:
    n = noise(xw*freq + 17.37*o, zw*freq + 31.91*o)   // sample one zoom level
    if ridged: n = 1 - 2 * abs(n)                      // flip into sharp ridges
    sum += amp * n                                     // add it, scaled by its strength
    freq = freq * 2.0; amp = amp * 0.5                 // next layer: 2x detail, half strength
  return sum * invNorm   // invNorm = 1 / (2 * (1 - 0.5^N)), keeps it near -1 to 1

xw = x0 + x;  zw = z0 + z
h  = clamp( floor( B + A * fbm(float(xw), float(zw)) ), -64, 319 )
```

Recent Minecraft versions have a world 384 blocks tall, from Y = -64 to Y = 319 ("Minecraft height limit 2026", n.d.), so we use that as our height limit. The three terrain types only differ in these settings (B = base height, A = amplitude in blocks, f0 = base frequency, N = octaves):

| Type | Noise | B | A | f0 | N | Height range (B ± A) | Smoothness limit |
|---|---|---:|---:|---:|---:|---|---|
| Plains | normal | 64 | 8 | 0.004 | 2 | 56 to 72 | ≤ 2 blocks |
| Hilly | normal | 72 | 24 | 0.008 | 4 | 48 to 96 | ≤ 4 blocks |
| Mountains | ridged | 100 | 90 | 0.004 | 6 | 10 to 190 | ≤ 8 blocks |

Smoothness limit means the largest height difference between two neighboring columns. These limits are our targets, checked at each terrain type’s normal number of octaves (plains 2, hilly 4, mountains 6). If a terrain type goes over its limit during testing, we adjust its numbers and report it. The 1 to 8 octave test for RQ2 is only about timing.

We will also look at the heightmap picture and the highest and lowest heights to make sure each terrain type looks like its name (plains flat, hilly rolling, mountains tall and ridged). Since we are adding several layers of noise together, the highs and lows of different layers can partly cancel each other out, so the terrain may come out flatter than we expect, like a plain with a few bumps. If that happens, for example if the mountain features get washed out, we will adjust the numbers. Patel (2015) says to expect this kind of hand-tuning when generating terrain from noise.

**D. Precision, rounding and clamping.**

- All the noise math uses 32-bit `float` on both the CPU and GPU, so they do the same kind of math. Heights are whole numbers (`int`), and totals are 64-bit because a big world adds up to more than a normal `int` can hold.
- We round down with `floorf`, then convert to `int`. We use floor because it handles negative heights the same way every time. Heights are clamped to [-64, 319], and we report how many columns were clamped (expected 0).
- Our CPU and GPU should give exactly the same heights. We still allow a 1 block difference in case of tiny rounding differences, but we expect 0.
- To keep both sides doing the same math, the GPU is compiled with `nvcc -O3 --fmad=false` and the CPU is built in Release mode without fast-math. These settings are shortcuts that make a number round a little differently, so we turn them off.
- For the grayscale picture only, a height h is scaled to 0 to 255 using `round(255 * (h - hmin) / (hmax - hmin))`. There can be up to 384 different heights but only 256 grays, so a few heights may share a gray. If hmax = hmin, the picture is all one gray. This is just for display and is not part of the comparison.

**E. Statistics.** We compute `hmax` (highest column), `hmin` (lowest column), `hsum` (all heights added up) and the average height. For smoothness, each column gets `d = max(|h(x,z) - h(x+1,z)|, |h(x,z) - h(x,z+1)|)`, which is how much its height differs from the neighbor next to it and the neighbor behind it. We only use neighbors that exist, so a 1 × 1 world has d = 0. We report `dmax` (the steepest spot) and the average d.

### What will not be parallelized

- **Building the permutation table from the seed.** It's only 256 values and is done once on the CPU. We build it the same way for both versions so the CPU and GPU get the same output, which makes comparing and testing on other computers easier.
- **Allocating memory and copying data between the CPU and GPU.** This includes copying the permutation table to the GPU. It's part of the GPU's total time, so we measure it separately (RQ4).
- **Saving files, the 3D viewer, comparing CPU and GPU results, the timing statistics, the CSV file and the graphs.** These only help us check and show the results. They are not what we are measuring.

**If we have time:** a multi-core (OpenMP) CPU version as an extra reference, grass/dirt/stone layers per column, a simple "snowy" color for high ground, and a 3D-printed model of a small piece of the terrain. These are not part of the main scope or the milestones. Caves, structures and blending between terrain types are out of scope (THIS IS HARD).

## 3. Related studies

Green (2005) already ran improved Perlin noise on a GPU using shaders, and the result exactly matched the CPU version, so GPU noise is not new. The chapter also says the permutation table can be shared between implementations so they give the same results, which is what we do by building our table once and copying it to the GPU.

For Minecraft, the C2ME mod speeds up chunk generation by using more CPU cores (C2ME, n.d.).

Patel (2015) is a tutorial that uses the same octave and ridged-noise approach as ours, and notes that noise alone has no large-scale features like rivers.

### How ours is different

The existing GPU noise work targets shading and texturing in graphics APIs (mostly 3D noise), and C2ME targets multi-core CPUs. Our project uses CUDA to generate a whole heightmap and focuses on measuring: the world size where the GPU wins, the effect of work per column, memory type and block size, and the cost of copying data. We report both kernel time and total time including transfers, and we use the same PC for the CPU and GPU. We do not claim a new terrain algorithm.

Minecraft does not say why it generates terrain on the CPU, but we think there are a few reasons. The world is made in small 16 × 16 chunks, which is not much work for a GPU, and the results have to go back to the CPU anyway for things like mobs and collisions. Structures also have a lot of `if` conditions, which GPUs handle badly because threads that take different paths slow each other down. Servers may also have no GPU. This is our own guess. Our project tests the first reason directly by measuring how big the world has to be before the GPU is worth it.

## 4. Our setup

| Item | Details |
|---|---|
| CPU | Intel Core i5-8400 (6 cores, 6 threads) |
| GPU | NVIDIA GeForce GTX 1660 Super (Turing, compute capability 7.5, 22 SMs, 1,408 CUDA cores, 6 GB) |
| RAM and OS | 16 GB, Windows 10 |
| Software | Visual Studio and CUDA Toolkit (versions to be added after installation), NVIDIA driver 617.42 |
| Languages | C++ (CPU version), CUDA C++ (GPU version), Python (viewer and graphs only) |

All results come from this one PC, so the CPU and GPU are compared on the same machine. Google Colab is not used for any reported result. If we ever test on another machine, its results go in a separate table.

## 5. How we will test and measure

- The CPU and GPU versions run the same seed, terrain type and world size, one after the other on the same PC.
- We time the CPU with `std::chrono` and the GPU with CUDA events. We report kernel time (just the GPU math) and total time (copy to GPU, kernels, copy back).
- Each test has one warm-up run, which we throw away, and then 10 timed runs. We report the average, the minimum and the standard deviation.
- Speedup = CPU average time / GPU average time. We always say whether it uses kernel time or total time.

**Correctness checks** (run on every edge-case size, both seeds (3762183612631 and 42) and all three origins):

| Check | Pass criterion |
|---|---|
| Heights, CPU vs GPU | Largest difference ≤ 1 block and at most 0.01% of columns different. We expect 0 and will report the real numbers. |
| Max, min, sum, smoothness (GPU vs CPU on the same heights) | Exactly equal (integers) |
| Two GPU runs, same input | Identical heights |
| Smoothness | `dmax` of each terrain type ≤ its limit in section C |

## 6. Schedule and mid-project milestone

| Week | Objectives |
|---|---|
| Week 7 | Milestone #1: Revised proposal |
| Week 10 | Milestone #2 (mid-project demo): The CPU version and the first CUDA version (global memory) both generate heightmaps for the plains and hilly terrain types, and their outputs match within our tolerance (section 5) on all the edge-case sizes. We have our first CPU vs. GPU timings for at least three world sizes. A first version of the 3D viewer is used to check that the heights look like terrain, so we can adjust the noise numbers. |
| Week 13 | Milestone #3 (final submission and demo): A polished, working project for all three terrain types (including mountains) and all five world sizes. This includes the constant and shared memory versions, the octave and block-size tests, the statistics and smoothness checks, all correctness tests passed, the results CSV and graphs, the write-up, and the finished GitHub repo. |

## References

C2ME. (n.d.). *Concurrent chunk management engine* [Mod]. CurseForge. https://www.curseforge.com/minecraft/mc-mods/c2me

Green, S. (2005). Implementing improved Perlin noise. In M. Pharr (Ed.), *GPU Gems 2* (pp. 409–416). Addison-Wesley. https://developer.nvidia.com/gpugems/GPUGems2/gpugems2_chapter26.html

Minecraft height limit 2026 – How high can you build? (Y=-64 to 319). (n.d.). BlockMosaic. https://mcimagetool.com/blog/minecraft-height-limit/

NVIDIA Corporation. (n.d.). *CUDA C++ programming guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

Patel, A. J. (2015). *Making maps with noise functions*. Red Blob Games. https://www.redblobgames.com/maps/terrain-from-noise/

Perlin, K. (2002). Improving noise. *ACM Transactions on Graphics, 21*(3), 681–682. https://doi.org/10.1145/566654.566636


Changes since the approved proposal

Dropped the managed-memory comparison: on Windows, CUDA doesn’t support prefetching managed memory, so that test can’t run as planned. We use cudaMalloc + cudaMemcpy, which also lets us time the copying directly (RQ4).
Added a multi-core CPU version (OpenMP, 6 threads) next to the single-core one, so the GPU is also compared against the whole CPU.
Tests change one setting at a time from a baseline instead of trying every combination.
“Total GPU time” = copying the table in + kernels + copying the heights back. One-time setup is reported separately.

VS Studio Settings
Device: compute_75,sm_75, 1660 super uses 75 architecture
Command line --fmad=false, turn shortcut off so we dont get mismatch values from cpu and gpu
Hist: /openMP allows loop to run on seveal CPU cores, so we can compare 6 CPU cores to the others