// main.cu - CPU vs. GPU terrain generation (Week 10 milestone version)
//
// What this program does:
//   test   : correctness - CPU vs. GPU heights on every edge-case size,
//            both seeds, all three origins, all three terrain types
//   bench  : timing - CPU (1 core), CPU (all cores, if OpenMP is on) and
//            GPU (global memory) on the five benchmark sizes
//   maps   : saves heightmaps (.bin for the 3D viewer, .bmp picture) and
//            prints height and smoothness stats for each terrain type
//   (no argument runs all three)
//
// Build in Visual Studio (CUDA Runtime project), Release, x64.
// Required nvcc option: --fmad=false   (see SETUP.md)

#define _CRT_SECURE_NO_WARNINGS   // lets Visual Studio accept fopen
#include "terrain.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>

// Creating the "results" folder (works without C++17)
#ifdef _WIN32
#include <direct.h>
#define MAKE_DIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MAKE_DIR(path) mkdir(path, 0755)
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
const uint64_t SEED_MAIN = 3762183612631ULL;
const uint64_t SEED_TEST = 42ULL;
const int WARMUP_RUNS = 1;
const int TIMED_RUNS  = 30;
const int BLOCK_X = 16, BLOCK_Y = 16;   // 256 threads per block (baseline)

struct Size { int w, h; };
struct Origin { int x0, z0; };

const Size BENCH_SIZES[] = {{16, 16}, {64, 64}, {256, 256}, {1024, 1024}, {4096, 4096}};
const Size EDGE_SIZES[]  = {{1, 1}, {3, 3}, {2, 100}, {100, 2}, {15, 17}, {17, 5},
                            {255, 257}, {1001, 777}, {4097, 4097}};
const Origin ORIGINS[]   = {{0, 0}, {-5000, 3000}, {1000000, -1000000}};

// ---------------------------------------------------------------------------
// A. Permutation table: Fisher-Yates shuffle driven by mt19937_64(seed)
// ---------------------------------------------------------------------------
void buildPermutation(uint64_t seed, uint8_t perm[512]) {
    std::mt19937_64 rng(seed);
    uint8_t p[256];
    for (int i = 0; i < 256; i++) p[i] = (uint8_t)i;
    for (int i = 255; i >= 1; i--) {
        int j = (int)(rng() % (uint64_t)(i + 1));
        std::swap(p[i], p[j]);
    }
    for (int i = 0; i < 512; i++) perm[i] = p[i & 255];
}

// ---------------------------------------------------------------------------
// CPU versions
// ---------------------------------------------------------------------------
void cpuGenerate(int W, int H, int x0, int z0, const TerrainParams& p,
                 const uint8_t* perm, int* out) {
    for (int z = 0; z < H; z++)
        for (int x = 0; x < W; x++)
            out[(size_t)z * W + x] = heightAt(x0, z0, x, z, p, perm);
}

void cpuGenerateParallel(int W, int H, int x0, int z0, const TerrainParams& p,
                         const uint8_t* perm, int* out) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int z = 0; z < H; z++)
        for (int x = 0; x < W; x++)
            out[(size_t)z * W + x] = heightAt(x0, z0, x, z, p, perm);
}

// ---------------------------------------------------------------------------
// E. Statistics (CPU reference; the GPU reduction comes in Week 13)
// ---------------------------------------------------------------------------
struct Stats {
    int hmin, hmax;
    long long hsum;
    double havg;
    int dmax;
    double davg;
    long long clamped;
};

Stats computeStats(const int* h, int W, int H) {
    Stats s{h[0], h[0], 0, 0.0, 0, 0.0, 0};
    long long dsum = 0;
    for (int z = 0; z < H; z++) {
        for (int x = 0; x < W; x++) {
            int v = h[(size_t)z * W + x];
            s.hmin = std::min(s.hmin, v);
            s.hmax = std::max(s.hmax, v);
            s.hsum += v;
            if (v == HEIGHT_MIN || v == HEIGHT_MAX) s.clamped++;
            int d = 0;
            if (x + 1 < W) d = std::max(d, std::abs(v - h[(size_t)z * W + x + 1]));
            if (z + 1 < H) d = std::max(d, std::abs(v - h[(size_t)(z + 1) * W + x]));
            s.dmax = std::max(s.dmax, d);
            dsum += d;
        }
    }
    double n = (double)W * H;
    s.havg = s.hsum / n;
    s.davg = dsum / n;
    return s;
}

// ---------------------------------------------------------------------------
// Output files
// ---------------------------------------------------------------------------
void saveHeightsBin(const std::string& path, const int* h, int W, int H) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("Could not write %s\n", path.c_str()); return; }
    fwrite(&W, sizeof(int), 1, f);
    fwrite(&H, sizeof(int), 1, f);
    fwrite(h, sizeof(int), (size_t)W * H, f);
    fclose(f);
}

// Grayscale picture as a 24-bit BMP (opens in Windows Photos without extra libraries).
// gray = round(255 * (h - hmin) / (hmax - hmin)); all gray if hmax == hmin.
void saveGrayBMP(const std::string& path, const int* h, int W, int H) {
    int hmin = h[0], hmax = h[0];
    for (size_t i = 0; i < (size_t)W * H; i++) { hmin = std::min(hmin, h[i]); hmax = std::max(hmax, h[i]); }
    int rowBytes = (W * 3 + 3) & ~3;
    uint32_t dataSize = (uint32_t)rowBytes * H, fileSize = 54 + dataSize;
    uint8_t hdr[54] = {'B', 'M'};
    auto put32 = [&](int off, uint32_t v) { memcpy(hdr + off, &v, 4); };
    auto put16 = [&](int off, uint16_t v) { memcpy(hdr + off, &v, 2); };
    put32(2, fileSize); put32(10, 54); put32(14, 40);
    put32(18, (uint32_t)W); put32(22, (uint32_t)H);
    put16(26, 1); put16(28, 24); put32(34, dataSize);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("Could not write %s\n", path.c_str()); return; }
    fwrite(hdr, 1, 54, f);
    std::vector<uint8_t> row(rowBytes, 0);
    for (int z = H - 1; z >= 0; z--) {          // BMP rows go bottom to top
        for (int x = 0; x < W; x++) {
            int v = h[(size_t)z * W + x];
            uint8_t g = (hmax == hmin) ? 128
                      : (uint8_t)std::lround(255.0 * (v - hmin) / (hmax - hmin));
            row[x * 3] = row[x * 3 + 1] = row[x * 3 + 2] = g;
        }
        fwrite(row.data(), 1, rowBytes, f);
    }
    fclose(f);
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
struct TimeStats { double avg, mn, sd, med; };

TimeStats summarize(const std::vector<double>& t) {
    double sum = 0, mn = t[0];
    for (double v : t) { sum += v; mn = std::min(mn, v); }
    double avg = sum / t.size(), var = 0;
    for (double v : t) var += (v - avg) * (v - avg);
    std::vector<double> sorted = t;
    std::sort(sorted.begin(), sorted.end());
    size_t m = sorted.size() / 2;
    double med = sorted.size() % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2.0;
    return {avg, mn, t.size() > 1 ? std::sqrt(var / (t.size() - 1)) : 0.0, med};
}

template <typename F>
TimeStats timeCPU(F fn) {
    for (int i = 0; i < WARMUP_RUNS; i++) fn();
    std::vector<double> t;
    for (int i = 0; i < TIMED_RUNS; i++) {
        auto a = std::chrono::high_resolution_clock::now();
        fn();
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    return summarize(t);
}

// ---------------------------------------------------------------------------
// GPU version 1: global memory, one thread per column
// ---------------------------------------------------------------------------
#ifdef __CUDACC__

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void terrainKernelGlobal(int* out, int W, int H, int x0, int z0,
                                    TerrainParams p, const uint8_t* perm) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && z < H)                      // bounds check for partial blocks
        out[(size_t)z * W + x] = heightAt(x0, z0, x, z, p, perm);
}

// GPU buffers are allocated once (setup, not timed) and reused.
struct GpuBuffers {
    int* d_out = nullptr;
    uint8_t* d_perm = nullptr;
    size_t capacity = 0;     // in columns
    void ensure(size_t columns) {
        if (columns <= capacity) return;
        if (d_out) CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaMalloc(&d_out, columns * sizeof(int)));
        capacity = columns;
        if (!d_perm) CUDA_CHECK(cudaMalloc(&d_perm, 512));
    }
    ~GpuBuffers() { if (d_out) cudaFree(d_out); if (d_perm) cudaFree(d_perm); }
};

struct GpuTiming { float kernelMs, totalMs; };

// One full GPU run: copy table in -> kernel -> copy heights back.
GpuTiming gpuGenerate(GpuBuffers& buf, int W, int H, int x0, int z0,
                      const TerrainParams& p, const uint8_t* perm, int* hostOut,
                      cudaEvent_t ev[4]) {
    size_t n = (size_t)W * H;
    buf.ensure(n);
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);   // round up

    CUDA_CHECK(cudaEventRecord(ev[0]));
    CUDA_CHECK(cudaMemcpy(buf.d_perm, perm, 512, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev[1]));
    terrainKernelGlobal<<<grid, block>>>(buf.d_out, W, H, x0, z0, p, buf.d_perm);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev[2]));
    CUDA_CHECK(cudaMemcpy(hostOut, buf.d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaEventSynchronize(ev[3]));

    GpuTiming t;
    CUDA_CHECK(cudaEventElapsedTime(&t.kernelMs, ev[1], ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.totalMs, ev[0], ev[3]));
    return t;
}

// Keeps the GPU busy for ~0.5 s so it leaves power-saving mode and reaches
// its full clock speed before we time anything (the GPU slows down while it
// sits idle during the CPU runs). Not timed.
void warmUpGPU(GpuBuffers& buf, const TerrainParams& p, const uint8_t* perm) {
    const int W = 4096, H = 4096;
    buf.ensure((size_t)W * H);
    CUDA_CHECK(cudaMemcpy(buf.d_perm, perm, 512, cudaMemcpyHostToDevice));
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - start).count() < 500.0) {
        terrainKernelGlobal<<<grid, block>>>(buf.d_out, W, H, 0, 0, p, buf.d_perm);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void printGpuInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | compute %d.%d | %d SMs | %.1f GB\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    int cma = 0;
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, 0);
    printf("Concurrent managed access (needed for prefetch): %s\n", cma ? "Yes" : "No");
}
#endif

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
void runTests() {
#ifndef __CUDACC__
    printf("[test] Needs the CUDA build. Skipped.\n");
#else
    printf("\n=== Correctness tests: CPU vs. GPU (global memory) ===\n");
    FILE* csv = fopen("results/correctness.csv", "w");
    fprintf(csv, "width,height,seed,x0,z0,type,max_diff,mismatches,mismatch_pct,gpu_repeat_identical,pass\n");
    GpuBuffers buf;
    cudaEvent_t ev[4];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
    uint64_t seeds[] = {SEED_MAIN, SEED_TEST};
    int total = 0, passed = 0;
    for (auto sz : EDGE_SIZES) {
        size_t n = (size_t)sz.w * sz.h;
        std::vector<int> cpu(n), gpu(n), gpu2(n);
        for (uint64_t seed : seeds) {
            uint8_t perm[512];
            buildPermutation(seed, perm);
            for (auto o : ORIGINS) {
                for (int type = 0; type < 3; type++) {
                    TerrainParams p = makePreset(type);
                    cpuGenerate(sz.w, sz.h, o.x0, o.z0, p, perm, cpu.data());
                    gpuGenerate(buf, sz.w, sz.h, o.x0, o.z0, p, perm, gpu.data(), ev);
                    gpuGenerate(buf, sz.w, sz.h, o.x0, o.z0, p, perm, gpu2.data(), ev);
                    int maxDiff = 0; long long mism = 0;
                    for (size_t i = 0; i < n; i++) {
                        int d = std::abs(cpu[i] - gpu[i]);
                        if (d) { mism++; maxDiff = std::max(maxDiff, d); }
                    }
                    bool same = (gpu == gpu2);
                    double pct = 100.0 * mism / n;
                    bool ok = maxDiff <= 1 && pct <= 0.01 && same;
                    total++; passed += ok;
                    fprintf(csv, "%d,%d,%llu,%d,%d,%s,%d,%lld,%.6f,%d,%d\n", sz.w, sz.h,
                            (unsigned long long)seed, o.x0, o.z0, typeName(type),
                            maxDiff, mism, pct, same, ok);
                    if (!ok || mism)
                        printf("  %dx%d seed=%llu origin=(%d,%d) %s: max_diff=%d mismatches=%lld %s\n",
                               sz.w, sz.h, (unsigned long long)seed, o.x0, o.z0, typeName(type),
                               maxDiff, mism, ok ? "PASS" : "FAIL");
                }
            }
        }
        printf("  %4d x %-4d done\n", sz.w, sz.h);
    }
    for (auto& e : ev) cudaEventDestroy(e);
    fclose(csv);
    printf("Passed %d / %d cases (details in results/correctness.csv)\n", passed, total);
#endif
}

void runBenchmarks() {
    printf("\n=== Benchmarks (seed %llu, origin (0,0), %d warm-up + %d timed runs) ===\n",
           (unsigned long long)SEED_MAIN, WARMUP_RUNS, TIMED_RUNS);
#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif
    FILE* csv = fopen("results/benchmark.csv", "w");
    fprintf(csv, "version,type,octaves,width,height,block,threads,median_ms,avg_ms,min_ms,sd_ms,"
                 "kernel_median_ms,kernel_avg_ms,kernel_min_ms,kernel_sd_ms\n");
    uint8_t perm[512];
    buildPermutation(SEED_MAIN, perm);

#ifdef __CUDACC__
    GpuBuffers buf;
    cudaEvent_t ev[4];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
    // One-time setup (GPU start-up + allocating the largest buffer), reported separately.
    auto s0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaFree(0));
    buf.ensure((size_t)4096 * 4096);
    auto s1 = std::chrono::high_resolution_clock::now();
    printf("GPU one-time setup (start-up + cudaMalloc): %.2f ms (not part of any timing)\n",
           std::chrono::duration<double, std::milli>(s1 - s0).count());
#endif

    printf("%-10s %-8s %11s %12s %12s %12s %12s %10s %10s\n", "type", "", "size",
           "CPU 1 core", "CPU all", "GPU kernel", "GPU total", "x kernel", "x total");
    for (int type : {PLAINS, HILLY}) {
        TerrainParams p = makePreset(type);
        for (auto sz : BENCH_SIZES) {
            size_t n = (size_t)sz.w * sz.h;
            std::vector<int> out(n);
            TimeStats c1 = timeCPU([&] { cpuGenerate(sz.w, sz.h, 0, 0, p, perm, out.data()); });
            fprintf(csv, "cpu_1core,%s,%d,%d,%d,0,1,%.6f,%.6f,%.6f,%.6f,,,,\n", typeName(type), p.octaves,
                    sz.w, sz.h, c1.med, c1.avg, c1.mn, c1.sd);
            TimeStats cN = timeCPU([&] { cpuGenerateParallel(sz.w, sz.h, 0, 0, p, perm, out.data()); });
            fprintf(csv, "cpu_all_cores,%s,%d,%d,%d,0,%d,%.6f,%.6f,%.6f,%.6f,,,,\n", typeName(type), p.octaves,
                    sz.w, sz.h, threads, cN.med, cN.avg, cN.mn, cN.sd);
#ifdef __CUDACC__
            warmUpGPU(buf, p, perm);   // GPU idled during the CPU runs, wake it up
            for (int i = 0; i < WARMUP_RUNS; i++) gpuGenerate(buf, sz.w, sz.h, 0, 0, p, perm, out.data(), ev);
            std::vector<double> tk, tt;
            for (int i = 0; i < TIMED_RUNS; i++) {
                GpuTiming g = gpuGenerate(buf, sz.w, sz.h, 0, 0, p, perm, out.data(), ev);
                tk.push_back(g.kernelMs); tt.push_back(g.totalMs);
            }
            TimeStats gk = summarize(tk), gt = summarize(tt);
            fprintf(csv, "gpu_global,%s,%d,%d,%d,%d,0,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", typeName(type),
                    p.octaves, sz.w, sz.h, BLOCK_X * BLOCK_Y, gt.med, gt.avg, gt.mn, gt.sd,
                    gk.med, gk.avg, gk.mn, gk.sd);
            printf("%-10s %-8s %5d x %-5d %10.3f ms %10.3f ms %10.4f ms %10.4f ms %9.1fx %9.1fx\n",
                   typeName(type), "", sz.w, sz.h, c1.med, cN.med, gk.med, gt.med,
                   c1.med / gk.med, c1.med / gt.med);
#else
            printf("%-10s %-8s %5d x %-5d %10.3f ms %10.3f ms\n", typeName(type), "", sz.w, sz.h,
                   c1.med, cN.med);
#endif
            fflush(csv);
        }
    }
#ifdef __CUDACC__
    for (auto& e : ev) cudaEventDestroy(e);
#endif
    fclose(csv);
    printf("CPU all-cores threads: %d%s\n", threads,
           threads == 1 ? " (OpenMP is off - see SETUP.md to turn it on)" : "");
    printf("Times shown are medians of %d runs (avg/min/sd are in the CSV).\n", TIMED_RUNS);
    printf("Speedups (x kernel / x total) are CPU 1 core divided by GPU time.\n");
    printf("Saved results/benchmark.csv\n");
}

void runMaps() {
    const int W = 1024, H = 1024;
    printf("\n=== Heightmaps (%dx%d, seed %llu) ===\n", W, H, (unsigned long long)SEED_MAIN);
    uint8_t perm[512];
    buildPermutation(SEED_MAIN, perm);
    std::vector<int> h((size_t)W * H);
    printf("%-10s %6s %6s %8s %6s %7s %8s\n", "type", "min", "max", "avg", "dmax", "davg", "clamped");
    for (int type = 0; type < 3; type++) {
        TerrainParams p = makePreset(type);
        cpuGenerate(W, H, 0, 0, p, perm, h.data());
        Stats s = computeStats(h.data(), W, H);
        printf("%-10s %6d %6d %8.2f %6d %7.3f %8lld\n", typeName(type), s.hmin, s.hmax, s.havg,
               s.dmax, s.davg, s.clamped);
        std::string base = std::string("results/") + typeName(type) + "_" +
                           std::to_string(W) + "x" + std::to_string(H);
        saveHeightsBin(base + ".bin", h.data(), W, H);
        saveGrayBMP(base + ".bmp", h.data(), W, H);
    }
    printf("Smoothness targets (dmax): plains <= 2, hilly <= 4, mountains <= 8\n");
    printf("Saved results/<type>_%dx%d.bin and .bmp\n", W, H);
}

int main(int argc, char** argv) {
    MAKE_DIR("results");   // does nothing if it already exists
    std::string mode = argc > 1 ? argv[1] : "all";
#ifdef __CUDACC__
    printGpuInfo();
#else
    printf("CPU-only build (no CUDA).\n");
#endif
    if (mode == "maps"  || mode == "all") runMaps();
    if (mode == "test"  || mode == "all") runTests();
    if (mode == "bench" || mode == "all") runBenchmarks();
    return 0;
}
