// main.cu - CPU vs. GPU terrain generation (Week 10 milestone version)
//
// Modes (pass one as the command argument; no argument runs all of them):
//   maps    : saves heightmaps (.bin for the 3D viewer, .bmp picture) and
//             height + smoothness stats for each terrain type      -> stats.csv
//   test    : correctness - CPU vs. GPU on every edge-case size, both seeds,
//             all three origins, all three terrain types            -> correctness.csv,
//                                                                      correctness_summary.csv
//   bench   : RQ1/RQ4 - timing on the benchmark sizes               -> benchmark_sizes.csv
//   octaves : RQ2 - timing with 1, 2, 4, 6, 8 octaves (hilly)       -> benchmark_octaves.csv
//   blocks  : RQ3 - timing with different threads per block         -> benchmark_blocks.csv
//
// Every mode also writes environment.txt (what PC and software ran this) and
// log_<mode>.txt (a copy of everything printed), so the results can be read on
// GitHub without access to the PC that ran them.
//
// All files go to the "results" folder at the top of the repo.
//
// Build in Visual Studio (CUDA Runtime project), Release, x64.
// Required nvcc option: --fmad=false   (see SETUP.md)
//
// ---------------------------------------------------------------------------
// HOW TO ADD A NEW GPU VERSION (Week 13: constant memory, shared memory, ...)
//   1. Add a name to enum GpuVariant and to variantName().
//   2. Write the new __global__ kernel next to terrainKernelGlobal.
//   3. Add one "case" to launchTerrain().
//   4. Add the name to TEST_VARIANTS (correctness) and BENCH_VARIANTS (timing).
// Nothing else has to change: the tests, timing, CSV files and plots pick it up.
// ---------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS   // lets Visual Studio accept fopen
#include "terrain.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <chrono>
#include <string>
#include <thread>
#include <algorithm>

// Creating the "results" folder (works without C++17)
#ifdef _WIN32
#include <direct.h>
#define MAKE_DIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MAKE_DIR(path) mkdir(path, 0755)
#endif

// CPU name for environment.txt
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
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

struct Size   { int w, h; };
struct Origin { int x0, z0; };
struct Block  { int x, y; };

const Block BASELINE_BLOCK = {16, 16};   // 256 threads per block

// RQ1: world sizes. 32 and 48 are extra points around the size where the GPU
// starts to win (between one chunk and "Tiny").
const Size BENCH_SIZES[] = {{16, 16}, {32, 32}, {48, 48}, {64, 64}, {256, 256},
                            {1024, 1024}, {4096, 4096}};
// Correctness: sizes that leave partly filled blocks.
const Size EDGE_SIZES[]  = {{1, 1}, {3, 3}, {2, 100}, {100, 2}, {15, 17}, {17, 5},
                            {255, 257}, {1001, 777}, {4097, 4097}};
const Origin ORIGINS[]   = {{0, 0}, {-5000, 3000}, {1000000, -1000000}};

// RQ2: octave counts tried on the hilly type.
const int  OCTAVE_SWEEP[]     = {1, 2, 4, 6, 8};
const Size OCTAVE_SWEEP_SIZE  = {1024, 1024};
// RQ3: threads per block (8x8 = 64, 16x16 = 256, 32x16 = 512, 32x32 = 1024).
const Block BLOCK_SWEEP[]       = {{8, 8}, {16, 16}, {32, 16}, {32, 32}};
const Size  BLOCK_SWEEP_SIZES[] = {{1024, 1024}, {4096, 4096}};

// GPU versions. Week 13 adds GPU_CONSTANT and GPU_SHARED here.
enum GpuVariant { GPU_GLOBAL = 0 };
inline const char* variantName(int v) {
    switch (v) {
        case GPU_GLOBAL: return "gpu_global";
        default:         return "gpu_unknown";
    }
}
const int TEST_VARIANTS[]  = {GPU_GLOBAL};   // versions checked for correctness
const int BENCH_VARIANTS[] = {GPU_GLOBAL};   // versions that get timed
const Block TEST_BLOCKS[]  = {BASELINE_BLOCK};

const int CANARY_INTS = 64;                  // guard values after the height array
const int POISON      = 0x7F7F7F7F;          // "nothing was written here" marker

// ---------------------------------------------------------------------------
// Output folder and log file
// ---------------------------------------------------------------------------
static std::string g_outDir = "results";
static FILE* g_log = nullptr;

// printf that also writes to results/log_<mode>.txt
void LOG(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt); vprintf(fmt, a); va_end(a);
    fflush(stdout);
    if (g_log) { va_start(a, fmt); vfprintf(g_log, fmt, a); va_end(a); fflush(g_log); }
}

std::string outPath(const std::string& name) { return g_outDir + "/" + name; }

// Visual Studio runs the program from TerrainGen/TerrainGen, so look upward
// for the repo (the folder that contains src/terrain.h) and use its results folder.
void findOutputDir() {
    const char* ups[] = {"", "../", "../../", "../../../"};
    for (const char* up : ups) {
        FILE* f = fopen((std::string(up) + "src/terrain.h").c_str(), "rb");
        if (f) { fclose(f); g_outDir = std::string(up) + "results"; break; }
    }
    MAKE_DIR(g_outDir.c_str());   // does nothing if it already exists
}

FILE* openOut(const std::string& name) {
    FILE* f = fopen(outPath(name).c_str(), "w");
    if (!f) { printf("Could not write %s\n", outPath(name).c_str()); exit(1); }
    return f;
}

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

int cpuThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
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
    if (!f) { LOG("Could not write %s\n", path.c_str()); return; }
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
    if (!f) { LOG("Could not write %s\n", path.c_str()); return; }
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

double nowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Runs fn once as a warm-up (thrown away), then TIMED_RUNS times. Returns every timed run.
template <typename F>
std::vector<double> timeCPU(F fn) {
    for (int i = 0; i < WARMUP_RUNS; i++) fn();
    std::vector<double> t;
    for (int i = 0; i < TIMED_RUNS; i++) {
        double a = nowMs();
        fn();
        t.push_back(nowMs() - a);
    }
    return t;
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------
#ifdef __CUDACC__

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    LOG("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// GPU version 1: lookup table in global memory, one thread per column.
__global__ void terrainKernelGlobal(int* out, int W, int H, int x0, int z0,
                                    TerrainParams p, const uint8_t* perm) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && z < H)                      // bounds check for partial blocks
        out[(size_t)z * W + x] = heightAt(x0, z0, x, z, p, perm);
}

// Tiny kernel used once to check that --fmad=false is really on (see checkFmad).
__global__ void fmaProbeKernel(float a, float b, float c, float* out) {
    *out = a * b + c;
}

// GPU buffers are allocated once (setup, not timed) and reused.
// d_out has CANARY_INTS extra values at the end for the overrun check.
struct GpuBuffers {
    int* d_out = nullptr;
    uint8_t* d_perm = nullptr;
    size_t capacity = 0;     // in columns
    void ensure(size_t columns) {
        if (!d_perm) CUDA_CHECK(cudaMalloc((void**)&d_perm, 512));
        if (columns <= capacity) return;
        if (d_out) CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaMalloc((void**)&d_out, (columns + CANARY_INTS) * sizeof(int)));
        capacity = columns;
    }
    void release() {
        if (d_out) cudaFree(d_out);
        if (d_perm) cudaFree(d_perm);
        d_out = nullptr; d_perm = nullptr; capacity = 0;
    }
};

struct Gpu {
    GpuBuffers buf;
    cudaEvent_t ev[4];
    double setupMs = 0;
};

inline dim3 gridFor(int W, int H, Block b) {
    return dim3((W + b.x - 1) / b.x, (H + b.y - 1) / b.y);   // round up
}

// The one place that knows which kernel belongs to which version.
void launchTerrain(int variant, dim3 grid, dim3 block, GpuBuffers& buf,
                   int W, int H, int x0, int z0, const TerrainParams& p) {
    switch (variant) {
        case GPU_GLOBAL:
            terrainKernelGlobal<<<grid, block>>>(buf.d_out, W, H, x0, z0, p, buf.d_perm);
            break;
        // Week 13: case GPU_CONSTANT: ...   case GPU_SHARED: ...
        default:
            LOG("Unknown GPU version %d\n", variant); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

struct GpuTiming { float h2dMs, kernelMs, d2hMs, totalMs; };

// One full GPU run: copy table in -> kernel -> copy heights back.
//   h2dMs    = copying the 512-byte table to the GPU
//   kernelMs = the GPU math only
//   d2hMs    = copying the whole height array back (W * H * 4 bytes)
//   totalMs  = all three together
GpuTiming gpuGenerate(Gpu& g, int variant, Block b, int W, int H, int x0, int z0,
                      const TerrainParams& p, const uint8_t* perm, int* hostOut) {
    size_t n = (size_t)W * H;
    g.buf.ensure(n);
    dim3 block(b.x, b.y);
    dim3 grid = gridFor(W, H, b);

    CUDA_CHECK(cudaEventRecord(g.ev[0]));
    CUDA_CHECK(cudaMemcpy(g.buf.d_perm, perm, 512, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(g.ev[1]));
    launchTerrain(variant, grid, block, g.buf, W, H, x0, z0, p);
    CUDA_CHECK(cudaEventRecord(g.ev[2]));
    CUDA_CHECK(cudaMemcpy(hostOut, g.buf.d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(g.ev[3]));
    CUDA_CHECK(cudaEventSynchronize(g.ev[3]));

    GpuTiming t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2dMs,    g.ev[0], g.ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernelMs, g.ev[1], g.ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2hMs,    g.ev[2], g.ev[3]));
    CUDA_CHECK(cudaEventElapsedTime(&t.totalMs,  g.ev[0], g.ev[3]));
    return t;
}

// Test helpers. poison() fills the height array and the guard values after it
// with a marker, so a column the kernel skipped shows up as a huge mismatch.
// canaryIntact() is true if the kernel wrote nothing past the end of the array.
void poison(GpuBuffers& buf, size_t n) {
    buf.ensure(n);
    CUDA_CHECK(cudaMemset(buf.d_out, 0x7F, (n + CANARY_INTS) * sizeof(int)));
}
bool canaryIntact(GpuBuffers& buf, size_t n) {
    int guard[CANARY_INTS];
    CUDA_CHECK(cudaMemcpy(guard, buf.d_out + n, sizeof(guard), cudaMemcpyDeviceToHost));
    for (int v : guard) if (v != POISON) return false;
    return true;
}

// Keeps the GPU busy for ~0.5 s so it leaves power-saving mode and reaches
// its full clock speed before we time anything (the GPU slows down while it
// sits idle during the CPU runs). Not timed.
void warmUpGPU(Gpu& g, const TerrainParams& p, const uint8_t* perm) {
    const int W = 4096, H = 4096;
    g.buf.ensure((size_t)W * H);
    CUDA_CHECK(cudaMemcpy(g.buf.d_perm, perm, 512, cudaMemcpyHostToDevice));
    dim3 block(BASELINE_BLOCK.x, BASELINE_BLOCK.y);
    dim3 grid = gridFor(W, H, BASELINE_BLOCK);
    double start = nowMs();
    while (nowMs() - start < 500.0) {
        launchTerrain(GPU_GLOBAL, grid, block, g.buf, W, H, 0, 0, p);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// One-time setup (GPU start-up + allocating the largest buffer), reported separately.
void gpuInit(Gpu& g) {
    double s0 = nowMs();
    CUDA_CHECK(cudaFree(0));
    g.buf.ensure((size_t)4097 * 4097);
    for (auto& e : g.ev) CUDA_CHECK(cudaEventCreate(&e));
    g.setupMs = nowMs() - s0;
}
void gpuShutdown(Gpu& g) {
    for (auto& e : g.ev) cudaEventDestroy(e);
    g.buf.release();
}

// Checks on the real GPU whether a*b+c is being merged into one fused
// multiply-add. With --fmad=false the answer is exactly 0; with FMA it is not.
// Returns 1 = merged (wrong flag), 0 = not merged (what we want).
int gpuFmaContracts() {
    volatile float a = 1.0f + 1.0f / 4096.0f;       // 1 + 2^-12
    volatile float c = -(1.0f + 1.0f / 2048.0f);    // -(1 + 2^-11)
    float* d = nullptr; float r = 0;
    CUDA_CHECK(cudaMalloc((void**)&d, sizeof(float)));
    fmaProbeKernel<<<1, 1>>>(a, a, c, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&r, d, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
    return r != 0.0f;
}
#else
struct Gpu {};
#endif

int cpuFmaContracts() {
    volatile float a = 1.0f + 1.0f / 4096.0f;
    volatile float c = -(1.0f + 1.0f / 2048.0f);
    float x = a, y = a, z = c;
    volatile float r = x * y + z;
    return r != 0.0f;
}

// ---------------------------------------------------------------------------
// environment.txt: what PC and software produced these results
// ---------------------------------------------------------------------------
std::string cpuName() {
    char brand[49] = {0};
#if defined(_MSC_VER)
    int r[4];
    for (int i = 0; i < 3; i++) { __cpuid(r, (int)(0x80000002u + i)); memcpy(brand + 16 * i, r, 16); }
#elif defined(__x86_64__) || defined(__i386__)
    unsigned r[4];
    for (int i = 0; i < 3; i++) {
        __get_cpuid(0x80000002u + i, &r[0], &r[1], &r[2], &r[3]);
        memcpy(brand + 16 * i, r, 16);
    }
#endif
    std::string s = brand;
    size_t a = s.find_first_not_of(' ');
    return a == std::string::npos ? "unknown" : s.substr(a);
}

// Writes one line to both environment.txt and the screen/log.
static FILE* g_envFile = nullptr;
void both(const char* fmt, ...) {
    char line[512];
    va_list a;
    va_start(a, fmt); vsnprintf(line, sizeof(line), fmt, a); va_end(a);
    if (g_envFile) fputs(line, g_envFile);
    LOG("%s", line);
}

void writeEnvironment(const std::string& mode, Gpu& gpu) {
    g_envFile = openOut("environment.txt");
    time_t now = time(nullptr);
    char when[64];
    strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", localtime(&now));
    both("=== Environment ===\n");
    both("Run started        : %s (mode: %s)\n", when, mode.c_str());
    both("CPU                : %s\n", cpuName().c_str());
    both("CPU logical cores  : %u\n", std::thread::hardware_concurrency());
#ifdef _OPENMP
    both("OpenMP             : on, %d threads for the all-cores CPU version\n", omp_get_max_threads());
#else
    both("OpenMP             : OFF (all-cores CPU version runs on 1 core - see SETUP.md)\n");
#endif
#if defined(_MSC_FULL_VER)
    both("Host compiler      : MSVC %d (_MSC_VER %d)\n", _MSC_FULL_VER, _MSC_VER);
#elif defined(__VERSION__)
    both("Host compiler      : %s\n", __VERSION__);
#endif
#ifdef NDEBUG
    both("Build              : Release (optimized)\n");
#else
    both("Build              : DEBUG - timings are not valid, rebuild in Release\n");
#endif
#ifdef __CUDACC__
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0, drv = 0, cma = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, 0);
    both("GPU                : %s\n", prop.name);
    both("GPU details        : compute capability %d.%d, %d SMs, %.1f GB\n", prop.major, prop.minor,
         prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    both("CUDA toolkit       : %d.%d (compiled with), runtime %d.%d, driver supports CUDA %d.%d\n",
         CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10, rt / 1000, (rt % 1000) / 10,
         drv / 1000, (drv % 1000) / 10);
    both("Managed prefetch   : %s (concurrent managed access)\n", cma ? "available" : "not available on this system");
    int gf = gpuFmaContracts();
    both("GPU a*b+c merging  : %s\n", gf ? "ON - WRONG, rebuild with --fmad=false (see SETUP.md)"
                                         : "off (--fmad=false is working)");
    both("GPU one-time setup : %.2f ms (start-up + cudaMalloc, not part of any timing)\n", gpu.setupMs);
#else
    (void)gpu;
    both("GPU                : none (CPU-only build, no CUDA)\n");
#endif
    both("CPU a*b+c merging  : %s\n", cpuFmaContracts() ? "ON - CPU and GPU may round differently" : "off");
    both("Math               : 32-bit float on CPU and GPU, same functions (terrain.h)\n");
    both("Seeds              : %llu (main), %llu (second test seed)\n",
         (unsigned long long)SEED_MAIN, (unsigned long long)SEED_TEST);
    both("Timing             : %d warm-up run (thrown away) + %d timed runs per case\n", WARMUP_RUNS, TIMED_RUNS);
    both("Baseline block     : %d x %d (%d threads per block)\n", BASELINE_BLOCK.x, BASELINE_BLOCK.y,
         BASELINE_BLOCK.x * BASELINE_BLOCK.y);
    both("Speedup            : CPU median time / GPU median time\n");
    both("GPU total time     : copy table in + kernel + copy heights back\n");
    fclose(g_envFile);
    g_envFile = nullptr;
}

// ---------------------------------------------------------------------------
// Mode: maps
// ---------------------------------------------------------------------------
void runMaps(Gpu& gpu) {
    const int W = 1024, H = 1024;
    LOG("\n=== Heightmaps (%dx%d, seed %llu, origin (0,0)) ===\n", W, H, (unsigned long long)SEED_MAIN);
    uint8_t perm[512];
    buildPermutation(SEED_MAIN, perm);
    std::vector<int> h((size_t)W * H), ref((size_t)W * H);
    FILE* csv = openOut("stats.csv");
    fprintf(csv, "type,octaves,seed,width,height,made_by,matches_cpu,hmin,hmax,hsum,havg,"
                 "dmax,davg,dmax_limit,smooth_pass,clamped\n");
    LOG("%-10s %-11s %6s %6s %8s %6s %7s %7s %8s\n", "type", "made by", "min", "max", "avg",
        "dmax", "limit", "davg", "clamped");
    for (int type = 0; type < 3; type++) {
        TerrainParams p = makePreset(type);
        cpuGenerate(W, H, 0, 0, p, perm, ref.data());
        const char* madeBy = "cpu_1core";
        int matches = 1;
#ifdef __CUDACC__
        gpuGenerate(gpu, GPU_GLOBAL, BASELINE_BLOCK, W, H, 0, 0, p, perm, h.data());
        madeBy = variantName(GPU_GLOBAL);
        matches = (h == ref);
#else
        (void)gpu;
        h = ref;
#endif
        Stats s = computeStats(h.data(), W, H);
        int limit = smoothnessLimit(type);
        LOG("%-10s %-11s %6d %6d %8.2f %6d %7d %7.3f %8lld%s\n", typeName(type), madeBy, s.hmin, s.hmax,
            s.havg, s.dmax, limit, s.davg, s.clamped, matches ? "" : "   <-- differs from CPU!");
        fprintf(csv, "%s,%d,%llu,%d,%d,%s,%d,%d,%d,%lld,%.4f,%d,%.6f,%d,%d,%lld\n", typeName(type), p.octaves,
                (unsigned long long)SEED_MAIN, W, H, madeBy, matches, s.hmin, s.hmax, s.hsum, s.havg,
                s.dmax, s.davg, limit, s.dmax <= limit, s.clamped);
        std::string base = outPath(std::string(typeName(type)) + "_" +
                                   std::to_string(W) + "x" + std::to_string(H));
        saveHeightsBin(base + ".bin", h.data(), W, H);
        saveGrayBMP(base + ".bmp", h.data(), W, H);
    }
    fclose(csv);
    LOG("dmax = biggest height step between neighboring columns; limit = our smoothness target.\n");
    LOG("Saved stats.csv and <type>_%dx%d.bin / .bmp in %s\n", W, H, g_outDir.c_str());
}

// ---------------------------------------------------------------------------
// Mode: test
// ---------------------------------------------------------------------------
void runTests(Gpu& gpu) {
#ifndef __CUDACC__
    (void)gpu;
    LOG("\n[test] Needs the CUDA build. Skipped.\n");
#else
    LOG("\n=== Correctness tests: CPU vs. GPU ===\n");
    LOG("Each case: heights must match the CPU (max difference <= 1, <= 0.01%% of columns),\n");
    LOG("two GPU runs must be identical, nothing may be written outside the array,\n");
    LOG("and the biggest neighbor step (dmax) must be within the terrain type's limit.\n\n");
    FILE* csv = openOut("correctness.csv");
    fprintf(csv, "width,height,columns,seed,x0,z0,type,version,block_x,block_y,grid_x,grid_y,"
                 "threads_launched,idle_threads,max_diff,mismatches,mismatch_pct,"
                 "gpu_repeat_identical,no_overrun,hmin,hmax,havg,dmax,dmax_limit,clamped,"
                 "match_pass,smooth_pass,pass\n");
    FILE* sum = openOut("correctness_summary.csv");
    fprintf(sum, "width,height,columns,block_x,block_y,grid_x,grid_y,threads_launched,idle_threads,"
                 "cases,passed,worst_max_diff,total_mismatches,worst_dmax_plains,worst_dmax_hilly,"
                 "worst_dmax_mountains\n");
    LOG("%11s %7s %9s %12s %12s %7s %9s %10s %s\n", "size", "block", "grid", "threads", "idle",
        "cases", "passed", "max diff", "worst dmax (plains/hilly/mountains)");
    uint64_t seeds[] = {SEED_MAIN, SEED_TEST};
    int total = 0, passed = 0;
    long long allMism = 0; int allMaxDiff = 0;
    for (auto sz : EDGE_SIZES) {
        size_t n = (size_t)sz.w * sz.h;
        std::vector<int> cpu(n), gpuOut(n), gpuOut2(n);
        for (Block b : TEST_BLOCKS) {
            dim3 grid = gridFor(sz.w, sz.h, b);
            long long launched = (long long)grid.x * grid.y * b.x * b.y;
            int cases = 0, ok_ = 0, worstDiff = 0, worstD[3] = {0, 0, 0};
            long long mismSum = 0;
            for (uint64_t seed : seeds) {
                uint8_t perm[512];
                buildPermutation(seed, perm);
                for (auto o : ORIGINS) {
                    for (int type = 0; type < 3; type++) {
                        TerrainParams p = makePreset(type);
                        cpuGenerate(sz.w, sz.h, o.x0, o.z0, p, perm, cpu.data());
                        for (int variant : TEST_VARIANTS) {
                            poison(gpu.buf, n);
                            gpuGenerate(gpu, variant, b, sz.w, sz.h, o.x0, o.z0, p, perm, gpuOut.data());
                            bool noOverrun = canaryIntact(gpu.buf, n);
                            poison(gpu.buf, n);
                            gpuGenerate(gpu, variant, b, sz.w, sz.h, o.x0, o.z0, p, perm, gpuOut2.data());
                            noOverrun = noOverrun && canaryIntact(gpu.buf, n);

                            long long maxDiff = 0, mism = 0;
                            for (size_t i = 0; i < n; i++) {
                                long long d = std::llabs((long long)cpu[i] - (long long)gpuOut[i]);
                                if (d) { mism++; maxDiff = std::max(maxDiff, d); }
                            }
                            bool same = (gpuOut == gpuOut2);
                            double pct = 100.0 * mism / n;
                            Stats s = computeStats(gpuOut.data(), sz.w, sz.h);
                            int limit = smoothnessLimit(type);
                            bool matchOk = maxDiff <= 1 && pct <= 0.01 && same && noOverrun;
                            bool smoothOk = s.dmax <= limit;
                            bool ok = matchOk && smoothOk;
                            total++; passed += ok; cases++; ok_ += ok;
                            mismSum += mism; allMism += mism;
                            worstDiff = (int)std::max<long long>(worstDiff, std::min<long long>(maxDiff, 1000000));
                            allMaxDiff = std::max(allMaxDiff, worstDiff);
                            worstD[type] = std::max(worstD[type], s.dmax);
                            fprintf(csv, "%d,%d,%llu,%llu,%d,%d,%s,%s,%d,%d,%u,%u,%lld,%lld,%lld,%lld,%.6f,"
                                         "%d,%d,%d,%d,%.4f,%d,%d,%lld,%d,%d,%d\n",
                                    sz.w, sz.h, (unsigned long long)n, (unsigned long long)seed, o.x0, o.z0,
                                    typeName(type), variantName(variant), b.x, b.y, grid.x, grid.y,
                                    launched, launched - (long long)n, maxDiff, mism, pct,
                                    same, noOverrun, s.hmin, s.hmax, s.havg, s.dmax, limit, s.clamped,
                                    matchOk, smoothOk, ok);
                            if (!ok)
                                LOG("  FAIL %dx%d seed=%llu origin=(%d,%d) %s %s: max_diff=%lld mismatches=%lld "
                                    "repeat_same=%d no_overrun=%d dmax=%d (limit %d)\n",
                                    sz.w, sz.h, (unsigned long long)seed, o.x0, o.z0, typeName(type),
                                    variantName(variant), maxDiff, mism, same, noOverrun, s.dmax, limit);
                        }
                    }
                }
            }
            char sizeStr[32], blockStr[16], gridStr[24], dStr[32];
            snprintf(sizeStr, sizeof(sizeStr), "%d x %d", sz.w, sz.h);
            snprintf(blockStr, sizeof(blockStr), "%dx%d", b.x, b.y);
            snprintf(gridStr, sizeof(gridStr), "%ux%u", grid.x, grid.y);
            snprintf(dStr, sizeof(dStr), "%d / %d / %d", worstD[0], worstD[1], worstD[2]);
            LOG("%11s %7s %9s %12lld %12lld %7d %9d %10d %s\n", sizeStr, blockStr, gridStr, launched,
                launched - (long long)n, cases, ok_, worstDiff, dStr);
            fprintf(sum, "%d,%d,%llu,%d,%d,%u,%u,%lld,%lld,%d,%d,%d,%lld,%d,%d,%d\n", sz.w, sz.h,
                    (unsigned long long)n, b.x, b.y, grid.x, grid.y, launched, launched - (long long)n,
                    cases, ok_, worstDiff, mismSum, worstD[0], worstD[1], worstD[2]);
            fflush(csv); fflush(sum);
        }
    }
    fclose(csv); fclose(sum);
    LOG("\nPassed %d / %d cases. Largest CPU-GPU height difference: %d. Columns that differ: %lld.\n",
        passed, total, allMaxDiff, allMism);
    LOG("\"idle\" = threads that were launched but had no column (stopped by the bounds check).\n");
    LOG("Saved correctness.csv (every case) and correctness_summary.csv (one row per size).\n");
#endif
}

// ---------------------------------------------------------------------------
// Benchmarks. One "case" = one terrain setting + world size + GPU version + block size.
// Each mode below just builds a list of cases and hands it to runExperiment.
// ---------------------------------------------------------------------------
struct Case {
    int type;        // PLAINS / HILLY / MOUNTAINS
    int octaves;     // usually the type's default
    Size sz;
    int variant;     // GpuVariant
    Block block;
};

struct CpuTimes { TimeStats one, all; };

// CPU timings are remembered, so a case that only changes a GPU setting
// (block size, memory type) does not time the CPU again.
static std::map<std::string, CpuTimes> g_cpuCache;

const char* BENCH_HEADER =
    "experiment,version,type,octaves,width,height,columns,block_x,block_y,threads_per_block,cpu_threads,runs,"
    "median_ms,avg_ms,min_ms,sd_ms,"
    "kernel_median_ms,kernel_avg_ms,kernel_min_ms,kernel_sd_ms,"
    "h2d_median_ms,d2h_median_ms,transfer_median_ms,transfer_pct,bytes_copied_back,"
    "speedup_vs_1core_total,speedup_vs_1core_kernel,speedup_vs_allcores_total,speedup_vs_allcores_kernel\n";
const char* RAW_HEADER =
    "experiment,version,type,octaves,width,height,block_x,block_y,run,total_ms,kernel_ms,h2d_ms,d2h_ms\n";

void runExperiment(const std::string& name, const std::vector<Case>& cases, Gpu& gpu) {
    FILE* csv = openOut("benchmark_" + name + ".csv");
    FILE* raw = openOut("raw_" + name + ".csv");
    fputs(BENCH_HEADER, csv);
    fputs(RAW_HEADER, raw);
    uint8_t perm[512];
    buildPermutation(SEED_MAIN, perm);
    int threads = cpuThreads();
    std::set<std::string> cpuRowsWritten;

    LOG("%-9s %3s %11s %6s %-11s %11s %11s %9s %10s %10s %10s %8s %8s %9s %7s\n", "type", "oct", "size",
        "block", "version", "CPU 1 core", "CPU all", "copy in", "kernel", "copy back", "GPU total",
        "x 1core", "x all", "x kernel", "copy %");

    for (const Case& c : cases) {
        TerrainParams p = makePreset(c.type);
        p.octaves = c.octaves;
        size_t n = (size_t)c.sz.w * c.sz.h;
        std::vector<int> out(n);
        char key[96];
        snprintf(key, sizeof(key), "%d|%d|%d|%d", c.type, c.octaves, c.sz.w, c.sz.h);

        // --- CPU (timed once per terrain setting + size) ---
        if (!g_cpuCache.count(key)) {
            std::vector<double> t1 = timeCPU([&] { cpuGenerate(c.sz.w, c.sz.h, 0, 0, p, perm, out.data()); });
            std::vector<double> tN = timeCPU([&] { cpuGenerateParallel(c.sz.w, c.sz.h, 0, 0, p, perm, out.data()); });
            g_cpuCache[key] = {summarize(t1), summarize(tN)};
            for (int i = 0; i < TIMED_RUNS; i++)
                fprintf(raw, "%s,cpu_1core,%s,%d,%d,%d,0,0,%d,%.6f,,,\n", name.c_str(), typeName(c.type),
                        c.octaves, c.sz.w, c.sz.h, i + 1, t1[i]);
            for (int i = 0; i < TIMED_RUNS; i++)
                fprintf(raw, "%s,cpu_all_cores,%s,%d,%d,%d,0,0,%d,%.6f,,,\n", name.c_str(), typeName(c.type),
                        c.octaves, c.sz.w, c.sz.h, i + 1, tN[i]);
        }
        const CpuTimes& ct = g_cpuCache[key];
        if (!cpuRowsWritten.count(key)) {
            cpuRowsWritten.insert(key);
            fprintf(csv, "%s,cpu_1core,%s,%d,%d,%d,%llu,0,0,0,1,%d,%.6f,%.6f,%.6f,%.6f,,,,,,,,,,1.0000,,%.4f,\n",
                    name.c_str(), typeName(c.type), c.octaves, c.sz.w, c.sz.h, (unsigned long long)n,
                    TIMED_RUNS, ct.one.med, ct.one.avg, ct.one.mn, ct.one.sd, ct.all.med / ct.one.med);
            fprintf(csv, "%s,cpu_all_cores,%s,%d,%d,%d,%llu,0,0,0,%d,%d,%.6f,%.6f,%.6f,%.6f,,,,,,,,,,%.4f,,1.0000,\n",
                    name.c_str(), typeName(c.type), c.octaves, c.sz.w, c.sz.h, (unsigned long long)n,
                    threads, TIMED_RUNS, ct.all.med, ct.all.avg, ct.all.mn, ct.all.sd, ct.one.med / ct.all.med);
        }

        char sizeStr[32], blockStr[16];
        snprintf(sizeStr, sizeof(sizeStr), "%d x %d", c.sz.w, c.sz.h);
        snprintf(blockStr, sizeof(blockStr), "%dx%d", c.block.x, c.block.y);
#ifdef __CUDACC__
        // --- GPU ---
        warmUpGPU(gpu, p, perm);   // GPU idled during the CPU runs, wake it up
        for (int i = 0; i < WARMUP_RUNS; i++)
            gpuGenerate(gpu, c.variant, c.block, c.sz.w, c.sz.h, 0, 0, p, perm, out.data());
        std::vector<double> tk, tt, th, td, tx;
        for (int i = 0; i < TIMED_RUNS; i++) {
            GpuTiming g = gpuGenerate(gpu, c.variant, c.block, c.sz.w, c.sz.h, 0, 0, p, perm, out.data());
            tk.push_back(g.kernelMs); tt.push_back(g.totalMs);
            th.push_back(g.h2dMs);    td.push_back(g.d2hMs);
            tx.push_back(g.h2dMs + g.d2hMs);
            fprintf(raw, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n", name.c_str(),
                    variantName(c.variant), typeName(c.type), c.octaves, c.sz.w, c.sz.h, c.block.x,
                    c.block.y, i + 1, g.totalMs, g.kernelMs, g.h2dMs, g.d2hMs);
        }
        TimeStats gk = summarize(tk), gt = summarize(tt), gh = summarize(th), gd = summarize(td),
                  gx = summarize(tx);
        double copyPct = 100.0 * gx.med / gt.med;
        fprintf(csv, "%s,%s,%s,%d,%d,%d,%llu,%d,%d,%d,0,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                     "%.6f,%.6f,%.6f,%.2f,%llu,%.4f,%.4f,%.4f,%.4f\n",
                name.c_str(), variantName(c.variant), typeName(c.type), c.octaves, c.sz.w, c.sz.h,
                (unsigned long long)n, c.block.x, c.block.y, c.block.x * c.block.y, TIMED_RUNS,
                gt.med, gt.avg, gt.mn, gt.sd, gk.med, gk.avg, gk.mn, gk.sd,
                gh.med, gd.med, gx.med, copyPct, (unsigned long long)(n * sizeof(int)),
                ct.one.med / gt.med, ct.one.med / gk.med, ct.all.med / gt.med, ct.all.med / gk.med);
        LOG("%-9s %3d %11s %6s %-11s %8.3f ms %8.3f ms %6.3f ms %7.4f ms %7.3f ms %7.3f ms %7.1fx %7.1fx %8.1fx %6.1f%%\n",
            typeName(c.type), c.octaves, sizeStr, blockStr, variantName(c.variant), ct.one.med, ct.all.med,
            gh.med, gk.med, gd.med, gt.med, ct.one.med / gt.med, ct.all.med / gt.med,
            ct.one.med / gk.med, copyPct);
#else
        (void)gpu;
        LOG("%-9s %3d %11s %6s %-11s %8.3f ms %8.3f ms\n", typeName(c.type), c.octaves, sizeStr, blockStr,
            "(no GPU)", ct.one.med, ct.all.med);
#endif
        fflush(csv); fflush(raw);
    }
    fclose(csv); fclose(raw);
    LOG("Times are medians of %d runs. \"x 1core\" and \"x all\" = CPU time / GPU total time (copies included).\n",
        TIMED_RUNS);
    LOG("\"x kernel\" = CPU 1 core / GPU kernel only. \"copy %%\" = share of GPU total spent copying.\n");
    LOG("CPU all-cores threads: %d%s\n", threads,
        threads == 1 ? " (OpenMP is off - see SETUP.md to turn it on)" : "");
    LOG("Saved benchmark_%s.csv (summary) and raw_%s.csv (every single run).\n", name.c_str(), name.c_str());
}

// RQ1 + RQ4: every terrain type on every benchmark size, baseline settings.
void runBenchSizes(Gpu& gpu) {
    LOG("\n=== Benchmark: world size (RQ1, RQ4) - seed %llu, origin (0,0) ===\n", (unsigned long long)SEED_MAIN);
    std::vector<Case> cases;
    for (int type : {PLAINS, HILLY, MOUNTAINS})
        for (auto sz : BENCH_SIZES)
            for (int v : BENCH_VARIANTS)
                cases.push_back({type, makePreset(type).octaves, sz, v, BASELINE_BLOCK});
    runExperiment("sizes", cases, gpu);
}

// RQ2: more octaves = more work per column. Hilly type, one size.
void runBenchOctaves(Gpu& gpu) {
    LOG("\n=== Benchmark: octaves (RQ2) - hilly, %d x %d ===\n", OCTAVE_SWEEP_SIZE.w, OCTAVE_SWEEP_SIZE.h);
    std::vector<Case> cases;
    for (int oct : OCTAVE_SWEEP)
        for (int v : BENCH_VARIANTS)
            cases.push_back({HILLY, oct, OCTAVE_SWEEP_SIZE, v, BASELINE_BLOCK});
    runExperiment("octaves", cases, gpu);
}

// RQ3 (first half): threads per block. Only the GPU changes; CPU times are reused.
void runBenchBlocks(Gpu& gpu) {
    LOG("\n=== Benchmark: threads per block (RQ3) - hilly ===\n");
    std::vector<Case> cases;
    for (auto sz : BLOCK_SWEEP_SIZES)
        for (Block b : BLOCK_SWEEP)
            for (int v : BENCH_VARIANTS)
                cases.push_back({HILLY, makePreset(HILLY).octaves, sz, v, b});
    runExperiment("blocks", cases, gpu);
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "all";
    const char* known[] = {"all", "maps", "test", "bench", "octaves", "blocks"};
    bool ok = false;
    for (const char* k : known) ok = ok || mode == k;
    if (!ok) {
        printf("Unknown mode \"%s\". Use one of: maps, test, bench, octaves, blocks (or nothing for all).\n",
               mode.c_str());
        return 1;
    }
    findOutputDir();
    g_log = fopen(outPath("log_" + mode + ".txt").c_str(), "w");

    Gpu gpu;
#ifdef __CUDACC__
    gpuInit(gpu);
#endif
    writeEnvironment(mode, gpu);
    LOG("Results folder     : %s\n", g_outDir.c_str());

    if (mode == "maps"    || mode == "all") runMaps(gpu);
    if (mode == "test"    || mode == "all") runTests(gpu);
    if (mode == "bench"   || mode == "all") runBenchSizes(gpu);
    if (mode == "octaves" || mode == "all") runBenchOctaves(gpu);
    if (mode == "blocks"  || mode == "all") runBenchBlocks(gpu);

#ifdef __CUDACC__
    gpuShutdown(gpu);
#endif
    LOG("\nDone. Next: run  py viewer\\plots.py  to make the graphs, then commit the results folder.\n");
    if (g_log) fclose(g_log);
    return 0;
}
