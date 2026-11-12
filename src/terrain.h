// terrain.h - shared terrain math for the CPU and GPU versions.
// Every function marked HD is compiled for both the CPU and the GPU, so both
// versions run exactly the same formulas (README section 2, parts A to D).
#pragma once
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Terrain type settings (README table in section C).
struct TerrainParams {
    float B;      // base height (blocks)
    float A;      // amplitude (blocks)
    float f0;     // base frequency
    int octaves;  // number of noise layers
    int ridged;   // 1 = ridged noise (mountains)
};

enum TerrainType { PLAINS = 0, HILLY = 1, MOUNTAINS = 2 };

inline const char* typeName(int t) {
    return t == PLAINS ? "plains" : t == HILLY ? "hilly" : "mountains";
}

inline TerrainParams makePreset(int type) {
    switch (type) {
        case PLAINS:    return {64.0f,  8.0f, 0.004f, 2, 0};
        case HILLY:     return {72.0f, 24.0f, 0.008f, 4, 0};
        default:        return {100.0f, 90.0f, 0.004f, 6, 1};
    }
}

// Smoothness target: largest allowed height step between two neighboring
// columns, checked at each type's default octaves (README section C).
inline int smoothnessLimit(int type) {
    return type == PLAINS ? 2 : type == HILLY ? 4 : 8;
}

const int HEIGHT_MIN = -64;
const int HEIGHT_MAX = 319;

// ---------------------------------------------------------------------------
// A. Permutation table (CPU only, built once from the seed). Defined in main.
// ---------------------------------------------------------------------------
void buildPermutation(uint64_t seed, uint8_t perm[512]);

// ---------------------------------------------------------------------------
// B. 2D Perlin noise (improved Perlin, 2002, adapted to 2D)
// ---------------------------------------------------------------------------
HD inline float fade(float t) {
    // 6t^5 - 15t^4 + 10t^3, written in nested form
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

HD inline float lerpf(float t, float a, float b) {
    return a + t * (b - a);
}

HD inline float grad(int h, float x, float z) {
    switch (h & 7) {
        case 0:  return  x;
        case 1:  return -x;
        case 2:  return  z;
        case 3:  return -z;
        case 4:  return  x + z;
        case 5:  return -x + z;
        case 6:  return  x - z;
        default: return -x - z;
    }
}

HD inline float noise2D(float x, float z, const uint8_t* perm) {
    float fx = floorf(x), fz = floorf(z);
    int xi = (int)fx, zi = (int)fz;
    int X = xi & 255, Z = zi & 255;
    float xf = x - fx, zf = z - fz;
    float u = fade(xf), v = fade(zf);
    int aa = perm[perm[X] + Z];
    int ba = perm[perm[X + 1] + Z];
    int ab = perm[perm[X] + Z + 1];
    int bb = perm[perm[X + 1] + Z + 1];
    return lerpf(v,
                 lerpf(u, grad(aa, xf, zf),        grad(ba, xf - 1.0f, zf)),
                 lerpf(u, grad(ab, xf, zf - 1.0f), grad(bb, xf - 1.0f, zf - 1.0f)));
}

// ---------------------------------------------------------------------------
// C. Octaves (fBm) and height
// ---------------------------------------------------------------------------
HD inline float fbm(float xw, float zw, const TerrainParams& p, const uint8_t* perm) {
    float sum = 0.0f, amp = 1.0f, freq = p.f0, ampTotal = 0.0f;
    for (int o = 0; o < p.octaves; o++) {
        float n = noise2D(xw * freq + 17.37f * (float)o,
                          zw * freq + 31.91f * (float)o, perm);
        if (p.ridged) n = 1.0f - 2.0f * fabsf(n);
        sum += amp * n;
        ampTotal += amp;          // = 2 * (1 - 0.5^N) after the loop
        freq *= 2.0f;
        amp  *= 0.5f;
    }
    return sum / ampTotal;        // same as sum * invNorm, no powf needed
}

// Height of column (x, z) in a world whose corner is at (x0, z0).
HD inline int heightAt(int x0, int z0, int x, int z,
                       const TerrainParams& p, const uint8_t* perm) {
    float xw = (float)(x0 + x);
    float zw = (float)(z0 + z);
    int h = (int)floorf(p.B + p.A * fbm(xw, zw, p, perm));
    if (h < HEIGHT_MIN) h = HEIGHT_MIN;
    if (h > HEIGHT_MAX) h = HEIGHT_MAX;
    return h;
}
