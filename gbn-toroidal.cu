/*
 * Optimize a point set to minimize Gaussian Blue Noise energy using
 * a gradient decent algorithm.
 * 2021-12-22: Created by Abdalla Ahmed
 * 2022-08-24: Revised for the paper
 */

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "getopt/getopt.h"
#include <cmath>
#include <string>
#include <signal.h>
#include "cairo/cairo.h"
#include <stdint.h>
#include <random>

#define M_PI 3.14159265359

#define PERIODS 5
#define MAX_DIM 32
#define MAX_HARMONICS 128

//#define USE_DOUBLE
#include "float.h"
#include "snapshot.h"

// Number of sigmas for a Gaussian to go below 1 significant bit:
#ifdef USE_DOUBLE
    #define MAX_DIST 8.5
#else
    #define MAX_DIST 5.8
#endif

// =============================================================================
// Data Structures
// =============================================================================

struct Vector {
    Float x, y;
};

typedef Vector Point;
typedef std::vector<Point> Points;
typedef std::vector<Float> Floats;

std::random_device rd;                                                          // Will be used to obtain a seed for the random number engine
std::mt19937 rnd(rd());

// =============================================================================
// Global parameters
// =============================================================================

int N;
Float sigma = 1;
Float scale = 0.25;
const int BS = 256;
const double res = 2.3283064365386962890625e-10;                                // 2 ^ -32
int w = 1024;                                                                   // Width of plotted frames
char *frameBaseName = NULL;
Float coverage = 0.2;
static bool interruptFlag = false;

// =============================================================================
// Optimization Kernels
// =============================================================================

// -----------------------------------------------------------------------------
// Nearest only: fastest
// -----------------------------------------------------------------------------

__global__
void gbn0(
    int N, Point *pIn, Point *pOut, Float ss2InvNeg, Float scale
) {                                                                             // Considering replicas, but in spatial domain
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Vector grad = {ZERO, ZERO};
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float x = pIn[i].x - pIn[j].x;
        if (x < -HALF) x += ONE;
        if (x >= HALF) x -= ONE;
        Float y = pIn[i].y - pIn[j].y;
        if (y < -HALF) y += ONE;
        if (y >= HALF) y -= ONE;
        Float g = expF(ss2InvNeg * (x * x + y * y));
        grad.x += x * g;
        grad.y += y * g;
    }
    Float x = pIn[i].x + scale * grad.x;
    Float y = pIn[i].y + scale * grad.y;
    if (x <  ZERO) x += ONE;
    if (x >=  ONE) x -= ONE;
    if (y <  ZERO) y += ONE;
    if (y >=  ONE) y -= ONE;
    pOut[i] = {x, y};
}

// -----------------------------------------------------------------------------
// Images on both sides
// -----------------------------------------------------------------------------

__global__
void gbn1(
    int N, Point *pIn, Point *pOut, Float ss2InvNeg, Float scale
) {                                                                             // Considering replicas, but in spatial domain
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Vector grad = {ZERO, ZERO};
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float xl = pIn[i].x - pIn[j].x;
        if (xl < ZERO) xl += ONE;
        Float xr = ONE - xl;
        Float yb = pIn[i].y - pIn[j].y;
        if (yb < ZERO) yb += ONE;
        Float yt = ONE - yb;
        Float gl = expF(ss2InvNeg * xl * xl);
        Float gr = expF(ss2InvNeg * xr * xr);
        Float gb = expF(ss2InvNeg * yb * yb);
        Float gt = expF(ss2InvNeg * yt * yt);
        grad.x += (gb + gt) * (xl * gl - xr * gr);
        grad.y += (gl + gr) * (yb * gb - yt * gt);
    }
    Float x = pIn[i].x + scale * grad.x;
    Float y = pIn[i].y + scale * grad.y;
    if (x <  ZERO) x += ONE;
    if (x >=  ONE) x -= ONE;
    if (y <  ZERO) y += ONE;
    if (y >=  ONE) y -= ONE;
    pOut[i] = {x, y};
}

// -----------------------------------------------------------------------------
// Per axis computation
// -----------------------------------------------------------------------------

template <int K>
__device__ inline void gAndGrad(
    Float x, Float &ss2InvNeg, Float &g, Float &grad
) {
    if (x < ZERO) x += ONE;
    g = ZERO;
    grad = ZERO;
    #pragma unroll
    for (int k = 0; k < 2 * K; k++) {
        Float dx = x + (k - K);
        Float f = expF(ss2InvNeg * dx * dx);
        g += f;
        grad += dx * f;
    }
}

// -----------------------------------------------------------------------------
// Template for multiple replicas
// -----------------------------------------------------------------------------

template <int K>
__global__
void gbn(
    int N, Point *pIn, Point *pOut, Float ss2InvNeg, Float scale
) {                                                                             // Considering replicas, but in spatial domain
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Vector grad = {ZERO, ZERO};
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float gx, gy, gradx, grady;
        gAndGrad<K>(pIn[i].x - pIn[j].x, ss2InvNeg, gx, gradx);
        gAndGrad<K>(pIn[i].y - pIn[j].y, ss2InvNeg, gy, grady);
        grad.x += gy * gradx;
        grad.y += gx * grady;
    }
    Float x = pIn[i].x + scale * grad.x;
    Float y = pIn[i].y + scale * grad.y;
    if (x <  ZERO) x += ONE;
    if (x >=  ONE) x -= ONE;
    if (y <  ZERO) y += ONE;
    if (y >=  ONE) y -= ONE;
    pOut[i] = {x, y};
}

// -----------------------------------------------------------------------------
// N-dimensional, Nearest only: fastest
// -----------------------------------------------------------------------------

__global__
void gbnND0(
    int N,
    int dims,
    Float *pIn, Float *pOut, Float ss2InvNeg, Float scale
) {                                                                             // high-dimension version
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Float grad[MAX_DIM];                                                        // Gradient
    for (int dim = 0; dim < dims; dim++) grad[dim] = ZERO;
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float g_dim_j[MAX_DIM];                                                     // Factors of the Gaussian
        Float grad_dim_j[MAX_DIM];                                                  // Gradient due to one point j
        for (int dim = 0; dim < dims; dim++) {
            Float x = pIn[i * dims + dim] - pIn[j * dims + dim];
            if (x < -HALF) x += ONE;
            if (x >= HALF) x -= ONE;
            Float g = expF(ss2InvNeg * x * x);
            g_dim_j[dim] = g;
            grad_dim_j[dim] = x * g;
        }
        for (int dim = 0; dim < dims; dim++) {
            Float product = grad_dim_j[dim];
            for (int l = 0; l < dims; l++) {
                if (l != dim) product *= g_dim_j[l];
            }
            grad[dim] += product;
        }
    }
    for (int dim = 0; dim < dims; dim++) {
        Float x = pIn[i * dims + dim] + scale * grad[dim];
        if (x <  0) x += 1;
        if (x >= 1) x -= 1;
        pOut[i * dims + dim] = x;
    }
}


// -----------------------------------------------------------------------------
// N-dimensional, Nearest only: fastest
// -----------------------------------------------------------------------------

template <int K>
__global__
void gbnND(
    int N,
    int dims,
    Float *pIn, Float *pOut, Float ss2InvNeg, Float scale
) {                                                                             // high-dimension version
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Float grad[MAX_DIM];                                                        // Gradient
    for (int dim = 0; dim < dims; dim++) grad[dim] = ZERO;
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float g_dim_j[MAX_DIM];                                                     // Factors of the Gaussian
        Float grad_dim_j[MAX_DIM];                                                  // Gradient due to one point j
        for (int dim = 0; dim < dims; dim++) {
            gAndGrad<K>(
                pIn[i * dims + dim] - pIn[j * dims + dim],
                ss2InvNeg, g_dim_j[dim], grad_dim_j[dim]
            );
        }
        for (int dim = 0; dim < dims; dim++) {
            Float product = grad_dim_j[dim];
            for (int l = 0; l < dims; l++) {
                if (l != dim) product *= g_dim_j[l];
            }
            grad[dim] += product;
        }
    }
    for (int dim = 0; dim < dims; dim++) {
        Float x = pIn[i * dims + dim] + scale * grad[dim];
        if (x <  0) x += 1;
        if (x >= 1) x -= 1;
        pOut[i * dims + dim] = x;
    }
}

// =============================================================================

__global__
void gbnHarmonics(
    int N,
    int dims,
    int harmonics,
    Float *pIn, Float *pOut, Float twoSigmaSq, Float scale
) {                                                                             // Using Fourier series: θ_3(πx, exp(-2*0.5^2*π^2)) = 1 + 2 * sum_{k=1}^{\inf} (exp(-2*0.5^2*π^2))^k^2 cos(2πx)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Float g_dim_j[MAX_DIM];                                                     // Factors of the summed Fourier series
    Float grad[MAX_DIM];                                                        // Gradient
    Float grad_dim_j[MAX_DIM];                                                  // Gradient due to one point j
    Float wt[MAX_HARMONICS + 1];                                                // Weights of cosine terms
    wt[0] = 1;                                                                  // DC
    Float sum(wt[0]);
    for (int f = 1; f <= harmonics; f++) {
        sum += wt[f] = 2 * expF(-twoSigmaSq * f * f * M_PI * M_PI);
    }
    Float inv = 1.f / sum;
    for (int f = 0; f <= harmonics; f++) {                                      // Normalize so that amplitude is one, as in the normal implementation
        wt[f] *= inv;
    }
    for (int dim = 0; dim < dims; dim++) grad[dim] = 0.0;
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        for (int dim = 0; dim < dims; dim++) {
            Float dx = pIn[i * dims + dim] - pIn[j * dims + dim];
            if (dx < 0) dx += 1;
            g_dim_j[dim] = wt[0];                                                // Initialize field due to point j
            grad_dim_j[dim] = 0.0;                                                  // ~ gradient. Note that the fixed term is dropped by differentiation
            for (int f = 1; f <= harmonics; f++) {
                Float a = f * 2 * M_PI;
                Float s, c;
                sincosF(a * dx, &s, &c);
                g_dim_j[dim] += wt[f] * c;
                grad_dim_j[dim] += a * wt[f] * s;
            }
        }
        for (int dim = 0; dim < dims; dim++) {
            Float product = 1;
            for (int l = 0; l < dims; l++) {
                product *= (l == dim) ? grad_dim_j[l] : g_dim_j[l];
            }
            grad[dim] += product;
        }
    }
    for (int dim = 0; dim < dims; dim++) {
        Float x = pIn[i * dims + dim] + scale * grad[dim];
        if (x <  0) x += 1;
        if (x >= 1) x -= 1;
        pOut[i * dims + dim] = x;
    }
}

// =============================================================================
// Optimization Routine
// =============================================================================

void optimize(Floats &p, Float sigma, int iterations) {
    Float ss2InvNeg = -N / (2 * sigma * sigma);
    int NB = (N + BS - 1) / BS;
    int bufferSize = N * sizeof(Point);
    Point *pIn, *pOut;
    cudaMalloc(&pIn, bufferSize);
    cudaMalloc(&pOut, bufferSize);
    cudaMemcpy(pIn, p.data(), bufferSize, cudaMemcpyHostToDevice);
    int periods = MAX_DIST * sigma / std::sqrt(N);
    periods = min(periods, 9);
    fprintf(stderr, "periods = %d\n", periods);
    if (frameBaseName) snapshot(
        (Float *)pIn, N, w, w, coverage, frameBaseName, 0
    );
    for (int i = 0; i < iterations && !interruptFlag; i++) {
        fprintf(stderr, "\rIteration %d", i);
        switch (periods) {
            case 0: gbn0  <<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 1: gbn1  <<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 2: gbn<2><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 3: gbn<3><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 4: gbn<4><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 5: gbn<5><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 6: gbn<6><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 7: gbn<7><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 8: gbn<8><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
            case 9: gbn<9><<<NB, BS>>>(N, pIn, pOut, ss2InvNeg, scale); break;
        }
        cudaDeviceSynchronize();
        std::swap(pIn, pOut);
        if (frameBaseName) snapshot(
            (Float *)pIn, N, w, w, coverage, frameBaseName, i + 1
        );
    }
    cudaMemcpy(p.data(), pIn, bufferSize, cudaMemcpyDeviceToHost);
    cudaFree(pIn);
    cudaFree(pOut);
}

void optimizeND(
    Floats &p, Float sigma, int iterations, int d
) {                                                                             // High-dimensional version
    Float ss = -pow(N, 2. / d) / (2 * sigma * sigma);
    const int BS = 256;
    int NB = (N + BS - 1) / BS;
    int bufferSize = N * d * sizeof(Float);
    Float *pIn, *pOut;
    cudaMalloc(&pIn, bufferSize);
    cudaMalloc(&pOut, bufferSize);
    cudaMemcpy(pIn, p.data(), bufferSize, cudaMemcpyHostToDevice);
    int periods = MAX_DIST * sigma / pow(N, 1. / d);
    periods = min(periods, 9);
    fprintf(stderr, "periods = %d\n", periods);
    for (int i = 0; i < iterations && !interruptFlag; i++) {
        fprintf(stderr, "\rIteration %d", i);
        switch (periods) {
            case 0: gbnND0  <<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 1: gbnND<1><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 2: gbnND<2><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 3: gbnND<3><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 4: gbnND<4><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 5: gbnND<5><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 6: gbnND<6><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 7: gbnND<7><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 8: gbnND<8><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
            case 9: gbnND<9><<<NB, BS>>>(N, d, pIn, pOut, ss, scale); break;
        }

        cudaDeviceSynchronize();
        std::swap(pIn, pOut);
    }
    cudaMemcpy(p.data(), pIn, bufferSize, cudaMemcpyDeviceToHost);
    cudaFree(pIn);
    cudaFree(pOut);
}


void optimizeHarmonics(
    Floats &p, Float sigma, int dims, int harmonics, int iterations
) {                                                                             // Harmonics-based version
    Float sigmaSqx2Neg = 2 * sigma * sigma * pow(N, -2. / dims);
    const int BS = 256;
    int NB = (N + BS - 1) / BS;
    int bufferSize = N * dims * sizeof(Float);
    Float *pIn, *pOut;
    cudaMalloc(&pIn, bufferSize);
    cudaMalloc(&pOut, bufferSize);
    cudaMemcpy(pIn, p.data(), bufferSize, cudaMemcpyHostToDevice);
    for (int i = 0; i < iterations && !interruptFlag; i++) {
        fprintf(stderr, "\rIteration %d", i);
        gbnHarmonics<<<NB, BS>>>(
            N, dims, harmonics, pIn, pOut, sigmaSqx2Neg, scale/N
        );
        cudaDeviceSynchronize();
        std::swap(pIn, pOut);
    }
    cudaMemcpy(p.data(), pIn, bufferSize, cudaMemcpyDeviceToHost);
    cudaFree(pIn);
    cudaFree(pOut);
}

// =============================================================================
// Text Printout
// =============================================================================

void printText(const Floats &p, int dims, const char *fileName) {
    FILE *file = fopen(fileName, "w");
    if (dims == 2) fprintf(file, "%d\n", N);
    else fprintf(file, "%d %d\n", N, dims);
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dims; d++) {
            fprintf(
                file, "%0.17f%s", p[i * dims + d], d < dims - 1 ? " " : "\n"
            );
        }
    }
    fclose(file);
}


static void signalHandler(int signalCode) {
    fprintf(stderr, "Aborting ...\n");
    interruptFlag = true;
}

const char *USAGE_MESSAGE =
"Usage: %s [options] <point count> <iterations> [fileName fileName ..]\n"
"Options:\n"
"   -g <sigma>          User-supplied sigma\n"
"   -s <time step>      Scale factor for forces\n"
"   -d <dimensions>     Select dimensions; default is 2\n"
"   -h <harmonics>      Use harmonics-based optimization. "
                        "Set parameter to 0 to use default 0.5 * N^(-1/d)\n"
"   -p                  Generate projective instead of collective "
                        "high-dimensional BN\n"
"   -w                  Width of plotted frames, default is 1024\n"
"   -f <frameName>      Take snapshots named frameName%04d\n"
"   -c <coverage>       Coverage ratio of dots in plotted frames"
                        " default is 0.2\n"
;


int main(int argc,char **argv) {
    int opt;
    bool serial = false;
    Float coverage = 0.2;
    int dims = 2;
    int harmonics = 0;
    bool useHarmonics = false;
    bool projective = false;
    while ((opt = getopt(argc, argv, "s:g:d:h:pf:c:w:")) != -1) {
        switch (opt) {
            case 's': scale = atof(optarg); break;
            case 'g': sigma = atof(optarg); break;
            case 'd': dims = atoi(optarg); break;
            case 'h': useHarmonics = true; harmonics = atoi(optarg); break;
            case 'p': projective = true; break;
            case 'f': frameBaseName = optarg; break;
            case 'c': coverage = atof(optarg); break;
            case 'w': w = atoi(optarg); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    signal(SIGINT, signalHandler);
    if (optind > argc - 2) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]);
        exit(1);
    }
    N = atoi(argv[optind++]);
    int iterations = atoi(argv[optind++]);
    // =========================================================================
    // Save point sets
    // =========================================================================
    if (optind < argc) {                                                        // File names supplied?
        Floats p(N * dims);
        if (useHarmonics) {
            if (!harmonics) harmonics = 0.5 * pow(N, 1. / dims);
        }
        else {
            scale *= pow(N, 2. / dims) / N;
            if (sigma > 1) scale /= sigma * sigma;
        }
        int n = argc - optind;
        double totalTime(0);
        for (int fileNo = 0; fileNo < n && !interruptFlag; fileNo++) {
            for (int i = 0; i < N * dims; i++) {
                p[i] = (Float)(res * rnd());
            }
            clock_t t0 = clock();
            if (useHarmonics) {
                fprintf(stderr, "Using %d harmonics\n", harmonics);
                optimizeHarmonics(p, sigma, dims, harmonics, iterations);
            }
            else {
                if (dims == 2) optimize(p, sigma, iterations);
                else optimizeND(p, sigma, iterations, dims);
            }
            double consumedTime = clock() - t0;
            totalTime += consumedTime;
            printText(p, dims, argv[optind + fileNo]);
            fprintf(stderr,
                    "\ndone! Saved to %s; Total time = %.6fs\n",
                    argv[optind + fileNo], consumedTime / CLOCKS_PER_SEC
            );
        }
        fprintf(stderr, "Total time = %0.6fs\n", totalTime / CLOCKS_PER_SEC);
    }
}
