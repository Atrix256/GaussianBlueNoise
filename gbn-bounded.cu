/*
 * Optimize a point set to minimize Gaussian Blue Noise energy using
 * a gradient decent algorithm.
 * 2021-12-25: Created by Abdalla Ahmed from an earlier version (bounded)
 *             and revised for the paper
 * 2022-08-27: Added high-dimensional code.
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

#define MAX_DIM 20

//#define USE_DOUBLE
#include "float.h"

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
Float scale = 1;
const double res = 2.3283064365386962890625e-10;                                // 2 ^ -32
static bool interruptFlag = false;

// =============================================================================
// Optimization Kernels
// =============================================================================

__global__
void relaxOnce(
    int N, Point *pIn, Point *pOut,
    Float s2Inv, Float ss4InvNeg, Float a, Float scale
) {                                                                             // Using a continuous domain
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Vector grad;
    // =========================================================================
    // Attraction of domain:
    // =========================================================================
    Float l = s2Inv * pIn[i].x;                                                 // Distance to left edge
    Float r = s2Inv * (ONE - pIn[i].x);                                         // ~ right
    Float b = s2Inv * pIn[i].y;                                                 // ~ bottom
    Float t = s2Inv * (ONE - pIn[i].y);                                         // ~ top
    grad.x = a * (erfF(b) + erfF(t)) * (expF(-(l * l)) - expF(-(r * r)));
    grad.y = a * (erfF(l) + erfF(r)) * (expF(-(b * b)) - expF(-(t * t)));
    // =========================================================================
    // Repulsion of other points:
    // =========================================================================
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float dx = pIn[i].x - pIn[j].x;
        Float dy = pIn[i].y - pIn[j].y;
        Float rr = dx * dx + dy * dy;
        Float g = expF(ss4InvNeg * rr);
        grad.x += dx * g;
        grad.y += dy * g;
    }
    pOut[i].x = pIn[i].x + scale * grad.x;
    pOut[i].y = pIn[i].y + scale * grad.y;
}


// =============================================================================
// Optimization Routine
// =============================================================================

void optimize(Floats &p, Float sigma, int iterations, int dims = 2) {
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int bufferSize = N * dims * sizeof(Float);
    Float *buffer1, *buffer2;
    cudaMalloc(&buffer1, bufferSize);
    cudaMalloc(&buffer2, bufferSize);
    cudaMemcpy(buffer1, p.data(), bufferSize, cudaMemcpyHostToDevice);
    Float n = pow(N, 1. / dims);
    Float sigmaUnit = sigma / n;                                                // Sigma scaled to unit domain
    Float s2Inv = 1 / (2 * sigmaUnit);
    Float ss4InvNeg = -1 / (4 * sigmaUnit * sigmaUnit);
    Float a = 2 * sigma * sigma * sigma * SQRT_PI;
    a /= n;                                                             // Scale to unit domain
    for (int i = 0; i < iterations && !interruptFlag; i++) {
        fprintf(stderr, "\rIteration %d.", i);
        relaxOnce<<<numBlocks, blockSize>>>(
            N, (Point *)buffer1, (Point *)buffer2,
            s2Inv, ss4InvNeg, a, scale
        );
        cudaDeviceSynchronize();
        std::swap(buffer1, buffer2);
    }
    cudaMemcpy(p.data(), buffer1, bufferSize, cudaMemcpyDeviceToHost);
    cudaFree(buffer1);
    cudaFree(buffer2);
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
;


int main(int argc,char **argv) {
    unsigned seed = time(NULL);
    int opt;
    bool serial = false;
    Float coverage = 0.2;
    int dims = 2;
    while ((opt = getopt(argc, argv, "s:g:d:")) != -1) {
        switch (opt) {
            case 's': scale = atof(optarg); break;
            case 'g': sigma = atof(optarg); break;
            case 'd': dims = atoi(optarg); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    srand(seed);
    //srand48(seed);
    signal(SIGINT, signalHandler);
    if (optind > argc - 2) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]);
        exit(1);
    }
    N = atoi(argv[optind++]);
    int iterations = atoi(argv[optind++]);
    scale *= 0.25;                                                              // This works well when scale = 1.
    // =========================================================================
    // Save point sets
    // =========================================================================
    if (optind < argc) {                                                        // File names supplied?
        Floats p(N * dims);
        if (sigma > 1) scale /= sigma * sigma;
        sigma /= sqrt(2);                                                       // We are using the filtering sigma as input
        int n = argc - optind;
        double totalTime(0);
        for (int fileNo = 0; fileNo < n && !interruptFlag; fileNo++) {
            for (int i = 0; i < N * dims; i++) {
                p[i] = (Float)(res * rnd());
            }
            clock_t t0 = clock();
            optimize(p, sigma, iterations, dims);
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
