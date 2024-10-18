/*
 * Reconstruct a bitmap from a stippling using a Gaussian filter
 * 2021-08-22: Created by Abdalla Ahmed
 * 2022-09-21: Final revision for the paper
 */

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "getopt/getopt.h"
#include <cmath>
#include <string>
#include <numeric>
#include "cairo/cairo.h"
#include <signal.h>
#include <algorithm>

#define USE_DOUBLE
#include "float.h"
#include "aggregate.h"

// =============================================================================
// Data Structures
// =============================================================================

struct Point {
    Float x, y;
};

typedef std::vector<Point> Points;
typedef std::vector<Float> Floats;
typedef std::string String;

// =============================================================================
// Global constants and variables
// =============================================================================

const int BS = 256;
bool interruptFlag = false;

// =============================================================================
// Load the point set
// =============================================================================

Points loadPoints(const char *fileName) {
    FILE *file = fopen(fileName, "r");
    int N;
    fscanf(file, " %d", &N);
    Points p(N);
    for (int i = 0; i < N; i++) {
        #ifdef USE_DOUBLE
        fscanf(file, " %lf %lf", &p[i].x, &p[i].y);
        #else
        fscanf(file, " %f %f", &p[i].x, &p[i].y);
        #endif
        if (feof(file)) {
            fprintf(stderr, "Error: could not load all points\n");
            exit(2);
        }
    }
    return p;
}


// =============================================================================
// kernel GPU Optimization
// =============================================================================

__global__
void optimizeKernels(
    int N, Point *p, Float *aIn, Float *aOut, Float ss2InvNeg
) {                                                                             // Scale amplitudes only
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Float density = 0;
    for (int j = 0; j < N; j++) {                                               // Iterate through the points to evaluate the density at reference point
        if (j == i) continue;
        Float dx = p[j].x - p[i].x;
        Float dy = p[j].y - p[i].y;
        density += aIn[j] * expF(aIn[j] * ss2InvNeg * (dx * dx + dy * dy));
    }
    aOut[i] = density;
}

// =============================================================================
// kernel CPU Optimization Routine
// =============================================================================

Float *optimizeKernels(
    int N, Point *p, Float *a, Float sigma, int iterations
) {
    Float ss2InvNeg = -N / (2 * sigma * sigma);                                 // For actual computation we scale sigma to the unit domain
    int NB = (N + BS - 1) / BS;
    Float *tmp1, *tmp2;
    cudaMalloc(&tmp1, N * sizeof(Float));
    cudaMalloc(&tmp2, N * sizeof(Float));
    for (int i = 0; i < iterations && !interruptFlag; i++) {
        fprintf(stderr, "\rIteration %d.", i);
        optimizeKernels<<<NB,BS>>>(N, p, a, tmp1, ss2InvNeg);
        std::swap(a, tmp1);
        normalize(a, N, tmp1, tmp2, 1);                                         // Make average a = 1
        cudaDeviceSynchronize();
    }
    cudaFree(tmp1);
    cudaFree(tmp2);
    return a;
}


// =============================================================================
// Reconstruction CUDA kernel
// =============================================================================

__global__
void computeImg(
    int N,                                                                      // Number of points
    Point *p,                                                                   // Array of points
    Float *a,                                                                   // Kernel shaping factors
    int w, int h,                                                               // Width and height of image
    Float *bmp,                                                                 // List of pixels
    Float pixelWidth,                                                           // Pixel size relative to unit domain
    Float ss2InvNeg                                                             // base exponent
) {
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixelIndex >= h * w) return;
    int X = pixelIndex % w;
    int Y = pixelIndex / w;
    Float x = X * pixelWidth;
    Float y = Y * pixelWidth;
    Float density(ZERO);
    for (int i = 0; i < N; i++) {
        Float dx = x - p[i].x;
        Float dy = y - p[i].y;
        Float rr = dx * dx + dy * dy;
        density += a[i] * expF(a[i] * ss2InvNeg * rr);
    }
    bmp[pixelIndex] = density;
}

// =============================================================================
// Reconstruction Host Routine
// =============================================================================

Floats reconstruct(
    int N, Point *p_gpu, Float *a, int w, int h, Float sigma
) {
    FloatsGPU img_gpu(h * w);
    Float *tmp1, *tmp2;
    cudaMalloc(&tmp1, h * w * sizeof(Float));
    cudaMalloc(&tmp2, h * w * sizeof(Float));
    int NB = (h * w + BS - 1) / BS;
    Float pixelWidth = ONE / std::max(w, h);
    Float ss2InvNeg = -N / (2 * sigma * sigma);
    computeImg<<<NB, BS>>>(N, p_gpu, a, w, h, img_gpu, pixelWidth, ss2InvNeg);
    cudaFree(tmp1);
    cudaFree(tmp2);
    return img_gpu.toHost();
}

// =============================================================================
// Save Image
// =============================================================================

void saveImage(Floats img, int w, int h, int avg, const char *fileName) {
    double mass = std::accumulate(img.begin(), img.end(), 0.);
    double scale = (255 - avg) * h * w / mass;
    std::vector<uint32_t> pixels(h * w);
    for (int Y = 0; Y < h; Y++) {
        for (int X = 0; X < w; X++) {
            int i = (h - 1 - Y) * w + X;
            int density = scale * img[i];
            uint32_t clr = 255 - std::min(density, 255);
            pixels[Y * w + X] = 0xff000000 + clr * 0x00010101;
        }
    }
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, w);
    cairo_surface_t *surface = cairo_image_surface_create_for_data(
        (unsigned char *)pixels.data(), CAIRO_FORMAT_RGB24, w, h, stride
    );
    cairo_surface_write_to_png(surface, fileName);
    cairo_surface_destroy(surface);
}

static void signalHandler(int signalCode) {
    fprintf(stderr, "Aborting ...\n");
    interruptFlag = true;
}

// =============================================================================
// Main
// =============================================================================

const char *USAGE_MESSAGE = "Usage: %s [options] <input.txt> <output.png>\n"
"Options:\n"
"   -w <width>          default 256, or height if given\n"
"   -h <height>         default 256, or width if given\n"
"   -g <sigma>          default 0.5\n"
"   -a <avgerageClr>    Average output pixel. default 127.5\n"
"   -i <iterations>     Iterations for kernel optimization, default is 15\n"
;

int main(int argc,char **argv) {
    int opt;
    int w = 0, h = 0;
    Float sigma = 0.5;
    Float avg = 127.5;
    int iterations = 15;
    while ((opt = getopt(argc, argv, "w:h:g:a:i:")) != -1) {
        switch (opt) {
            case 'w': w = atoi(optarg); if (!h) h = w; break;
            case 'h': h = atoi(optarg); if (!w) w = h; break;
            case 'g': sigma = atof(optarg); break;
            case 'a': avg = atof(optarg); break;
            case 'i': iterations = atoi(optarg); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    if (!h && !w) w = h = 256;
    if (optind > argc - 2) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
    }
    char *inputFileName = argv[optind];
    char *outputFileName = argv[optind + 1];
    Points p = loadPoints(inputFileName);
    signal(SIGINT, signalHandler);
    int N = p.size();
    int NB = (N + BS - 1) / BS;
    Point *p_gpu;
    cudaMalloc(&p_gpu, N * sizeof(Point));
    cudaMemcpy(p_gpu, p.data(), N * sizeof(Point), cudaMemcpyHostToDevice);
    fprintf(stderr, "sigma = %f\n", sigma);
    Float *a;
    cudaMalloc(&a, N * sizeof(Float));
    setAll<<<NB, BS>>>(a, N, 1);
    fprintf(stderr, "Optimizing kernels ..\n");
    clock_t t0 = clock();
    a = optimizeKernels(N, p_gpu, a, sigma, iterations);
    clock_t t1 = clock();
    double totalTime = (double)(t1 - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "done! time = %.6fs\n", totalTime);
    fprintf(stderr, "Reconstructing image .. ");
    t0 = clock();
    Floats img = reconstruct(N, p_gpu, a, w, h, sigma);
    t1 = clock();
    totalTime = (double)(t1 - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "done! time = %.6fs\n", totalTime);
    saveImage(img, w, h, avg, outputFileName);
    cudaFree(p_gpu);
    cudaFree(a);
}

