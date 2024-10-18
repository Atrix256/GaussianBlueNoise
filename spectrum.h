/*
 */

#ifndef SPECTRUM_H
#define SPECTRUM_H

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "cairo/cairo.h"
#include <cmath>

#ifdef USE_DOUBLE                                                               // If used, this should be defined in the including file before including this header
    #define Float double
    #define fabsF fabs
    #define log2F log2
    #define sqrtF sqrt
    #define sincosF sincos
    #define ONE 1.0
    #define ZERO 0.0
    #define SCALE 0.25                                                          /* As defined in PSA */
    #define TWOPI_NEG (-6.2831853071795864769)
#else
    #define Float float
    #define fabsF fabsf
    #define log2F log2f
    #define sqrtF sqrtf
    #define sincosF sincosf
    #define ONE 1.0f
    #define ZERO 0.0f
    #define SCALE 0.25                                                          /* As defined in PSA */
    #define TWOPI_NEG (-6.2831853071795864769f)
#endif

struct PointFloat {
    Float x, y;
};

// =============================================================================
// Tex parametes
// =============================================================================

float ymin = 0;
float ymax = 4.2;
float xmin = 0;
float xmax = -1;
char *userInstructions = "";
Float average = 0;



// =============================================================================

// =============================================================================


void saveImage(
    unsigned char *data,
    unsigned width,
    unsigned height,
    const char *fname
) {
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, width);
    cairo_surface_t *surface = cairo_image_surface_create_for_data(
        data, CAIRO_FORMAT_RGB24,
        width, height, stride
    );
    cairo_surface_write_to_png(surface, fname);
    cairo_surface_destroy(surface);
}

// =============================================================================
// Uniform
// =============================================================================

__device__ Float frequencyEnergy(PointFloat *p, int N, int f_X, int f_Y) {      // Compute the energy at frequency (f_X, f_Y) of the N points pointed to by p at
    Float wx = TWOPI_NEG * f_X;
    Float wy = TWOPI_NEG * f_Y;
    Float E_x = ZERO;
    Float E_y = ZERO;
    Float s, c;
    for (int i = 0; i < N; i++) {
        double angle = wx * p[i].x + wy * p[i].y;
        sincosF(angle, &s, &c);
        E_x += c;
        E_y += s;
    }
    return E_x * E_x + E_y * E_y;
}

__global__
void powerSpectrum_gpu (
    PointFloat *p,                                                              // A contiguous list of point sets
    int n,                                                                      // Number of sets
    int N,                                                                      // Number of points in a set
    Float *spectrum,                                                            // Buffer for storing the 2D spectrum
    int width                                                                   // Total width of spectrum; will be centered
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int O = width >> 1;
    int X = tid % (width + 1);                                                  // i % n
    int Y = tid / (width + 1);                                                  // i / n
    int f_X = X - O;
    int f_Y = Y - O;
    if (f_Y > 0) return;                                                        // Take advantage of origin symmetry of the power spectrum
    Float P = ZERO;
    for (int i = 0; i < n; i++) {
        P += frequencyEnergy(&p[i * N], N, f_X, f_Y);
    }
    P /= n * N;
    if (X < width) spectrum[Y * width + X] = P;
    if (X > 0) spectrum[(O - f_Y) * width + (O - f_X)] = P;
}

// =============================================================================
// Weighted
// =============================================================================

__device__ Float frequencyEnergy(
    PointFloat *p, Float *wt, int N, int f_X, int f_Y
) {                                                                             // Compute the energy at frequency (f_X, f_Y) of the N points pointed to by p at
    Float wx = TWOPI_NEG * f_X;
    Float wy = TWOPI_NEG * f_Y;
    Float E_x = ZERO;
    Float E_y = ZERO;
    Float s, c;
    for (int i = 0; i < N; i++) {
        double angle = wx * p[i].x + wy * p[i].y;
        sincosF(angle, &s, &c);
        E_x += wt[i] * c;
        E_y += wt[i] * s;
    }
    return E_x * E_x + E_y * E_y;
}

__global__
void powerSpectrum_gpu (
    PointFloat *p,                                                              // A contiguous list of point sets
    Float *wt,                                                                  // A corresponding list of weights
    int n,                                                                      // Number of sets
    int N,                                                                      // Number of points in a set
    Float *spectrum,                                                            // Buffer for storing the 2D spectrum
    int width                                                                   // Total width of spectrum; will be centered
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int O = width >> 1;
    int X = tid % (width + 1);                                                  // i % n
    int Y = tid / (width + 1);                                                  // i / n
    int f_X = X - O;
    int f_Y = Y - O;
    if (f_Y > 0) return;                                                        // Take advantage of origin symmetry of the power spectrum
    Float P = ZERO;
    for (int i = 0; i < n; i++) {
        P += frequencyEnergy(&p[i * N], &wt[i * N], N, f_X, f_Y);
    }
    P /= n * N;
    if (X < width) spectrum[Y * width + X] = P;
    if (X > 0) spectrum[(O - f_Y) * width + (O - f_X)] = P;
}


// =============================================================================

// =============================================================================


std::vector<Float> powerSpectrum (                                              // A host wrapper function
    PointFloat *p_gpu,                                                          // A contiguous list of point sets in GPU memory
    Float *wt,                                                                  // An optional corresponding list of weights in GPU memory
    int n,                                                                      // Number of sets
    int N,                                                                      // Number of points in a set
    int width                                                                   // Total width of spectrum; will be centered
) {
    int widthSq = width * width;
    std::vector<Float> spectrum(widthSq);
    Float *spectrum_gpu;
    cudaMalloc(&spectrum_gpu, widthSq * sizeof(Float));
    int blockSize = 256;
    int numBlocks = (widthSq + blockSize - 1) / blockSize;
    if (wt) {
        powerSpectrum_gpu<<<numBlocks, blockSize>>>(
            p_gpu, wt, n, N, spectrum_gpu, width
        );
    }
    else {
        powerSpectrum_gpu<<<numBlocks, blockSize>>>(
            p_gpu, n, N, spectrum_gpu, width
        );
    }
    cudaDeviceSynchronize();
    cudaMemcpy(
        spectrum.data(), spectrum_gpu,
        widthSq * sizeof(Float), cudaMemcpyDeviceToHost
    );
    cudaFree(spectrum_gpu);
    if (average != 0) {
        Float dc = spectrum[(width/2) * width + (width/2)];
        Float scale = average / dc;
        for (int i = 0; i < widthSq; i++) {
            spectrum[i] *= scale;
        }
    }
    return spectrum;
}

// =============================================================================

// =============================================================================


void plotSpectrum (                                                             // Saves a PNG plot of a frequency power spectrum
    std::vector<Float> spectrum,                                                // Like the output of powerSpectrum
    int width,                                                                  // This is supposed to be sqrt(spectrum.size())
    const char *fileName
) {
    std::vector<int> img(spectrum.size());
    for (int Y = 0; Y < width; Y++) {
        for (int X = 0; X < width; X++) {
            //int i = (width - 1 - Y) * width + X;                              // This should be the right one.
            int i = Y * width + X;                                              // For an unclear reason, the flipped plot is different from PSA
            Float P = spectrum[i];
            P = std::sqrt(P);                                                   // As in PSA
            P = std::log2(ONE + SCALE * P);                                     // Logarithmic tone mapping, as in PSA
            P = std::min(P, ONE);                                               // Clamping
            unsigned char c = 255 * P;
            img[Y * width + X] = 0xff000000 + c * 0x00010101;
        }
    }
    saveImage((unsigned char *)img.data(), width, width, fileName);
}


// =============================================================================
// Compute evacuated energy
// =============================================================================

Float bnLoss (
    std::vector<Float> spectrum,                                                // Like the output of powerSpectrum
    int width,                                                                  // This is supposed to be sqrt(spectrum.size())
    int N                                                                       // Number of points
) {
    Float sum = ZERO;
    int O = width / 2;                                                          // Origin coordinates
    Float sqrtN = std::sqrt(N);
    for (int Y = 0; Y < width; Y++) {
        for (int X = 0; X < width; X++) {
            int f_X = X - O;
            int f_Y = Y - O;
            int rr = f_X * f_X + f_Y * f_Y;
            Float a = spectrum[Y * width + X];
            if (rr < N && a < 1) {
                sum += log(a);
            }
        }
    }
    return -sum / N;
}

// =============================================================================

// =============================================================================

void plotRadialPower (
    std::vector<Float> spectrum,                                                // Like the output of powerSpectrum
    int width,                                                                  // This is supposed to be sqrt(spectrum.size())
    const char *fileName,
    int N,                                                                      // Number of points
    Float scale = 0.5,                                                          // Scale of frequency range. This is the default of PSA
    bool loglog = false,
    bool allBins = false                                                        // Populate all distinct radii
) {
    if (allBins) scale = 1;
    // =========================================================================
    // Populate 1D histogram
    // =========================================================================
    int O = width / 2;                                                          // Origin coordinates
    int nbins = 2 * O * O;
    std::vector<Float> rp(nbins, ZERO);
    std::vector<int> hits(nbins, 0);
    for (int Y = 0; Y < width; Y++) {
        for (int X = 0; X < width; X++) {
            int f_X = X - O;
            int f_Y = Y - O;
            int rr = f_X * f_X + f_Y * f_Y;
            int binNo = scale * (allBins ? rr : std::sqrt(rr));
            if (binNo < nbins) {
                rp[binNo] += spectrum[Y * width + X];
                hits[binNo]++;
            }
        }
    }
    for (int binNo = 0; binNo < nbins; binNo++) {
        if (hits[binNo]) rp[binNo] /= hits[binNo];
    }
    // =========================================================================
    // Plot
    // =========================================================================
    FILE *file = fopen(fileName, "w");
    Float unit = 1.0 / (scale * sqrt(N));                                       // Unit frequency relative to number of points per dimension
    Float maxFrequecny =
        unit * (allBins ? sqrt(nbins - 1) : int(scale * O) - 1);
    if (xmax < 0) xmax = maxFrequecny;
    if (strstr(fileName, ".tex")) {
        fprintf(
            file,
            "\\documentclass{standalone}\n"
            "\\usepackage{tikz}\n"
            "\\usepackage{pgfplots}\n"
            "\\usepgfplotslibrary{fillbetween}\n"
            "\\begin{document}\n"
            "\\begin{tikzpicture}\n"
            "  \\begin{axis}[\n"
            "    width=10cm,\n"
            "    height=5cm,\n"
            "    ymin = %e,\n"
            "    ymax = %e,\n"
            "    xmin = %e,\n"
            "    xmax=%e,\n",
            ymin, ymax, xmin, xmax

        );
        if(loglog) {
            fprintf(
                file,
                "    cycle list name=color list,\n"
                "    xmode=log,\n"
                "    ymode=log,\n"
                "  ]\n"
                "  \\addplot table {\n"
            );
        }
        else {
            fprintf(
                file,
                "    %%ticks=none, hide axis,\n"
                "    xtick align = outside,\n"
                "  ]\n"
                "  \\addplot[blue, thick, name path=A] table {\n"
            );
        }
        for (int binNo = 1; binNo < nbins; binNo++) {
            Float r = allBins ? std::sqrt(binNo) : binNo;
            Float f = unit * r;
            if (f <= xmax && hits[binNo]) {
                fprintf(file, "    %19.17f %21.17f\n", f, rp[binNo]);
            }
        }
        fprintf(file, "  };\n");
        if (!loglog) {
            fprintf(
                file,
                "  \\addplot[draw=none, domain=%f:%f, name path=B] {0};\n"
                "  \\addplot[blue!10] fill between[of=A and B];\n"
                "  \\addplot[help lines,dashed, domain=0:%f] {1};\n",
                unit, maxFrequecny, maxFrequecny
            );
        }
        fprintf(
            file, "%s\n\\end{axis};\n\\end{tikzpicture}\n\\end{document}\n",
            userInstructions
        );
    }
    else if (strstr(fileName, ".txt")) {
        for (int binNo = 1; binNo < nbins; binNo++) {
            Float r = allBins ? std::sqrt(binNo) : binNo;
            if (hits[binNo]) {
                fprintf(file, "    %19.17f %21.17f\n", unit * r, rp[binNo]);
            }
        }
    }
    fclose(file);
}


#endif                                                                          // SPECTRUM_H
