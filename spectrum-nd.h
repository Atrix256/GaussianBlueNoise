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

// =============================================================================
// Global parameters
// =============================================================================

#define MAX_DIM 16

// =============================================================================
// GPU kernel
// =============================================================================

__global__
void powerSpectrum_gpu (
    Float *points,                                                              // A contiguous list of point sets
    int n,                                                                      // Number of sets
    int N,                                                                      // Number of points in a set
    int dims,                                                                   // Number of dimensions
    int size,                                                                   // Total number of frequencies along each dimension
    int volume,                                                                 // Size of the list, = (2 * size + 1) ^ dims
    Float *spectrum                                                             // Buffer for storing the spectrum
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > volume) return;
    Float w[MAX_DIM];                                                           // Angular frequencies per dimension
    int tmp = tid;
    int base = size * 2 + 1;
    for (int dim = 0; dim < dims; dim++) {
        w[dim] = TWOPI_NEG * (tmp % base - size);
        tmp /= base;
    }
    Float P = ZERO;
    for (int setNo = 0; setNo < n; setNo++) {
        Float *p = &points[setNo * N * dims];
        Float E_x = ZERO;
        Float E_y = ZERO;
        for (int i = 0; i < N; i++) {
            Float angle = ZERO;
            Float s, c;
            for (int dim = 0; dim < dims; dim++) {
                angle += w[dim] * p[i * dims + dim];
            }
            sincosF(angle, &s, &c);
            E_x += c;
            E_y += s;
        }
        P += E_x * E_x + E_y * E_y;
    }
    P /= n * N;
    spectrum[tid] = P;
}



#endif                                                                          // SPECTRUM_H
