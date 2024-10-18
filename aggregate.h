/*
 * CUDA reduction kernel adapted to our code
 * 2021-08-26: Created by Abdalla Ahmed
 *
 */

#ifndef AGGREGATE_H
#define AGGREGATE_H

#include "float.h"

// =============================================================================
// Sums the list in buffer1 into buffer1[0] and returns buffer1
// Buffer2 is used as a tmp storage.
// The contents in buffer1 are modified and should be copied if needed.
// =============================================================================

static const int aggrgBlckSize = 256;

__global__
void reduce(
    int listLength, Float *in, Float *out
) {
    int idx = threadIdx.x;                                                      // Thread index in the block
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              // Global thread index
    __shared__ Float accum[aggrgBlckSize];                                          // Per-block shared memory buffer
    accum[idx] = (i < listLength) ? in[i] : ZERO;                               // Each thread reads one element to shared memory
    __syncthreads();                                                            // Synchronize threads in the block to ensure all data is copied

    for (int size = aggrgBlckSize >> 1; size > 0; size >>= 1) {                 // Iterate through successive halves
        if (idx < size) {                                                       // If the thread is in the upper half, it is done
            accum[idx] += accum[idx + size];                                    // If it is in the lower half, it accumulate the value from its counter part in the upper half
        }
        __syncthreads();                                                        // Synch the threads to Ensure the operation is finished
    }
    if (idx == 0) out[blockIdx.x] = accum[0];                                   // Use thread 0 to write the accumulated value
}

__global__
void reduceMax(
    int listLength, Float *in, Float *out
) {
    int idx = threadIdx.x;                                                      // Thread index in the block
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              // Global thread index
    __shared__ Float accum[aggrgBlckSize];                                          // Per-block shared memory buffer
    accum[idx] = (i < listLength) ? in[i] : ZERO;                               // Each thread reads one element to shared memory
    __syncthreads();                                                            // Synchronize threads in the block to ensure all data is copied

    for (int size = aggrgBlckSize >> 1; size > 0; size >>= 1) {                 // Iterate through successive halves
        if (idx < size) {                                                       // If the thread is in the upper half, it is done
            accum[idx] = max(accum[idx], accum[idx + size]);                    // If it is in the lower half, it accumulate the value from its counter part in the upper half
        }
        __syncthreads();                                                        // Synch the threads to Ensure the operation is finished
    }
    if (idx == 0) out[blockIdx.x] = accum[0];                                   // Use thread 0 to write the accumulated value
}



// =============================================================================
// Host routine
// =============================================================================

inline Float *aggregate(Float *buffer1, int N, Float *buffer2) {
    const int blockSize = aggrgBlckSize;
    for (int listLength = N; listLength > 1; ) {
        int numBlocks = (listLength + blockSize - 1) / blockSize;
        reduce<<<numBlocks, blockSize>>>(
            listLength, buffer1, buffer2
        );
        listLength = numBlocks;
        std::swap(buffer1, buffer2);                                            // The result will always be at buffer1
    }
    return buffer1;
}

inline Float *aggregateMax(Float *buffer1, int N, Float *buffer2) {
    const int blockSize = aggrgBlckSize;
    for (int listLength = N; listLength > 1; ) {
        int numBlocks = (listLength + blockSize - 1) / blockSize;
        reduceMax<<<numBlocks, blockSize>>>(
            listLength, buffer1, buffer2
        );
        listLength = numBlocks;
        std::swap(buffer1, buffer2);                                            // The result will always be at buffer1
    }
    return buffer1;
}

void normalize(
    Float *x, int N, Float *tmp1, Float *tmp2, Float targetAvg
) {
    const int blockSize = aggrgBlckSize;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cudaMemcpy(tmp1, x, N * sizeof(Float), cudaMemcpyDeviceToDevice);
    Float *avgInv = aggregate(tmp1, N, tmp2);
    invAll<<<1, 1>>>(avgInv, 1);                                                // Now avgInv contains 1 / sum
    mulAll<<<1, 1>>>(avgInv, 1, targetAvg * N);                                 // Now avgInv contains (targetAvg * N / sum) = (targetAvg / avg)
    mulAll<<<numBlocks, blockSize>>>(x, N, avgInv);                             // Now avg(x) = targetAvg
    cudaDeviceSynchronize();
}



#endif
