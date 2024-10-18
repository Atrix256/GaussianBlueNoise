/*
 * To select between single- and double-precision float
 * 2022-09-21: Revised by Abdalla Ahmed for inclusion with the paper
 */

#ifndef FLOAT_H
#define FLOAT_H

#include <vector>

#ifdef USE_DOUBLE
    #define Float double
    #define ZERO 0.0
    #define ONE 1.0
    #define TWO 2.0
    #define HALF 0.5
    #define QUARTER 0.25
    #define fabsF fabs
    #define expF exp
    #define erfF erf
    #define erfcF erfc
    #define copysignF copysign
    #define sqrtF sqrt
    #define rsqrtF rsqrt
    #define cosF cos
    #define sincosF sincos
    #define SQRT_PI 1.772453850905516027298
    #define MIN (1e-16)
    #define BITS (52)
    #define floorF floor
    #define ceilF ceil
    #define SUPPORT 8.5
#else
    #define Float float
    #define ZERO 0.0f
    #define ONE 1.0f
    #define TWO 2.0f
    #define HALF 0.5f
    #define QUARTER 0.25
    #define fabsF fabsf
    #define erfF erff
    #define erfcF erfcf
    #ifdef __CUDA_ARCH__
        #define expF __expf
    #else
        #define expF expf
    #endif
    #define copysignF copysignf
    #define sqrtF sqrtf
    #define rsqrtF rsqrtf
    #define cosF __cosf
    #define sincosF __sincosf
    #define SQRT_PI 1.772453850905516027298f
    #define MIN (1e-6)
    #define BITS (24)
    #define floorF floorf
    #define ceilF ceilf
    #define SUPPORT 5.8
#endif

// =========================================================================
// These utilities are made publicly available
// =========================================================================
__global__ void mulAll(Float *v, int N, Float s) {                              // s supplied by host
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    v[i] *= s;
}

__global__ void mulAll(Float *v, int N, Float *s) {                             // s already in device memory space, e.g. average
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    v[i] *= s[0];
}

__global__ void sqrtAll(Float *v, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    v[i] = sqrtF(v[i]);
}

__global__ void sqAll(Float *v, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    v[i] *= v[i];
}

__global__ void invAll(Float *aa, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    aa[i] = ONE / aa[i];
}

__global__
void setAll(Float *v, int N, Float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    v[i] = value;
}

__global__
void mulEqual(Float *left, int N, Float *right) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    left[i] *= right[i];
}

__global__
void divEqual(Float *left, int N, Float *right) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    left[i] /= right[i];
}

__global__
void plusEqual(Float *left, int N, Float *right) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    left[i] += right[i];
}

__global__
void printFloat(Float *x) {
    printf(" -- %e -- ", x[0]);
}


class FloatsGPU {                                                               // A vector of Floats with associated operators
private:
    Float *v;                                                                   // Pointer to the list in GPU memory
    int N;                                                                      // Number of elements
    static const int blockSize = 256;
    int numBlocks;
public:
    FloatsGPU() : v(NULL), N(0), numBlocks(0) { } ;
    FloatsGPU(int N);                                                           // Create a non-initialized list of N entries
    FloatsGPU(int N, Float value);                                              // Create a list of N entries and initialize all to value
    FloatsGPU(int N, Float *v);                                                 // Claim a given GPU buffer.
    FloatsGPU(std::vector<Float> hostVector);                                   // Create a list of Floats in the GPU and copies values from a host vector
    void operator=(FloatsGPU other);                                            // Copies entries from another list
    void memcpy(Float *ptr);                                                    // Copies entries pointed to by ptr
    void operator=(Float value);                                                // Set all elements to value
    void operator*=(Float s);                                                   // Multiply all elements by s, supplied by host
    void operator*=(Float *s);                                                  // Multiply all elements by *s in device memory
    void operator*=(FloatsGPU other);                                           // Multiply elements by the corresponding elements in other
    void sq() { sqAll<<<numBlocks, blockSize>>>(v, N); };                       // Replace all entries by their squares
    void sqrt() { sqrtAll<<<numBlocks, blockSize>>>(v, N); };                   // Replace all entries by their square roots
    void inv() { invAll<<<numBlocks, blockSize>>>(v, N); }                      // Replace all entries by their inverses
    Float* &data() { return v; };
    operator Float*&() { return v; };                                           // Use just as a pointer, instead of calling data()
    int size() { return N; };
    std::vector<Float> toHost();
    ~FloatsGPU();
};

FloatsGPU::FloatsGPU(int N) : N(N) {
    cudaMalloc(&v, N * sizeof(Float));
    numBlocks = (N + blockSize - 1) / blockSize;
}

FloatsGPU::FloatsGPU(int N, Float *v) : N(N), v(v) {
    numBlocks = (N + blockSize - 1) / blockSize;
}

FloatsGPU::FloatsGPU(int N, Float value) : N(N) {
    cudaMalloc(&v, N * sizeof(Float));
    numBlocks = (N + blockSize - 1) / blockSize;
    setAll<<<numBlocks, blockSize>>>(v, N, value);
}

FloatsGPU::FloatsGPU(std::vector<Float> hostVector) {
    N = hostVector.size();
    cudaMalloc(&v, N * sizeof(Float));
    cudaMemcpy(
        v, hostVector.data(), N * sizeof(Float), cudaMemcpyHostToDevice
    );
    numBlocks = (N + blockSize - 1) / blockSize;
}

void FloatsGPU::operator=(FloatsGPU other) {
    if (N != other.N) {
        if (v) cudaFree(v);
        N = other.N;
        cudaMalloc(&v, N * sizeof(Float));
        numBlocks = (N + blockSize - 1) / blockSize;
    }
    cudaMemcpy(v, other.v, N * sizeof(Float), cudaMemcpyDeviceToDevice);
}

void FloatsGPU::memcpy(Float *ptr) {
    cudaMemcpy(v, ptr, N * sizeof(Float), cudaMemcpyDeviceToDevice);
}


void FloatsGPU::operator*=(Float s) {
    mulAll<<<numBlocks, blockSize>>>(v, N, s);
}

void FloatsGPU::operator*=(Float *s) {
    mulAll<<<numBlocks, blockSize>>>(v, N, s);
}

void FloatsGPU::operator=(Float value) {
    setAll<<<numBlocks, blockSize>>>(v, N, value);
}


std::vector<Float> FloatsGPU::toHost() {
    std::vector<Float> output(N);
    cudaMemcpy(output.data(), v, N * sizeof(Float), cudaMemcpyDeviceToHost);
    return output;
}

FloatsGPU::~FloatsGPU() {
    if (v) cudaFree(v);
}


#endif


