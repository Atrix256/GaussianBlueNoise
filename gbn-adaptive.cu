/*
 * Adaptive GBN sampling of a density map.
 * This is a minimal code for testing.
 * The density map is expected as a .pgm file.
 * 2022-09-21: Final revision by Abdalla Ahmed for inclusion with the paper
 */

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "getopt/getopt.h"
#include <cmath>
#include <string>
#include <signal.h>
#include <stdint.h>
#include <numeric>                                                              // For std::accumulate
#include <algorithm>                                                            // For std::max_element

// Comment/uncomment this to use single/double precision:
//#define USE_DOUBLE

#include "gpu-point-set.h"

// =============================================================================
// Data Structures
// =============================================================================

typedef unsigned char Byte;                                                     // For grayscale pixels
typedef std::vector<Byte> Bytes;                                                // For arrays of pixels

struct Image {                                                                  // Data structure for holding image information
    int w, h;                                                                   // Width and height
    Bytes pixels;                                                               // Image data
    Byte& operator[](int i) { return pixels[i]; };                              // Cast to pixel array
};

// =============================================================================
// Global Parameters
// =============================================================================

static bool interruptFlag = false;                                              // To stop the optimization manually

// =============================================================================
// CUDA Kernel Optimization: Algorithm 2 in the paper
// =============================================================================

__global__
void optimizeKernels(
    int N,                                                                      // Number of points
    Point *p,                                                                   // Point coordinates
    Float *aIn, Float *aOut,                                                    // Shaping factors input and output
    int w, int h, Float *bmp,                                                   // Image parameters and pixel data
    Float aPixel,                                                               // Shaping fixed factor of pixel kernels
    Float ss2InvNeg,                                                            // Exponent coefficient: Sigma squared, multiplied by 2, inversed, negated
    int pixelSupport                                                            // Pixel neighborhood, beyond which the kernels are below numeric precision
) {                                                                             // Scale amplitudes only
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              // Thread index used as point index
    if (i >= N) return;                                                         // Needed where N is not a multiple of thread block size
    Float density = 0;                                                          // Density at the point due to other kernels
    // -------------------------------------------------------------------------
    // Density due to other points
    // -------------------------------------------------------------------------
    for (int j = 0; j < N; j++) {
        if (j == i) continue;                                                   // Skip self
        Float dx = p[j].x - p[i].x;
        Float dy = p[j].y - p[i].y;
        density += aIn[j] * expF(aIn[j] * ss2InvNeg * (dx * dx + dy * dy));
    }
    // -------------------------------------------------------------------------
    // Density due to pixels, taken positive here
    // -------------------------------------------------------------------------
    int X0(p[i].x), Y0(p[i].y);
    int Xmin = max(X0 - pixelSupport, 0);
    int Xmax = min(X0 + pixelSupport, w - 1);
    int Ymin = max(Y0 - pixelSupport, 0);
    int Ymax = min(Y0 + pixelSupport, h - 1);
    for (int Y = Ymin; Y <= Ymax; Y++) {
        Float dy = p[i].y - Y;
        for (int X = Xmin; X <= Xmax; X++) {
            Float dx = p[i].x - X;
            Float rr = dx * dx + dy * dy;
            Float g = aPixel * expF(aPixel * ss2InvNeg * rr);
            density += bmp[Y * w + X] * g;
        }
    }
    aOut[i] = density;                                                          // This will be normalized subsequently
}

// =============================================================================
// CUDA Position Optimization
// =============================================================================

__global__
void relaxOnce(
    int N,                                                                      // Number of sample points
    Point *pIn,                                                                 // Input point coordinates
    Point *pOut,                                                                // Output coordinates
    Float *a,                                                                   // Kernel shaping factors
    int w, int h, Float *bmp,                                                   // Image parameters and pixel data
    Float aPixel,                                                               // Shaping factor of pixels, fixed
    Float ss2InvNeg,                                                            // Exponent coefficient: Sigma squared, times 2, inversed, negated
    Float s,                                                                    // Gradient time step
    Float sigma                                                                 // Nominal sigma, needed for computing pixel neighborhood
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              // Thread id
    if (i >= N) return;                                                         // Needed when number of points in not a multiple of block size
    Vector grad = {ZERO, ZERO};                                                 // Gradient
    s *= 0.5;                                                                   // Our default, mostly empirical
    ss2InvNeg *= 2 * a[i];                                                      // Factored from the (2.a_i.a_j/(a_i + a_j)) exponenet cooeficient
    s *= a[i] * a[i];                                                           // Factored from the (a_i.a_j/(a_i + a_j))^2 amplitude cooeficient
    // -------------------------------------------------------------------------
    // Attraction of image
    // -------------------------------------------------------------------------
    Float b = aPixel / (a[i] + aPixel);                                         // Rest of exponenet cooeficient
    int support = 6 * sigma/min(a[i], aPixel);                                  // Neighborhood depends on stretch of kernel
    int X0(pIn[i].x), Y0(pIn[i].y);
    int Xmin = max(X0 - support, 0);
    int Xmax = min(X0 + support, w - 1);
    int Ymin = max(Y0 - support, 0);
    int Ymax = min(Y0 + support, h - 1);
    for (int Y = Ymin; Y <= Ymax; Y++) {
        for (int X = Xmin; X <= Xmax; X++) {
            Float dx = pIn[i].x - X;
            Float dy = pIn[i].y - Y;
            Float rr = dx * dx + dy * dy;
            Float g = b * b * expF(b * ss2InvNeg * rr);                         // Further optimization is possible, but we preferred clarity here
            g *= bmp[Y * w + X];
            grad.x -= dx * g;
            grad.y -= dy * g;
        }
    }
    // -------------------------------------------------------------------------
    // Repulsion of other points:
    // -------------------------------------------------------------------------
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float dx = pIn[i].x - pIn[j].x;
        Float dy = pIn[i].y - pIn[j].y;
        Float rr = dx * dx + dy * dy;
        Float b = a[j] / (a[i] + a[j]);                                         // This is possibly costly, but no workaround found
        Float g = b * b * expF(b * ss2InvNeg * rr);
        grad.x += dx * g;
        grad.y += dy * g;
    }
    // -------------------------------------------------------------------------
    // Update point:
    // -------------------------------------------------------------------------
    Float x = pIn[i].x + s * grad.x;
    Float y = pIn[i].y + s * grad.y;
    if (
        x >= 0 && x < w && y >= 0 && y < h                                      // Restrict to image boundary. In theory it should not be needed
        && bmp[int(y) * w + int(x)] > 0                                         // Restrict to non-white pixels. Also should not be required
    ) {                                                                         // If new location is acceptible
        pOut[i] = {x, y};                                                       // Commit
    }
    else {                                                                      // Otherwise
        pOut[i] = pIn[i];                                                       // Fall back to old location
    }
}

// =============================================================================
// Adaptive Class. Common GPU-related items kept in a base class.
// =============================================================================

class GBNAdaptive : public GPUPointSet {
private:
    Float sigma;                                                                // Sigma of energy kernel
    Float ss2InvNeg;                                                            // Exponent factor: Sigma squared, times 2, inversed, negated
    Float *bmp;                                                                 // List of pixels in GPU memory
    Float *a;                                                                   // Buffer for holding kernel shaping factors (amplitudes)
    Float aPixel;                                                               // Common shaping factor of pixels
    Float areaRatio;                                                            // A factor to account for empty areas of image
    int pixelSupport;                                                           // Coverage of pixel kernels for density estimation
    void init(const Byte pixels256[]);                                          // Initialize parameters and buffers
public:
    GBNAdaptive(                                                                // Construct from scratch
        int N,                                                                  // Number of points
        Image img,                                                              // Density map
        bool initRandom = false                                                 // Equal chance for all non-empty pixels; default is weighted
    );
    GBNAdaptive(char *txtFileName, const Image img);                            // Load a given point set
    void setSigma(Float v);                                                     // Set sigma to a given value
    void optimize(int iterations, bool terminateOnNoMove = false);              // Apply optimization
    ~GBNAdaptive();
};

// -----------------------------------------------------------------------------
// Construct from scratch
// -----------------------------------------------------------------------------

GBNAdaptive::GBNAdaptive(
    int N, Image img, bool initRandom
) : GPUPointSet(N, img.w, img.h) {
    Byte maxPixel = *std::max_element(img.pixels.begin(), img.pixels.end());    // Highest pixel density; high is darker
    for (int i = 0; i < N; /*conditionally increment inside loop*/) {           // Insert points proportional to density using rejection sampling
        Float x = rand(w);                                                      // Random x in [0, w)
        Float y = rand(h);                                                      // Random y in [0, h)
        Float threshold = initRandom ? 0 : rand(maxPixel);                      // Random threshold or zero to accept all non-empty pixel locations
        if (img[int(y) * w + int(x)] > threshold) {                             // Is underlying pixel above threshold?
            p[i++] = {x, y};                                                    // If yes, insert a point at this location
        }
    }
    init(img.pixels.data());                                                    // Common initialization tasks
}

// -----------------------------------------------------------------------------
// Load a given point set
// -----------------------------------------------------------------------------

GBNAdaptive::GBNAdaptive(
    char *txtFileName, Image img
) : GPUPointSet(txtFileName, img.w, img.h) {
    init(img.pixels.data());
}

// -----------------------------------------------------------------------------
// Initialize parameters and buffers
// -----------------------------------------------------------------------------

void GBNAdaptive::init(const Byte pixels256[]) {
    std::vector<Float> bmpCPU(pixels256, pixels256 + h * w);                    // Create a local copy for processing
    Float mass = std::accumulate(bmpCPU.begin(), bmpCPU.end(), ZERO);           // Sum of all pixels
    outBlackRatio = mass / (w * h * 255);                                       // Default output rendering black level to match input image
    Float avg = mass / (h * w);                                                 // Current average pixel
    for (int i = 0; i < w * h; i++) { bmpCPU[i] *= (1. / avg); }                // Make average 1
    Float sumSq(0);                                                             // sum of squares of pixel densities
    for (int i = 0; i < h * w; i++) {                                           // Iterate through pixels
        sumSq += bmpCPU[i] * bmpCPU[i];                                         // Once as a density and once as an amplitude shaping factor
    }
    areaRatio = h * w / sumSq;                                                  // Effectve coverage of image area
    fprintf(stderr, "Covered area = %5.2f%%.\n", areaRatio * 100);
    for (int i = 0; i < w * h; i++) { bmpCPU[i] *= N / Float(h * w); }          // Make mass of pixels equal to points
    setSigma(1);                                                                // Default sigma
    cudaMalloc(&bmp , h * w * sizeof(Float));                                   // Allocate memory for pixels
    cudaMalloc(&a  , N * sizeof(Float));                                        // Allocate memory for shaping factors
    cudaMemcpy(
        bmp  , bmpCPU.data(), h * w * sizeof(Float), cudaMemcpyHostToDevice
    );                                                                          // Copy pixels to device memory
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------

GBNAdaptive::~GBNAdaptive() {
    cudaFree(bmp);                                                              // Free image buffer
    cudaFree(a);                                                                // Free shaping-factors buffer
}

// -----------------------------------------------------------------------------
// Set energy sigma
// -----------------------------------------------------------------------------

void GBNAdaptive::setSigma(Float v) {
    sigma = v                                                                   // Input
            / sqrt(N)                                                           // Scale to point area
            / sqrt(2)                                                           // Account for the pixels, 50% share
            * sqrt(w * h / Float(n * n))                                        // Account for clipped part of the domain
            * sqrt(areaRatio)                                                   // Account for empty space of image
            * n                                                                 // Scale to pixel units
    ;
    ss2InvNeg = -1 / (2 * sigma * sigma);                                       // Will be used by both optimization routines
    pixelSupport = SUPPORT * sigma * sqrt(N / Float(w * h));
    aPixel = Float(w * h) / N;                                                  // Update shaping factor of pixel kernels
    fprintf(
        stderr, "Sigma set to = %f pixels; pixel support = %d\n",
        sigma, pixelSupport
    );
}

// -----------------------------------------------------------------------------
// Optimization Routine
// -----------------------------------------------------------------------------

void GBNAdaptive::optimize(int iterations, bool terminateOnNoMove) {
    cudaMemcpy(p1  , p.data(), N * sizeof(Point), cudaMemcpyHostToDevice);      // Copy points to device memory buffer
    cudaMemcpy(pRef,       p1, N * sizeof(Point), cudaMemcpyDeviceToDevice);    // Copy initial point locations as a reference
    setAll<<<NB,BS>>>(a, N, 1);                                                 // Initialize all shaping factors to 1
    for (int i = 0; i < iterations && !interruptFlag; i++) {
        fprintf(stderr, "\rIteration: %6d", i);
        // ---------------------------------------------------------------------
        // Kernel Optimization
        // ---------------------------------------------------------------------
        optimizeKernels<<<NB,BS>>>(
            N, p1, a, tmp1, w, h, bmp, aPixel, ss2InvNeg, pixelSupport
        );
        std::swap(a, tmp1);
        normalize(a, N, tmp1, tmp2, 1);                                         // Make average a = 1
        // ---------------------------------------------------------------------
        // Position Optimization
        // ---------------------------------------------------------------------
        relaxOnce<<<NB, BS>>>(
            N, p1, p2, a, w, h, bmp, aPixel, ss2InvNeg, s, sigma
        );
        cudaDeviceSynchronize();
        std::swap(p1, p2);
        if (terminateOnNoMove) {
            Float dmax = maxDistance(p1, p2);
            fprintf(stderr, ", max displacement: %12.6e", dmax);
            if (!dmax) break;
        }
        if (!frameBaseName.empty()) snapshot(i);
    }
    fprintf(stderr, "\n");
    cudaMemcpy(p.data(), p1, N * sizeof(Point), cudaMemcpyDeviceToHost);        // Copy back to CPU memory
}


// =============================================================================
// Image loading
// =============================================================================

Image loadPGM(const char *fileName) {
    FILE *pgmfile = fopen(fileName, "r");
    if (!pgmfile) {
        fprintf(stderr, "Failed to open file %s\n", fileName);
        exit(1);
    }
    Image img;
    int clrMax;
    fscanf(pgmfile, "P2 %d %d %d", &img.w, &img.h, &clrMax);
    Float clrScl = 255. / clrMax;
    img.pixels.resize(img.w * img.h);
    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            int clr;
            fscanf(pgmfile, " %d", &clr);
            if (feof(pgmfile)) {
                fprintf(stderr,
                        "Sorry, failed to read image information\n");
                exit(2);
            }
            img[(img.h - 1 - y) * img.w + x] = clrScl * (clrMax - clr);
        }
    }
    return img;
}


// =============================================================================
// User Interruption Handling
// =============================================================================

static void signalHandler(int signalCode) {
    fprintf(stderr, "Aborting ...\n");
    interruptFlag = true;
}

const char *USAGE_MESSAGE =
"Usage:"
" %s [options] <image.pgm> <point count> <iterations> "
"[output.abc output.efg]\n"
"Suppoted save types are .txt, .eps, .png, .pdf\n"
"Options:\n"
"   -g <sigma>          User-supplied sigma\n"
"   -s <time step>      Scale factor for gradient relative to default\n"
"   -R                  Random initialization;"
                        " default is proportional to density\n"
"   -T                  Terminate optimization on no move. This will require "
                        " computing distance, hence adds some delay.\n"
"   -c <outBlackRatio>  Coverage ratio of black in renderings,"
                        " default is computed from image mean\n"
"   -l <fileName>       Load points from txt file\n"
"   -f <frameName>      Take snapshots named frameName%%07d.ext\n"
"   -F <format>         Bitmask of frame name extensions: "
                        "1: txt, 2: eps, 4: png, 8: pdf; default is 4.\n"
"   -W <width>          Width of recorded frames; default is image width\n"
"   -H <height>         Height of recorded frames; default is image height\n"
"   -C <condition>      Condition for frame recording: "
                        "1: visible change, 2: powers of 2 iteration index, "
                        "therwise: all iterations.\n"
"   -M <margin>         Add a fractional margin in plots; default is not\n"
;


int main(int argc,char **argv) {
    int opt;
    char *inputFileName = NULL;
    char *frameBaseName = NULL;
    Float s = 0;
    Float sigma = 0;
    Float sigmaPixels = 0;
    Float outBlackRatio = 0;
    int W = 0, H = 0;
    int snapshotCondition = 0;
    int frameFormat = 0;
    bool terminateOnNoMove = false;
    bool initRandom = false;
    Float margin = 0;
    while ((opt = getopt(argc, argv, "g:s:c:l:f:W:H:C:F:TRM:")) != -1) {
        switch (opt) {
            case 'g': sigma = atof(optarg); break;
            case 's': s = atof(optarg); break;
            case 'c': outBlackRatio = atof(optarg); break;
            case 'l': inputFileName = optarg; break;
            case 'f': frameBaseName = optarg; break;
            case 'F': frameFormat = atoi(optarg); break;
            case 'W': W = atoi(optarg); if (!H) H = W; break;
            case 'H': H = atoi(optarg); if (!W) W = H; break;
            case 'C': snapshotCondition = atoi(optarg); break;
            case 'T': terminateOnNoMove = true; break;
            case 'R': initRandom = true; break;
            case 'M': margin = atof(optarg); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    if (optind > argc - 3) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]);
        exit(1);
    }
    signal(SIGINT, signalHandler);
    char *imageFileName = argv[optind++];
    int N = atoi(argv[optind++]);
    int iterations = atoi(argv[optind++]);
    Image img = loadPGM(imageFileName);

    // In my windows compile of this code, the below was creating temporary GBNAdaptive objects, which would allocate
    // host memory and copy those pointers to gbn.  The temporary would then go out of scope and free that memory that
    // gbn was still holding onto.  This made use after free problems, like the cudaMemcpy calls in GBNAdaptive::optimize()
    // would fail due to trying to copy to and from freed host memory. working around this to get the point sets we need.
    /*
    GBNAdaptive gbn = (
        inputFileName ?
        GBNAdaptive(inputFileName, img) :
        GBNAdaptive(N, img, initRandom)
    );
    */

    GBNAdaptive *gbnp = NULL;
    if (inputFileName)
        gbnp = new GBNAdaptive(inputFileName, img);
    else
        gbnp = new GBNAdaptive(N, img, initRandom);
    GBNAdaptive& gbn = *gbnp;

    if (sigma) gbn.setSigma(sigma);
    if (s) gbn.setTimeStep(s);
    if (outBlackRatio) gbn.setOutBlackRatio(outBlackRatio);
    if (frameBaseName) gbn.setFrameBaseName(frameBaseName);
    if (snapshotCondition) gbn.setSnaptshotCondition(snapshotCondition);
    if (frameFormat) gbn.setFrameFormat(frameFormat);
    if (W) gbn.setFrameSize(W, H);
    if (margin) gbn.setMargin(margin);
    clock_t t0 = clock();
    gbn.optimize(iterations, terminateOnNoMove);
    clock_t t1 = clock();
    double totalTime = (double)(t1 - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "\ndone! Total time = %.6fs\n", totalTime);
    // =========================================================================
    // Save point sets
    // =========================================================================
    while (optind < argc) { gbn.save(argv[optind++]); }

}
