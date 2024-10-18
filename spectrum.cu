/*
 */

// This must precede inclusion of spectrum.h
#define USE_DOUBLE

#include "spectrum.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "getopt/getopt.h"
#include <string>
#include <fstream>

Float scale = 0.5;                                                              // This is the default of PSA
bool loglog = false;
bool allBins = false;

const char *USAGE_MESSAGE = "Usage: %s [options] file1, [file2, ..]\n"
"Options:\n"
"   -c <cycles>         Default is 2\n"
"   -o <fileName>       Plot to fileName\n"
"   -w <width>          Default is 10 * sqrt(N)\n"
"   -W                  The points are weighted. "
                        "Format: x y weight\n"
"   -r <fileName>       Outputs a radial power plot\n"
"   -s <freq. scale>    For radial power; default is 0.5, as in PSA\n"
"   -l                  Use log-log scale for radial power spectrum plot\n"
"   -x <minFrequency>   Default is 0\n"
"   -X <maxFrequency>   Default is largest harmonic\n"
"   -y <minAmplitude>   Default is 0\n"
"   -Y <maxAmplitude>   Default is 4.2\n"
"   -t <tikz>           Append tikz instructions to plot\n"
"   -e                  Report evacuated energy\n"
"   -A <average>        Set the DC peak\n"
;

int main(int argc, char** argv) {
    int opt;                                                                    // For use by getopt, the command line options utility
    std::string plotFileName;
    std::string rpFileName;
    Float cycles = 5;
    int width = 0;
    bool reportEnergy = false;
    bool weighted = false;
    while ((opt = getopt(argc, argv, "c:o:r:w:s:lax:X:y:Y:t:eWA:")) != -1) {
        switch (opt) {
            case 'c': cycles = atof(optarg); printf("cycles = %f\n", (float)cycles); break;
            case 'o': plotFileName = optarg; printf("plotFileName = %s\n", plotFileName.c_str()); break;
            case 'r': rpFileName = optarg; printf("rpFileName = %s\n", rpFileName.c_str()); break;
            case 'w': width = atoi(optarg); printf("width = %i\n", width); break;
            case 's': scale = atof(optarg); printf("scale = %f\n", (float)scale); break;
            case 'l': loglog = true; printf("loglog = true\n"); break;
            case 'a': allBins = true; printf("allBins = true\n"); break;
            case 'x': xmin = atof(optarg); printf("xmin = %f\n", xmin); break;
            case 'X': xmax = atof(optarg); printf("xmax = %f\n", xmax); break;
            case 'y': ymin = atof(optarg); printf("ymin = %f\n", ymin); break;
            case 'Y': ymax = atof(optarg); printf("xmax = %f\n", ymax); break;
            case 't': userInstructions = optarg; printf("%s\n",userInstructions); break;
            case 'e': reportEnergy = true; printf("reportEnergy = true\n"); break;
            case 'W': weighted = true; printf("weighted = true\n"); break;
            case 'A': average = atof(optarg); printf("average = %f\n", average); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    if (optind >= argc) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
    }
    int n = argc - optind;
    int N = 0;
    std::fstream file(argv[optind]);
    printf("opening argv[%i] \"%s\"\n", optind, argv[optind]);
    file >> N;
    file.close();
    std::vector<PointFloat> p(n * N);
    std::vector<Float> wt;
    if (weighted) wt.resize(n * N);
    for (int fileNo = 0, pointNo = 0; fileNo < n; fileNo++) {
        file.open(argv[optind + fileNo]);
        printf("opening argv[%i] \"%s\"\n", optind + fileNo, argv[optind + fileNo]);
        int N_fileNo = 0;
        file >> N_fileNo;
        if (N_fileNo != N) {
            fprintf(
                stderr, "Number of points in %s is different from %s (%i vs %i)\n",
                argv[optind], argv[optind + fileNo], N_fileNo, N
            );
            exit(1);
        }
        for (int i = 0; i < N; i++) {
            file >> p[pointNo].x >> p[pointNo].y;
            if (weighted) file >> wt[pointNo];
            if (file.eof()) {
                fprintf(
                    stderr, "Failed to load all points from %s\n",
                    argv[optind + fileNo]
                );
                exit(2);
            }
            pointNo++;
        }
        file.close();
    }
    if (!width) width = 2 * cycles * sqrt(N);
    if (width & 1) width -= 1;
    clock_t t0 = clock();
    PointFloat *p_gpu;
    Float *wt_gpu = NULL;
    int dataSize = n * N * sizeof(PointFloat);
    cudaMalloc(&p_gpu, dataSize);
    cudaMemcpy(p_gpu, p.data(), dataSize, cudaMemcpyHostToDevice);
    if (weighted) {
        cudaMalloc(&wt_gpu, n * N * sizeof(Float));
        cudaMemcpy(wt_gpu, p.data(),
        n * N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    std::vector<Float> spectrum = powerSpectrum(p_gpu, wt_gpu, n, N, width);
    cudaFree(p_gpu);
    if (weighted) cudaFree(wt_gpu);
    if (!plotFileName.empty()) {
        plotSpectrum(spectrum, width, plotFileName.c_str());
    }
    if (!rpFileName.empty()) {
        plotRadialPower(
            spectrum, width, rpFileName.c_str(), N, scale, loglog, allBins
        );
    }
    if (reportEnergy) {
        printf("%f\n", bnLoss(spectrum, width, N));
    }
    clock_t t1 = clock();
    Float totalTime = (Float)(t1 - t0) / CLOCKS_PER_SEC;
    fprintf(
        stderr, "Total time = %.6fs, averaged %d sets of %d points each\n",
        totalTime, n, N
    );
}
