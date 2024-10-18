/*
 */


#define USE_DOUBLE                                                              // This must precede inclusion of spectrum.h

#include "spectrum-nd.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "getopt/getopt.h"
#include <string>
#include <fstream>

// =============================================================================
// Tex parametes
// =============================================================================

float ymin = 0;
float ymax = 4.2;
float xmin = 0;
float xmax = -1;
char *userInstructions = "";

// =============================================================================
// Plotting
// =============================================================================

Float plotRadialPower (
    const std::vector<Float> &p,                                                // List of points
    int n,                                                                      // Number of point sets
    int N,                                                                      // Number of points per set
    int dims,                                                                   // Number of dimensions
    int size,                                                                   // Number of frequencies along each dimension
    const char *fileName,                                                       // Output file name
    bool loglog = false,                                                        // Use a log-log scale
    bool allBins = false                                                        // Populate all distinct radii
) {
    // =========================================================================
    // Generate histogram
    // =========================================================================
    Float *p_gpu;
    int dataSize = p.size() * sizeof(Float);
    cudaMalloc(&p_gpu, dataSize);
    cudaMemcpy(p_gpu, p.data(), dataSize, cudaMemcpyHostToDevice);
    int base = size * 2 + 1;
    int volume(1);
    for (int dim = 0; dim < dims; dim++) {
        volume *= base;
    }
    std::vector<Float> spectrum(volume);
    Float *spectrum_gpu;
    cudaMalloc(&spectrum_gpu, volume * sizeof(Float));
    int blockSize = 256;
    int numBlocks = (volume + blockSize - 1) / blockSize;
    powerSpectrum_gpu<<<numBlocks, blockSize>>>(
        p_gpu, n, N, dims, size, volume, spectrum_gpu
    );
    cudaDeviceSynchronize();
    cudaMemcpy(
        spectrum.data(), spectrum_gpu,
               volume * sizeof(Float), cudaMemcpyDeviceToHost
    );
    cudaFree(p_gpu);
    cudaFree(spectrum_gpu);
    // =========================================================================
    // Populate 1D histogram
    // =========================================================================
    Float bnLoss = 0;
    int nbins = dims * size * size;
    std::vector<Float> rp(nbins, ZERO);
    std::vector<int> hits(nbins, 0);
    Float rrRef = std::pow(N, 2.0/dims);                                        // Square of radius of one ring
    for (int i = 0; i < volume; i++) {
        int tmp = i;
        int rr = 0;
        for (int dim = 0; dim < dims; dim++) {
            int f = tmp % base - size;
            rr += f * f;
            tmp /= base;
        }
        int binNo = allBins ? rr : std::sqrt(rr);
        if (binNo < nbins) {
            rp[binNo] += spectrum[i];
            hits[binNo]++;
        }
        if (rr != 0 && rr < rrRef && spectrum[i] < 1) {
            bnLoss -= log(spectrum[i]);
        }
    }
    bnLoss /= N;
    for (int binNo = 0; binNo < nbins; binNo++) {
        if (hits[binNo]) rp[binNo] /= hits[binNo];
    }
    // =========================================================================
    // Plot
    // =========================================================================
    if (fileName) {
        FILE *file = fopen(fileName, "w");
        Float unit = std::pow(N, -1.0/dims);                                    // Unit frequency relative to number of points per dimension
        Float maxFrequecny = unit * sqrt(nbins - 1);
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
                    //"    0        100\n"
                );
            }
            for (int binNo = 1; binNo < nbins; binNo++) {
                Float r = allBins ? std::sqrt(binNo) : binNo;
                if (unit * r <= xmax && hits[binNo]) {
                    fprintf(
                        file, "    %19.17f %21.17f\n", unit * r, rp[binNo]
                    );
                }
            }
            fprintf(file, "  };\n");
            if (!loglog) {
                fprintf(
                    file,
                    "  \\addplot[draw=none, domain=%f:%f, name path=B] {0};\n"
                    "  \\addplot[blue!10] fill between[of=A and B];\n"
                    "  \\addplot[help lines,dashed, domain=0:%f] "
                    "{1};\n",
                    unit, maxFrequecny, maxFrequecny
                );
            }
            fprintf(
                file, "%s\n\\end{axis};\n\\end{tikzpicture}\n"
                "\\end{document}\n",
                userInstructions
            );
        }
        else if (strstr(fileName, ".txt")) {
            for (int binNo = 1; binNo < nbins; binNo++) {
                Float r = allBins ? std::sqrt(binNo) : binNo;
                if (hits[binNo]) {
                    fprintf(
                        file, "    %19.17f %21.17f\n", unit * r, rp[binNo]
                    );
                }
            }
        }
        fclose(file);
    }
    return bnLoss;
}

/******************************************************************************/
/******************************************************************************/

const char *USAGE_MESSAGE = "Usage: %s [options] file1, [file2, ..]\n"
"Options:\n"
"   -c <cycles>         Default is 2\n"
"   -h <harmonics>      Override cycles to directly specify number of "
                        "harmonics\n"
"   -o <fileName>       Output to fileName\n"
"   -l                  Use log-log scale for radial power spectrum plot\n"
"   -a                  Use all distinct frequencies in histogram\n"
"   -2                  Supplied point files are 2D, skip number of dims\n"
"   -x <minFrequency>   Default is 0\n"
"   -X <maxFrequency>   Default is largest harmonic\n"
"   -y <minAmplitude>   Default is 0\n"
"   -Y <maxAmplitude>   Default is 4.2\n"
"   -t <tikz>           Append tikz instructions to plot\n"
"   -q                  Quiet: suppress messages\n"
"   -e                  Report bnLoss\n"
;

int main(int argc, char** argv) {
    int opt;                                                                    // For use by getopt, the command line options utility
    char *outputFileName = NULL;
    Float cycles = 1;
    int size = 0;
    int dims = 2;
    bool loglog = false;
    bool allBins = false;
    bool is2D = false;
    bool quiet = false;
    bool reportEnergy = false;
    while ((opt = getopt(argc, argv, "o:c:lah:2x:X:y:Y:t:qe")) != -1) {
        switch (opt) {
            case 'o': outputFileName = optarg; break;
            case 'c': cycles = atof(optarg); break;
            case 'h': size = atoi(optarg) + 1; break;
            case 'l': loglog = true; break;
            case 'a': allBins = true; break;
            case '2': is2D = true; break;
            case 'x': xmin = atof(optarg); break;
            case 'X': xmax = atof(optarg); break;
            case 'y': ymin = atof(optarg); break;
            case 'Y': ymax = atof(optarg); break;
            case 't': userInstructions = optarg; break;
            case 'q': quiet = true; break;
            case 'e': reportEnergy = true; break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    if (optind >= argc) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
    }
    int n = argc - optind;
    int N;
    std::fstream file(argv[optind]);
    file >> N;
    if (!is2D) file >> dims;
    file.close();
    std::vector<Float> p(n * N * dims);
    for (int fileNo = 0, pointNo = 0; fileNo < n; fileNo++) {
        file.open(argv[optind + fileNo]);
        int N_file;
        int dims_file = 2;
        file >> N_file;
        if (!is2D) file >> dims_file;
        if (N_file != N || dims_file != dims) {
            fprintf(
                stderr,
                "Number of points or dimensions in %s is different from %s\n",
                argv[optind], argv[optind + fileNo]
            );
            exit(1);
        }
        for (int i = 0; i < N; i++) {
            for (int dim = 0; dim < dims; dim++) {
                file >> p[pointNo * dims + dim];
            }
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
    if (!size) size = cycles * pow(N, 1.0 / dims);
    clock_t t0 = clock();
    Float bnLoss = plotRadialPower(
        p, n, N, dims, size, outputFileName, loglog, allBins
    );
    clock_t t1 = clock();
    Float totalTime = (Float)(t1 - t0) / CLOCKS_PER_SEC;
    if (!quiet) {
        fprintf(
            stderr, "Total time = %.6fs, averaged %d sets of %d points each\n",
            totalTime, n, N
        );
    }
    if (reportEnergy) printf("%f\n", bnLoss);
}
