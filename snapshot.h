/*
 * Take a snapshot of a point distribution and save as png image
 */

#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include "float.h"
#include <cmath>

// =============================================================================

// =============================================================================


void snapshot(
    Float *p_gpu,
    int N,
    int w, int h,
    Float coverage,
    const char *baseName,
    int frameNo,
    Float scale = 1,
    bool useNegative = false
) {
    std::vector<Float> p(2 * N);
    cudaMemcpy(
        p.data(), p_gpu, 2 * N * sizeof(Float), cudaMemcpyDeviceToHost
    );
    cairo_surface_t *surface = cairo_image_surface_create(
        CAIRO_FORMAT_RGB24, w, h
    );
    cairo_t *cr = cairo_create(surface);
    int n = std::max(w, h);
    double area = double(h * w) / (n * n);
    const double r = sqrt(area * coverage / (N * M_PI));
    cairo_identity_matrix(cr);
    cairo_translate(cr, 0, h);
    cairo_scale(cr, n, -n);
    if (useNegative) {
        cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
        cairo_paint(cr);
        cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1);
    }
    else {
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
        cairo_paint(cr);
        cairo_set_source_rgba(cr, 0, 0, 0, 1);
    }
    for (int i = 0; i < N; ++i) {
        cairo_arc(cr, scale * p[2 * i], scale * p[2 * i + 1], r, 0, 2 * M_PI);
        cairo_fill(cr);
    }
    cairo_destroy (cr);
    char fileName[100];
    sprintf(fileName, "%s%07d.png", baseName, frameNo);
    cairo_surface_write_to_png (surface, fileName);
    cairo_surface_destroy (surface);
}

void snapshotEPS(
    Float *p_gpu,
    int N,
    int w, int h,
    Float coverage,
    const char *baseName,
    int frameNo,
    bool useNegative = false
) {
    std::vector<Float> p(2 * N);
    cudaMemcpy(
        p.data(), p_gpu, 2 * N * sizeof(Float), cudaMemcpyDeviceToHost
    );
    char fileName[100];
    sprintf(fileName, "%s%07d.eps", baseName, frameNo);
    FILE *file = fopen(fileName, "w");
    int n = std::max(w, h);
    fprintf(
        file,
        "%%!PS-Adobe-3.0 EPSF-3.0\n"
        "%%%%BoundingBox: 0 0 %d %d\n"
        "/N %d def\n"
        "/w %d def\n"
        "/h %d def\n"
        "/n %d def\n"
        "/coverage %f def\n"
        "/useNegative %s def\n"
        "/PI 3.141592654 def\n"
        "/r w h mul n n mul div N div coverage mul PI div sqrt def\n"
        "/p {r 0 360 arc fill} def\n"
        "0 setlinewidth\n"
        "n 16 mul dup scale\n"
        "useNegative {0 0 w n div h n div rectfill 1 setgray} if\n",
        16 * w, 16 * h, N, w, h, n, coverage,
        useNegative ? "true" : "false"
    );
    Float nInv = 1.0 / n;
    for (int i = 0; i < N; i++) {
        fprintf(
            file, "%0.17f %0.17f p\n", nInv * p[2 * i], nInv * p[2 * i + 1]
        );
    }
    fprintf(file, "showpage\n");
    fclose(file);
}


#endif
