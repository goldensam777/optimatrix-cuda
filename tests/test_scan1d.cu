/*
 * test_scan1d.cu — Test et benchmark du selective scan 1D
 *
 * Compare les deux implementations :
 *   A) Sequentielle : 1 thread par (d,m), boucle sur L
 *   B) Blelloch     : prefix scan parallele en O(log L)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimatrix.h"

#define TOL 1e-3f

/* ── Reference CPU ───────────────────────────────────────────────── */

static void cpu_scan1d(
    const float *x,  const float *A,
    const float *B,  const float *C,
    const float *dt,
    float *y,        float *h,
    int L, int D, int M)
{
    /* Zeroiser y */
    for (int i = 0; i < L * D; i++) y[i] = 0.0f;

    for (int d = 0; d < D; d++) {
        for (int m = 0; m < M; m++) {
            int dm = d * M + m;
            float h_prev = 0.0f;
            for (int t = 0; t < L; t++) {
                int t_d  = t * D + d;
                int t_dm = t * D * M + dm;
                float dt_val = dt[t_d];
                float a = expf(dt_val * A[dm]);
                float b = dt_val * B[t_dm] * x[t_d];
                float h_cur = a * h_prev + b;
                h[t_dm] = h_cur;
                y[t_d] += C[t_dm] * h_cur;
                h_prev = h_cur;
            }
        }
    }
}

/* ── Utilitaires ─────────────────────────────────────────────────── */

static void fill_random(float *buf, int n, float lo, float hi)
{
    for (int i = 0; i < n; i++)
        buf[i] = lo + ((float)rand() / RAND_MAX) * (hi - lo);
}

static float max_error(const float *a, const float *b, int n)
{
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/* ── Correctness ─────────────────────────────────────────────────── */

static int test_correctness(int L, int D, int M)
{
    int P = L;
    float *h_x  = (float *)malloc(P * D * sizeof(float));
    float *h_A  = (float *)malloc(D * M * sizeof(float));
    float *h_B  = (float *)malloc(P * D * M * sizeof(float));
    float *h_C  = (float *)malloc(P * D * M * sizeof(float));
    float *h_dt = (float *)malloc(P * D * sizeof(float));
    float *h_y_ref = (float *)calloc(P * D, sizeof(float));
    float *h_h_ref = (float *)calloc(P * D * M, sizeof(float));
    float *h_y_gpu = (float *)malloc(P * D * sizeof(float));

    srand(42);
    fill_random(h_x,  P * D,         -1.0f, 1.0f);
    fill_random(h_A,  D * M,         -0.5f, -0.01f);
    fill_random(h_B,  P * D * M,     -0.5f, 0.5f);
    fill_random(h_C,  P * D * M,     -0.5f, 0.5f);
    fill_random(h_dt, P * D,          0.01f, 0.5f);

    cpu_scan1d(h_x, h_A, h_B, h_C, h_dt,
               h_y_ref, h_h_ref, L, D, M);

    float *d_x, *d_A, *d_B, *d_C, *d_dt, *d_y, *d_h;
    OM_CHECK(cudaMalloc(&d_x,  P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_A,  D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_B,  P * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_C,  P * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dt, P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y,  P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_h,  P * D * M * sizeof(float)));

    OM_CHECK(cudaMemcpy(d_x,  h_x,  P*D*sizeof(float),     cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_A,  h_A,  D*M*sizeof(float),     cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_B,  h_B,  P*D*M*sizeof(float),   cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_C,  h_C,  P*D*M*sizeof(float),   cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_dt, h_dt, P*D*sizeof(float),     cudaMemcpyHostToDevice));

    om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);
    OM_CHECK(cudaMemcpy(h_y_gpu, d_y, P * D * sizeof(float),
                        cudaMemcpyDeviceToHost));

    float err = max_error(h_y_gpu, h_y_ref, P * D);
    int ok = err < TOL;
    printf("  Scan1D L=%-4d D=%-3d M=%-2d : max_err=%.2e  %s\n",
           L, D, M, err, ok ? "OK" : "FAIL");

    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C); cudaFree(d_dt); cudaFree(d_y); cudaFree(d_h);
    free(h_x); free(h_A); free(h_B); free(h_C);
    free(h_dt); free(h_y_ref); free(h_h_ref); free(h_y_gpu);

    return ok;
}

/* ── Benchmark ───────────────────────────────────────────────────── */

static float bench(int L, int D, int M, int iters)
{
    int P = L;
    float *d_x, *d_A, *d_B, *d_C, *d_dt, *d_y, *d_h;
    OM_CHECK(cudaMalloc(&d_x,  P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_A,  D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_B,  P * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_C,  P * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dt, P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y,  P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_h,  P * D * M * sizeof(float)));

    /* Warmup */
    om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C); cudaFree(d_dt); cudaFree(d_y); cudaFree(d_h);

    return ms / iters;
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void)
{
    printf("════════════════════════════════════════\n");
    printf("  CORRECTNESS — Scan 1D\n");
    printf("════════════════════════════════════════\n\n");

    int pass = 0, total = 0;

    pass += test_correctness(16,  8,  4); total++;
    pass += test_correctness(64,  16, 8); total++;
    pass += test_correctness(128, 32, 8); total++;
    pass += test_correctness(512, 64, 16); total++;
    pass += test_correctness(1024,128, 16); total++;

    printf("\n%d/%d tests OK\n\n", pass, total);

    printf("════════════════════════════════════════\n");
    printf("  BENCHMARK — Scan 1D (temps moyen ms)\n");
    printf("════════════════════════════════════════\n\n");

    printf("  %-20s  %s\n", "Config", "Temps");
    printf("  ──────────────────────────────────────\n");

    typedef struct { int L, D, M, iters; } BC;
    BC configs[] = {
        { 64,   32,  8,  200},
        {128,   64,  8,  100},
        {256,  128, 16,   50},
        {512,  128, 16,   20},
        {1024, 128, 16,   10},
    };

    for (int c = 0; c < 5; c++) {
        BC *b = &configs[c];
        float ms = bench(b->L, b->D, b->M, b->iters);
        printf("  L=%-4d D=%-3d M=%-2d   %8.3f ms\n",
               b->L, b->D, b->M, ms);
    }

    printf("\n");
    return (pass == total) ? 0 : 1;
}
