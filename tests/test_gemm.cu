/*
 * test_gemm.cu — Tests GEMV et GEMM
 *
 * Compare GPU vs reference CPU naive.
 * Teste plusieurs tailles : petite, moyenne, non-alignee sur TILE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimatrix.h"

#define TOL 1e-3f

/* ── References CPU ──────────────────────────────────────────────── */

static void cpu_gemv(const float *A, const float *x, float *y, int m, int n)
{
    for (int i = 0; i < m; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++)
            s += A[i * n + j] * x[j];
        y[i] = s;
    }
}

static void cpu_gemm(const float *A, const float *B, float *C,
                     int m, int n, int k)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < k; p++)
                s += A[i * k + p] * B[p * n + j];
            C[i * n + j] = s;
        }
}

/* ── Utilitaires ─────────────────────────────────────────────────── */

static void fill_random(float *buf, int n)
{
    for (int i = 0; i < n; i++)
        buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
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

/* ── Tests ───────────────────────────────────────────────────────── */

static int test_gemv(int m, int n)
{
    float *h_A = (float *)malloc(m * n * sizeof(float));
    float *h_x = (float *)malloc(n * sizeof(float));
    float *h_y = (float *)malloc(m * sizeof(float));
    float *h_ref = (float *)malloc(m * sizeof(float));

    fill_random(h_A, m * n);
    fill_random(h_x, n);
    cpu_gemv(h_A, h_x, h_ref, m, n);

    float *d_A, *d_x, *d_y;
    OM_CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y, m * sizeof(float)));
    OM_CHECK(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

    om_gemv(d_A, d_x, d_y, m, n);
    OM_CHECK(cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_error(h_y, h_ref, m);
    int ok = err < TOL;
    printf("  GEMV %dx%d : max_err = %.2e  %s\n", m, n, err, ok ? "OK" : "FAIL");

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    free(h_A); free(h_x); free(h_y); free(h_ref);
    return ok;
}

static int test_gemm(int m, int n, int k)
{
    float *h_A = (float *)malloc(m * k * sizeof(float));
    float *h_B = (float *)malloc(k * n * sizeof(float));
    float *h_C = (float *)malloc(m * n * sizeof(float));
    float *h_ref = (float *)malloc(m * n * sizeof(float));

    fill_random(h_A, m * k);
    fill_random(h_B, k * n);
    cpu_gemm(h_A, h_B, h_ref, m, n, k);

    float *d_A, *d_B, *d_C;
    OM_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));
    OM_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    om_gemm(d_A, d_B, d_C, m, n, k);
    OM_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_error(h_C, h_ref, m * n);
    int ok = err < TOL;
    printf("  GEMM %dx%dx%d : max_err = %.2e  %s\n", m, n, k, err, ok ? "OK" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return ok;
}

int main(void)
{
    printf("=== Test GEMV / GEMM ===\n");
    srand(42);

    int pass = 0, total = 0;

    /* GEMV : tailles variees */
    pass += test_gemv(16, 16);    total++;
    pass += test_gemv(64, 64);    total++;
    pass += test_gemv(37, 53);    total++;  /* non-aligne */

    /* GEMM : tailles variees */
    pass += test_gemm(16, 16, 16);   total++;
    pass += test_gemm(64, 64, 64);   total++;
    pass += test_gemm(33, 47, 25);   total++;  /* non-aligne sur TILE */
    pass += test_gemm(128, 128, 128); total++;

    printf("\n%d/%d tests OK\n", pass, total);
    return (pass == total) ? 0 : 1;
}
