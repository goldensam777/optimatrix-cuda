/*
 * test_scan1d_backward.cu — Gradient check du scan adjoint
 *
 * Strategie : differences finies centrales vs gradients analytiques.
 *
 *   grad_numerique(p_i) = (L(p + eps*e_i) - L(p - eps*e_i)) / (2*eps)
 *
 * Loss = sum(y)  ->  dy[t,d] = 1 pour tout t, d.
 *
 * Pour chaque parametre (x, A, B, C, dt), on compare le gradient
 * analytique (om_scan1d_backward) au gradient numerique.
 * On teste TOUS les elements pour les petites configurations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimatrix.h"

#define EPS     1e-3f
#define TOL     1e-1f   /* tolerance pour float32 + accumulation */

/* ── Helpers ─────────────────────────────────────────────────────── */

static void fill_random(float *buf, int n, float lo, float hi)
{
    for (int i = 0; i < n; i++)
        buf[i] = lo + ((float)rand() / RAND_MAX) * (hi - lo);
}

/* Lance un forward GPU et retourne sum(y) comme scalaire de loss. */
static float gpu_scan1d_loss(
    const float *h_x, const float *h_A,
    const float *h_B, const float *h_C, const float *h_dt,
    int L, int D, int M)
{
    float *d_x, *d_A, *d_B, *d_C, *d_dt, *d_y, *d_h;

    OM_CHECK(cudaMalloc(&d_x,  L*D     * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_A,  D*M     * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_B,  L*D*M   * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_C,  L*D*M   * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dt, L*D     * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y,  L*D     * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_h,  L*D*M   * sizeof(float)));

    OM_CHECK(cudaMemcpy(d_x,  h_x,  L*D   *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_A,  h_A,  D*M   *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_B,  h_B,  L*D*M *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_C,  h_C,  L*D*M *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_dt, h_dt, L*D   *sizeof(float), cudaMemcpyHostToDevice));

    om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);

    float *h_y = (float *)malloc(L * D * sizeof(float));
    OM_CHECK(cudaMemcpy(h_y, d_y, L*D*sizeof(float), cudaMemcpyDeviceToHost));

    float loss = 0.0f;
    for (int i = 0; i < L * D; i++) loss += h_y[i];

    free(h_y);
    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_dt); cudaFree(d_y); cudaFree(d_h);
    return loss;
}

/* ── Test principal ───────────────────────────────────────────────── */

static int test_scan1d_grad(int L, int D, int M)
{
    int nx  = L * D;
    int nA  = D * M;
    int nBc = L * D * M;
    int ndt = L * D;

    float *h_x  = (float *)malloc(nx  * sizeof(float));
    float *h_A  = (float *)malloc(nA  * sizeof(float));
    float *h_B  = (float *)malloc(nBc * sizeof(float));
    float *h_C  = (float *)malloc(nBc * sizeof(float));
    float *h_dt = (float *)malloc(ndt * sizeof(float));

    fill_random(h_x,  nx,  -0.5f, 0.5f);
    fill_random(h_A,  nA,  -0.3f, -0.05f); /* A negatif = stable */
    fill_random(h_B,  nBc, -0.5f, 0.5f);
    fill_random(h_C,  nBc, -0.5f, 0.5f);
    fill_random(h_dt, ndt,  0.1f, 0.5f);   /* dt positif */

    /* ── Forward GPU + backward analytique ─────────────────────── */
    float *d_x, *d_A, *d_B, *d_C, *d_dt, *d_y, *d_h;
    float *d_dx, *d_dA, *d_dB, *d_dC, *d_ddt, *d_dy;

    OM_CHECK(cudaMalloc(&d_x,   nx  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_A,   nA  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_B,   nBc * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_C,   nBc * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dt,  ndt * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y,   nx  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_h,   nBc * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dy,  nx  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dx,  nx  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dA,  nA  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dB,  nBc * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dC,  nBc * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_ddt, ndt * sizeof(float)));

    OM_CHECK(cudaMemcpy(d_x,  h_x,  nx  *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_A,  h_A,  nA  *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_B,  h_B,  nBc *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_C,  h_C,  nBc *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_dt, h_dt, ndt *sizeof(float), cudaMemcpyHostToDevice));

    /* dy = all-ones (loss = sum(y)) */
    float *h_ones = (float *)malloc(nx * sizeof(float));
    for (int i = 0; i < nx; i++) h_ones[i] = 1.0f;
    OM_CHECK(cudaMemcpy(d_dy, h_ones, nx*sizeof(float), cudaMemcpyHostToDevice));
    free(h_ones);

    om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);

    om_scan1d_backward(d_dy,
                       d_x, d_A, d_B, d_C, d_dt, d_h,
                       d_dx, d_dA, d_dB, d_dC, d_ddt,
                       L, D, M);

    /* Copier gradients analytiques sur CPU */
    float *h_dx  = (float *)malloc(nx  * sizeof(float));
    float *h_dA  = (float *)malloc(nA  * sizeof(float));
    float *h_dB  = (float *)malloc(nBc * sizeof(float));
    float *h_dC  = (float *)malloc(nBc * sizeof(float));
    float *h_ddt = (float *)malloc(ndt * sizeof(float));

    OM_CHECK(cudaMemcpy(h_dx,  d_dx,  nx  *sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_dA,  d_dA,  nA  *sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_dB,  d_dB,  nBc *sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_dC,  d_dC,  nBc *sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_ddt, d_ddt, ndt *sizeof(float), cudaMemcpyDeviceToHost));

    /* ── Differences finies ────────────────────────────────────── */
    float max_err_x = 0, max_err_A = 0, max_err_B = 0;
    float max_err_C = 0, max_err_dt = 0;

#define FD_CHECK(param_arr, size, max_err_var) do {          \
    for (int i = 0; i < (size); i++) {                       \
        float orig = (param_arr)[i];                         \
        (param_arr)[i] = orig + EPS;                         \
        float lp = gpu_scan1d_loss(h_x, h_A, h_B, h_C, h_dt, L, D, M); \
        (param_arr)[i] = orig - EPS;                         \
        float lm = gpu_scan1d_loss(h_x, h_A, h_B, h_C, h_dt, L, D, M); \
        (param_arr)[i] = orig;                               \
        float ng = (lp - lm) / (2.0f * EPS);                \
        float err = fabsf(ng - grad_arr[i]);                 \
        if (err > (max_err_var)) (max_err_var) = err;        \
    }                                                        \
} while(0)

    /* Gradient analytique pour chaque parametre */
    float *grad_arr;

    grad_arr = h_dx;
    FD_CHECK(h_x, nx, max_err_x);

    grad_arr = h_dA;
    FD_CHECK(h_A, nA, max_err_A);

    grad_arr = h_dB;
    FD_CHECK(h_B, nBc, max_err_B);

    grad_arr = h_dC;
    FD_CHECK(h_C, nBc, max_err_C);

    grad_arr = h_ddt;
    FD_CHECK(h_dt, ndt, max_err_dt);

#undef FD_CHECK

    int ok = (max_err_x  < TOL) && (max_err_A  < TOL) &&
             (max_err_B  < TOL) && (max_err_C  < TOL) &&
             (max_err_dt < TOL);

    printf("  Scan1D backward L=%-4d D=%-3d M=%-2d : "
           "dx=%.2e dA=%.2e dB=%.2e dC=%.2e ddt=%.2e  %s\n",
           L, D, M,
           max_err_x, max_err_A, max_err_B, max_err_C, max_err_dt,
           ok ? "OK" : "FAIL");

    /* Cleanup */
    cudaFree(d_x);  cudaFree(d_A);   cudaFree(d_B);   cudaFree(d_C);
    cudaFree(d_dt); cudaFree(d_y);   cudaFree(d_h);   cudaFree(d_dy);
    cudaFree(d_dx); cudaFree(d_dA);  cudaFree(d_dB);  cudaFree(d_dC);
    cudaFree(d_ddt);
    free(h_x);  free(h_A);  free(h_B);  free(h_C);  free(h_dt);
    free(h_dx); free(h_dA); free(h_dB); free(h_dC); free(h_ddt);

    return ok;
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void)
{
    printf("════════════════════════════════════════════════\n");
    printf("  TEST Scan1D backward (gradient check)\n");
    printf("════════════════════════════════════════════════\n\n");

    srand(42);
    int pass = 0, total = 0;

    pass += test_scan1d_grad(8,  4, 2); total++;
    pass += test_scan1d_grad(16, 8, 4); total++;
    pass += test_scan1d_grad(32, 8, 4); total++;

    printf("\n%d/%d tests OK\n", pass, total);
    return (pass == total) ? 0 : 1;
}
