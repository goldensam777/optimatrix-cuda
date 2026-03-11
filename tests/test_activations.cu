/*
 * test_activations.cu — Tests unitaires pour ReLU, Sigmoid, SiLU, Softplus
 *
 * Strategie : on calcule la reference sur CPU, on compare avec le GPU.
 * Tolerance : 1e-5 (float32 + exp/log).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimatrix.h"

#define N 1024
#define TOL 1e-5f

/* ── References CPU ──────────────────────────────────────────────── */

static float cpu_relu(float x)     { return fmaxf(x, 0.0f); }
static float cpu_sigmoid(float x)  { return 1.0f / (1.0f + expf(-x)); }
static float cpu_silu(float x)     { return x * cpu_sigmoid(x); }
static float cpu_softplus(float x) { return log1pf(expf(x)); }

/* ── Utilitaires ─────────────────────────────────────────────────── */

static void fill_random(float *buf, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
        buf[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;  /* [-2, 2] */
}

static int check(const char *name, const float *gpu, const float *ref, int n)
{
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(gpu[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    int ok = max_err < TOL;
    printf("  %-10s : max_err = %.2e  %s\n", name, max_err, ok ? "OK" : "FAIL");
    return ok;
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Test Activations (N=%d) ===\n", N);

    /* Alloc CPU */
    float *h_in  = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));
    fill_random(h_in, N);

    /* Alloc GPU */
    float *d_in, *d_out;
    OM_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    OM_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int pass = 0, total = 0;

    /* ── ReLU ── */
    for (int i = 0; i < N; i++) h_ref[i] = cpu_relu(h_in[i]);
    om_relu(d_in, d_out, N);
    OM_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    pass += check("ReLU", h_out, h_ref, N); total++;

    /* ── Sigmoid ── */
    for (int i = 0; i < N; i++) h_ref[i] = cpu_sigmoid(h_in[i]);
    om_sigmoid(d_in, d_out, N);
    OM_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    pass += check("Sigmoid", h_out, h_ref, N); total++;

    /* ── SiLU ── */
    for (int i = 0; i < N; i++) h_ref[i] = cpu_silu(h_in[i]);
    om_silu(d_in, d_out, N);
    OM_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    pass += check("SiLU", h_out, h_ref, N); total++;

    /* ── Softplus ── */
    for (int i = 0; i < N; i++) h_ref[i] = cpu_softplus(h_in[i]);
    om_softplus(d_in, d_out, N);
    OM_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    pass += check("Softplus", h_out, h_ref, N); total++;

    /* ── Bilan ── */
    printf("\n%d/%d tests OK\n", pass, total);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);

    return (pass == total) ? 0 : 1;
}
