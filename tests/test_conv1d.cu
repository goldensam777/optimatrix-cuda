/*
 * test_conv1d.cu — Test de la conv1d depthwise causale
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimatrix.h"

#define TOL 1e-5f

/* ── Reference CPU ───────────────────────────────────────────────── */

static void cpu_conv1d(
    const float *input, const float *weight, const float *bias,
    float *output, int L, int D, int K)
{
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                if (t - k >= 0)
                    acc += weight[k * D + d] * input[(t - k) * D + d];
            }
            if (bias) acc += bias[d];
            output[t * D + d] = acc;
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

/* ── Test ────────────────────────────────────────────────────────── */

static int test_conv1d(int L, int D, int K, int use_bias)
{
    int in_size  = L * D;
    int w_size   = K * D;
    int out_size = L * D;

    float *h_in  = (float *)malloc(in_size  * sizeof(float));
    float *h_w   = (float *)malloc(w_size   * sizeof(float));
    float *h_b   = use_bias ? (float *)malloc(D * sizeof(float)) : NULL;
    float *h_out = (float *)malloc(out_size * sizeof(float));
    float *h_ref = (float *)malloc(out_size * sizeof(float));

    fill_random(h_in, in_size);
    fill_random(h_w, w_size);
    if (h_b) fill_random(h_b, D);

    cpu_conv1d(h_in, h_w, h_b, h_ref, L, D, K);

    float *d_in, *d_w, *d_b = NULL, *d_out;
    OM_CHECK(cudaMalloc(&d_in,  in_size  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_w,   w_size   * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    if (use_bias) OM_CHECK(cudaMalloc(&d_b, D * sizeof(float)));

    OM_CHECK(cudaMemcpy(d_in, h_in, in_size * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_w,  h_w,  w_size  * sizeof(float), cudaMemcpyHostToDevice));
    if (use_bias)
        OM_CHECK(cudaMemcpy(d_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice));

    om_conv1d(d_in, d_w, d_b, d_out, L, D, K);
    OM_CHECK(cudaMemcpy(h_out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_error(h_out, h_ref, out_size);
    int ok = err < TOL;
    printf("  Conv1D L=%-4d D=%-4d K=%d bias=%d : max_err=%.2e  %s\n",
           L, D, K, use_bias, err, ok ? "OK" : "FAIL");

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    if (d_b) cudaFree(d_b);
    free(h_in); free(h_w); free(h_out); free(h_ref);
    if (h_b) free(h_b);
    return ok;
}

int main(void)
{
    printf("=== Test Conv1D depthwise causale ===\n\n");
    srand(42);

    int pass = 0, total = 0;

    pass += test_conv1d(4,   3,  2, 0); total++;   /* mini */
    pass += test_conv1d(128, 64, 4, 1); total++;   /* standard */
    pass += test_conv1d(256, 128,3, 0); total++;   /* grand D */
    pass += test_conv1d(512, 16, 4, 1); total++;   /* grand L */
    pass += test_conv1d(64,  64, 8, 0); total++;   /* grand K */

    printf("\n%d/%d tests OK\n", pass, total);
    return (pass == total) ? 0 : 1;
}
