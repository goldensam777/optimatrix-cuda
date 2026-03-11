/*
 * test_hadamard.cu — Test unitaire pour le produit element par element
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimatrix.h"

#define N 2048
#define TOL 1e-6f

int main(void)
{
    printf("=== Test Hadamard (N=%d) ===\n", N);

    /* Alloc CPU */
    float *h_a   = (float *)malloc(N * sizeof(float));
    float *h_b   = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++) {
        h_a[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_b[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_ref[i] = h_a[i] * h_b[i];
    }

    /* Alloc GPU */
    float *d_a, *d_b, *d_out;
    OM_CHECK(cudaMalloc(&d_a,   N * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_b,   N * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    OM_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Run */
    om_hadamard(d_a, d_b, d_out, N);
    OM_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify */
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    int ok = max_err < TOL;
    printf("  Hadamard : max_err = %.2e  %s\n", max_err, ok ? "OK" : "FAIL");
    printf("\n%d/1 tests OK\n", ok ? 1 : 0);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);
    free(h_ref);

    return ok ? 0 : 1;
}
