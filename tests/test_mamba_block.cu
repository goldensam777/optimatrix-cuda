/*
 * test_mamba_block.cu — Test du MambaBlock forward complet
 *
 * Strategie : reference CPU naive vs GPU.
 * On ne gradient-check pas ici (pas de backward encore),
 * on verifie juste que le pipeline produit des sorties correctes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimatrix.h"

#define TOL 1e-2f  /* tolerance plus large : accumulation d'erreurs sur tout le pipeline */

/* ── Declarations internes ───────────────────────────────────────── */

typedef struct {
    int L, dim, D, M, K;
    const float *W_in, *W_dt, *W_B, *W_C;
    const float *A, *conv_w, *conv_b, *W_out;
} MambaParams;

extern "C" void om_mamba_block_forward(
    const float *d_x_in, float *d_y_out,
    const MambaParams *p);

/* ── Reference CPU ───────────────────────────────────────────────── */

static float silu(float x) { return x / (1.0f + expf(-x)); }
static float softplus(float x) { return log1pf(expf(x)); }

static void cpu_mamba_block(
    const float *x_in, float *y_out,
    const float *W_in, const float *W_dt,
    const float *W_B,  const float *W_C,
    const float *A,    const float *conv_w,
    const float *W_out,
    int L, int dim, int D, int M, int K)
{
    /* Alloc intermediaires */
    float *xz     = (float *)calloc(L * 2 * D,  sizeof(float));
    float *x      = (float *)calloc(L * D,       sizeof(float));
    float *z      = (float *)calloc(L * D,       sizeof(float));
    float *dt     = (float *)calloc(L * D,       sizeof(float));
    float *B      = (float *)calloc(L * D * M,   sizeof(float));
    float *C      = (float *)calloc(L * D * M,   sizeof(float));
    float *y_scan = (float *)calloc(L * D,       sizeof(float));
    float *h_buf  = (float *)calloc(L * D * M,   sizeof(float));

    /* Etape 1 : xz = x_in @ W_in^T */
    for (int t = 0; t < L; t++)
        for (int d = 0; d < 2 * D; d++) {
            float s = 0.0f;
            for (int k = 0; k < dim; k++)
                s += x_in[t * dim + k] * W_in[d * dim + k];
            xz[t * 2 * D + d] = s;
        }

    /* Etape 2 : split + SiLU(x) */
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++) {
            x[t * D + d] = silu(xz[t * 2 * D + d]);
            z[t * D + d] = xz[t * 2 * D + D + d];
        }

    /* Etape 3 : conv1d causal sur x (in-place) */
    float *x_conv = (float *)calloc(L * D, sizeof(float));
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                if (t - k >= 0)
                    acc += conv_w[k * D + d] * x[(t - k) * D + d];
            x_conv[t * D + d] = acc;
        }
    for (int i = 0; i < L * D; i++) x[i] = x_conv[i];
    free(x_conv);

    /* Etape 4 : projections dt, B, C */
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++) {
            float s = 0.0f;
            for (int k = 0; k < dim; k++)
                s += x_in[t * dim + k] * W_dt[d * dim + k];
            dt[t * D + d] = softplus(s);
        }
    for (int t = 0; t < L; t++)
        for (int dm = 0; dm < D * M; dm++) {
            float sb = 0.0f, sc = 0.0f;
            for (int k = 0; k < dim; k++) {
                sb += x_in[t * dim + k] * W_B[dm * dim + k];
                sc += x_in[t * dim + k] * W_C[dm * dim + k];
            }
            B[t * D * M + dm] = sb;
            C[t * D * M + dm] = sc;
        }

    /* Etape 5 : selective scan 1D */
    for (int d = 0; d < D; d++)
        for (int m = 0; m < M; m++) {
            int dm = d * M + m;
            float h_prev = 0.0f;
            for (int t = 0; t < L; t++) {
                int t_d  = t * D + d;
                int t_dm = t * D * M + dm;
                float a   = expf(dt[t_d] * A[dm]);
                float b   = dt[t_d] * B[t_dm] * x[t_d];
                float h_t = a * h_prev + b;
                h_buf[t_dm] = h_t;
                y_scan[t_d] += C[t_dm] * h_t;
                h_prev = h_t;
            }
        }

    /* Etape 6 : gating */
    float *y_gate = (float *)calloc(L * D, sizeof(float));
    for (int i = 0; i < L * D; i++)
        y_gate[i] = y_scan[i] * silu(z[i]);

    /* Etape 7 : projection sortie */
    for (int t = 0; t < L; t++)
        for (int o = 0; o < dim; o++) {
            float s = 0.0f;
            for (int d = 0; d < D; d++)
                s += y_gate[t * D + d] * W_out[o * D + d];
            y_out[t * dim + o] = s;
        }

    free(xz); free(x); free(z); free(dt);
    free(B);  free(C); free(y_scan); free(h_buf); free(y_gate);
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

/* ── Test ────────────────────────────────────────────────────────── */

static int test_mamba(int L, int dim, int D, int M, int K)
{
    /* Alloc CPU */
    float *h_x_in = (float *)malloc(L * dim    * sizeof(float));
    float *h_W_in = (float *)malloc(2*D * dim  * sizeof(float));
    float *h_W_dt = (float *)malloc(D * dim    * sizeof(float));
    float *h_W_B  = (float *)malloc(D*M * dim  * sizeof(float));
    float *h_W_C  = (float *)malloc(D*M * dim  * sizeof(float));
    float *h_A    = (float *)malloc(D * M      * sizeof(float));
    float *h_cw   = (float *)malloc(K * D      * sizeof(float));
    float *h_W_ou = (float *)malloc(dim * D    * sizeof(float));
    float *h_ref  = (float *)calloc(L * dim,     sizeof(float));
    float *h_gpu  = (float *)malloc(L * dim    * sizeof(float));

    srand(42);
    fill_random(h_x_in, L * dim,      -0.5f, 0.5f);
    fill_random(h_W_in, 2*D * dim,    -0.1f, 0.1f);
    fill_random(h_W_dt, D * dim,      -0.1f, 0.1f);
    fill_random(h_W_B,  D*M * dim,    -0.1f, 0.1f);
    fill_random(h_W_C,  D*M * dim,    -0.1f, 0.1f);
    fill_random(h_A,    D * M,        -0.5f, -0.01f);  /* A < 0 = stable */
    fill_random(h_cw,   K * D,        -0.5f, 0.5f);
    fill_random(h_W_ou, dim * D,      -0.1f, 0.1f);

    /* Reference CPU */
    cpu_mamba_block(h_x_in, h_ref,
                    h_W_in, h_W_dt, h_W_B, h_W_C,
                    h_A, h_cw, h_W_ou,
                    L, dim, D, M, K);

    /* GPU — alloc et transfert */
    float *d_x_in, *d_W_in, *d_W_dt, *d_W_B, *d_W_C;
    float *d_A, *d_cw, *d_W_ou, *d_y_out;

    OM_CHECK(cudaMalloc(&d_x_in, L*dim    * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_W_in, 2*D*dim  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_W_dt, D*dim    * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_W_B,  D*M*dim  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_W_C,  D*M*dim  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_A,    D*M      * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_cw,   K*D      * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_W_ou, dim*D    * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y_out,L*dim    * sizeof(float)));

    OM_CHECK(cudaMemcpy(d_x_in, h_x_in, L*dim   *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_W_in, h_W_in, 2*D*dim *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_W_dt, h_W_dt, D*dim   *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_W_B,  h_W_B,  D*M*dim *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_W_C,  h_W_C,  D*M*dim *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_A,    h_A,    D*M     *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_cw,   h_cw,   K*D     *sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_W_ou, h_W_ou, dim*D   *sizeof(float), cudaMemcpyHostToDevice));

    MambaParams p = {L, dim, D, M, K,
                     d_W_in, d_W_dt, d_W_B, d_W_C,
                     d_A, d_cw, NULL, d_W_ou};

    om_mamba_block_forward(d_x_in, d_y_out, &p);
    OM_CHECK(cudaMemcpy(h_gpu, d_y_out, L*dim*sizeof(float),
                        cudaMemcpyDeviceToHost));

    float err = max_error(h_gpu, h_ref, L * dim);
    int ok = err < TOL;
    printf("  MambaBlock L=%-4d dim=%-3d D=%-3d M=%-2d K=%d : "
           "max_err=%.2e  %s\n",
           L, dim, D, M, K, err, ok ? "OK" : "FAIL");

    cudaFree(d_x_in); cudaFree(d_W_in); cudaFree(d_W_dt);
    cudaFree(d_W_B);  cudaFree(d_W_C);  cudaFree(d_A);
    cudaFree(d_cw);   cudaFree(d_W_ou); cudaFree(d_y_out);
    free(h_x_in); free(h_W_in); free(h_W_dt);
    free(h_W_B);  free(h_W_C);  free(h_A);
    free(h_cw);   free(h_W_ou); free(h_ref); free(h_gpu);

    return ok;
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void)
{
    printf("════════════════════════════════════════════════\n");
    printf("  TEST MambaBlock forward complet\n");
    printf("════════════════════════════════════════════════\n\n");

    int pass = 0, total = 0;

    pass += test_mamba(8,  16, 8,  4, 2); total++;
    pass += test_mamba(16, 32, 16, 8, 4); total++;
    pass += test_mamba(32, 64, 32, 8, 4); total++;

    printf("\n%d/%d tests OK\n", pass, total);
    return (pass == total) ? 0 : 1;
}
