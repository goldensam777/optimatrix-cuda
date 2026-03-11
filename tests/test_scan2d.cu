/*
 * test_scan2d.cu — Test et benchmark des 4 strategies de scan 2D
 *
 * 1. Correctness : compare chaque strategie a une reference CPU
 * 2. Benchmark   : mesure le temps a plusieurs tailles
 * 3. Rapport     : tableau comparatif (vitesse, scalabilite)
 *
 * Strategies :
 *   A  — naive      : 1 kernel/diag, 1 thread = (pos, d), boucle M
 *   A' — naive_vec  : 1 kernel/diag, 1 thread = (pos, d, m), warp reduce
 *   B  — coop       : persistent kernel + grid.sync()
 *   C  — tiled      : wavefront par tuiles 8x8
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "optimatrix.h"

#define TOL 1e-4f

/* ── Declarations des 4 strategies ───────────────────────────────── */

extern "C" void om_scan2d_naive(
    const float *, const float *, const float *,
    const float *, const float *, const float *,
    float *, float *, int, int, int, int);

extern "C" void om_scan2d_naive_vec(
    const float *, const float *, const float *,
    const float *, const float *, const float *,
    float *, float *, int, int, int, int);

extern "C" void om_scan2d_coop(
    const float *, const float *, const float *,
    const float *, const float *, const float *,
    float *, float *, int, int, int, int);

extern "C" void om_scan2d_tiled(
    const float *, const float *, const float *,
    const float *, const float *, const float *,
    float *, float *, int, int, int, int);

/* ── Reference CPU ───────────────────────────────────────────────── */

static void cpu_scan2d(
    const float *x,  const float *A1, const float *A2,
    const float *B,  const float *C,  const float *dt,
    float *y,        float *h,
    int d1, int d2, int D, int M)
{
    int num_diags = d1 + d2 - 1;
    for (int diag = 0; diag < num_diags; diag++) {
        int i_start = (diag - d2 + 1 > 0) ? diag - d2 + 1 : 0;
        int i_end   = (diag < d1 - 1) ? diag : d1 - 1;
        for (int i = i_start; i <= i_end; i++) {
            int j = diag - i;
            int ij = i * d2 + j;
            for (int d = 0; d < D; d++) {
                int ij_d = ij * D + d;
                float dt_val = dt[ij_d];
                float y_val = 0.0f;
                for (int m = 0; m < M; m++) {
                    int dm    = d * M + m;
                    int ij_dm = ij * D * M + dm;
                    float a1 = expf(dt_val * A1[dm]);
                    float a2 = expf(dt_val * A2[dm]);
                    float h_top  = (i > 0) ?
                        h[((i-1)*d2+j)*D*M + dm] : 0.0f;
                    float h_left = (j > 0) ?
                        h[(i*d2+(j-1))*D*M + dm] : 0.0f;
                    float h_val = a1 * h_top + a2 * h_left
                                + dt_val * B[ij_dm] * x[ij_d];
                    h[ij_dm] = h_val;
                    y_val += C[ij_dm] * h_val;
                }
                y[ij_d] = y_val;
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

/* ── Typedef pour les strategies ─────────────────────────────────── */

typedef void (*scan2d_fn)(
    const float *, const float *, const float *,
    const float *, const float *, const float *,
    float *, float *, int, int, int, int);

typedef struct {
    const char *name;
    scan2d_fn   fn;
} Strategy;

/* ── Allocation GPU + transfert ──────────────────────────────────── */

typedef struct {
    float *d_x, *d_A1, *d_A2, *d_B, *d_C, *d_dt, *d_y, *d_h;
    int P, D, M;
} GpuBuffers;

static GpuBuffers alloc_gpu(int d1, int d2, int D, int M,
    const float *h_x,  const float *h_A1, const float *h_A2,
    const float *h_B,  const float *h_C,  const float *h_dt)
{
    GpuBuffers g;
    int P = d1 * d2;
    g.P = P; g.D = D; g.M = M;

    OM_CHECK(cudaMalloc(&g.d_x,  P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&g.d_A1, D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&g.d_A2, D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&g.d_B,  P * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&g.d_C,  P * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&g.d_dt, P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&g.d_y,  P * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&g.d_h,  P * D * M * sizeof(float)));

    OM_CHECK(cudaMemcpy(g.d_x,  h_x,  P*D*sizeof(float),     cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(g.d_A1, h_A1, D*M*sizeof(float),     cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(g.d_A2, h_A2, D*M*sizeof(float),     cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(g.d_B,  h_B,  P*D*M*sizeof(float),   cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(g.d_C,  h_C,  P*D*M*sizeof(float),   cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(g.d_dt, h_dt, P*D*sizeof(float),     cudaMemcpyHostToDevice));

    return g;
}

static void reset_outputs(GpuBuffers *g)
{
    OM_CHECK(cudaMemset(g->d_y, 0, g->P * g->D * sizeof(float)));
    OM_CHECK(cudaMemset(g->d_h, 0, g->P * g->D * g->M * sizeof(float)));
}

static void free_gpu(GpuBuffers *g)
{
    cudaFree(g->d_x);  cudaFree(g->d_A1); cudaFree(g->d_A2);
    cudaFree(g->d_B);  cudaFree(g->d_C);  cudaFree(g->d_dt);
    cudaFree(g->d_y);  cudaFree(g->d_h);
}

/* ── Test de correctness ─────────────────────────────────────────── */

static int test_correctness(Strategy *strats, int num_strats,
                            int d1, int d2, int D, int M)
{
    int P = d1 * d2;

    float *h_x  = (float *)malloc(P * D * sizeof(float));
    float *h_A1 = (float *)malloc(D * M * sizeof(float));
    float *h_A2 = (float *)malloc(D * M * sizeof(float));
    float *h_B  = (float *)malloc(P * D * M * sizeof(float));
    float *h_C  = (float *)malloc(P * D * M * sizeof(float));
    float *h_dt = (float *)malloc(P * D * sizeof(float));
    float *h_y_ref = (float *)calloc(P * D, sizeof(float));
    float *h_h_ref = (float *)calloc(P * D * M, sizeof(float));
    float *h_y_gpu = (float *)malloc(P * D * sizeof(float));

    srand(42);
    fill_random(h_x,  P * D,         -1.0f, 1.0f);
    /* A negatif avec |A| suffisant : garantit a1 + a2 < 1 (stabilite 2D)
     * exp(dt_max * A_max) = exp(0.5 * -1.4) ≈ 0.497 < 0.5 → OK */
    fill_random(h_A1, D * M,         -3.0f, -1.4f);
    fill_random(h_A2, D * M,         -3.0f, -1.4f);
    fill_random(h_B,  P * D * M,     -0.5f,  0.5f);
    fill_random(h_C,  P * D * M,     -0.5f,  0.5f);
    fill_random(h_dt, P * D,          0.01f,  0.5f);

    cpu_scan2d(h_x, h_A1, h_A2, h_B, h_C, h_dt,
               h_y_ref, h_h_ref, d1, d2, D, M);

    GpuBuffers g = alloc_gpu(d1, d2, D, M,
                             h_x, h_A1, h_A2, h_B, h_C, h_dt);

    int pass = 0;
    for (int s = 0; s < num_strats; s++) {
        reset_outputs(&g);
        strats[s].fn(g.d_x, g.d_A1, g.d_A2,
                     g.d_B, g.d_C, g.d_dt,
                     g.d_y, g.d_h, d1, d2, D, M);
        OM_CHECK(cudaMemcpy(h_y_gpu, g.d_y, P * D * sizeof(float),
                            cudaMemcpyDeviceToHost));

        float err = max_error(h_y_gpu, h_y_ref, P * D);
        int ok = err < TOL;
        printf("  %-12s %dx%d D=%d M=%d : max_err=%.2e  %s\n",
               strats[s].name, d1, d2, D, M, err, ok ? "OK" : "FAIL");
        pass += ok;
    }

    free_gpu(&g);
    free(h_x); free(h_A1); free(h_A2);
    free(h_B); free(h_C);  free(h_dt);
    free(h_y_ref); free(h_h_ref); free(h_y_gpu);

    return pass;
}

/* ── Benchmark ───────────────────────────────────────────────────── */

static float benchmark(scan2d_fn fn, GpuBuffers *g,
                       int d1, int d2, int D, int M, int iters)
{
    /* Warmup */
    reset_outputs(g);
    fn(g->d_x, g->d_A1, g->d_A2,
       g->d_B, g->d_C, g->d_dt,
       g->d_y, g->d_h, d1, d2, D, M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        reset_outputs(g);
        fn(g->d_x, g->d_A1, g->d_A2,
           g->d_B, g->d_C, g->d_dt,
           g->d_y, g->d_h, d1, d2, D, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void)
{
    Strategy strats[] = {
        {"naive",      om_scan2d_naive},
        {"naive_vec",  om_scan2d_naive_vec},
        {"coop",       om_scan2d_coop},
        {"tiled",      om_scan2d_tiled},
    };
    int num_strats = 4;

    /* ════════════════════════════════════════════════════════════════
     * PARTIE 1 : CORRECTNESS
     * ════════════════════════════════════════════════════════════════ */
    printf("══════════════════════════════════════════════════════\n");
    printf("  CORRECTNESS — scan2d (4 strategies) vs ref CPU\n");
    printf("══════════════════════════════════════════════════════\n\n");

    int pass = 0, total = 0;

    /* Petit */
    int n = test_correctness(strats, num_strats, 4, 4, 8, 4);
    pass += n; total += num_strats;

    /* Moyen */
    n = test_correctness(strats, num_strats, 8, 8, 16, 8);
    pass += n; total += num_strats;

    /* Non-carre */
    n = test_correctness(strats, num_strats, 6, 10, 8, 4);
    pass += n; total += num_strats;

    /* Plus grand */
    n = test_correctness(strats, num_strats, 16, 16, 32, 8);
    pass += n; total += num_strats;

    printf("\nCorrectness : %d/%d OK\n\n", pass, total);

    /* ════════════════════════════════════════════════════════════════
     * PARTIE 2 : BENCHMARK
     * ════════════════════════════════════════════════════════════════ */
    printf("══════════════════════════════════════════════════════\n");
    printf("  BENCHMARK — temps moyen (ms)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    typedef struct { int d1, d2, D, M, iters; } BenchConfig;
    BenchConfig configs[] = {
        { 8,   8,  16,  8,  100},
        {16,  16,  32,  8,   50},
        {32,  32,  64,  8,   20},
        {64,  64, 128, 16,   10},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("  %-14s", "Config");
    for (int s = 0; s < num_strats; s++)
        printf("  %12s", strats[s].name);
    printf("  %10s\n", "best");
    printf("  ──────────────────────────────────────────────────────"
           "──────────────────\n");

    for (int c = 0; c < num_configs; c++) {
        BenchConfig *cfg = &configs[c];
        int P = cfg->d1 * cfg->d2;

        float *h_x  = (float *)malloc(P * cfg->D * sizeof(float));
        float *h_A1 = (float *)malloc(cfg->D * cfg->M * sizeof(float));
        float *h_A2 = (float *)malloc(cfg->D * cfg->M * sizeof(float));
        float *h_B  = (float *)malloc(P * cfg->D * cfg->M * sizeof(float));
        float *h_C  = (float *)malloc(P * cfg->D * cfg->M * sizeof(float));
        float *h_dt = (float *)malloc(P * cfg->D * sizeof(float));

        srand(42);
        fill_random(h_x,  P * cfg->D,              -1.0f, 1.0f);
        fill_random(h_A1, cfg->D * cfg->M,         -3.0f, -1.4f);
        fill_random(h_A2, cfg->D * cfg->M,         -3.0f, -1.4f);
        fill_random(h_B,  P * cfg->D * cfg->M,     -0.5f, 0.5f);
        fill_random(h_C,  P * cfg->D * cfg->M,     -0.5f, 0.5f);
        fill_random(h_dt, P * cfg->D,               0.01f, 0.5f);

        GpuBuffers g = alloc_gpu(cfg->d1, cfg->d2, cfg->D, cfg->M,
                                 h_x, h_A1, h_A2, h_B, h_C, h_dt);

        char label[32];
        snprintf(label, sizeof(label), "%dx%d D%d M%d",
                 cfg->d1, cfg->d2, cfg->D, cfg->M);
        printf("  %-14s", label);

        float times[4];
        float best_time = 1e30f;
        int best_idx = 0;

        for (int s = 0; s < num_strats; s++) {
            times[s] = benchmark(strats[s].fn, &g,
                                 cfg->d1, cfg->d2, cfg->D, cfg->M,
                                 cfg->iters);
            printf("  %10.3f ms", times[s]);
            if (times[s] < best_time) {
                best_time = times[s];
                best_idx = s;
            }
        }

        printf("  <- %s", strats[best_idx].name);
        printf("\n");

        free_gpu(&g);
        free(h_x); free(h_A1); free(h_A2);
        free(h_B); free(h_C);  free(h_dt);
    }

    printf("\n");
    return (pass == total) ? 0 : 1;
}
