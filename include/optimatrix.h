#ifndef OPTIMATRIX_H
#define OPTIMATRIX_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Erreur handling ─────────────────────────────────────────────── */

#define OM_CHECK(call) do {                                         \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                    \
    }                                                               \
} while(0)

/* ── Activations ─────────────────────────────────────────────────── */

void om_relu(const float *d_in, float *d_out, int n);
void om_sigmoid(const float *d_in, float *d_out, int n);
void om_silu(const float *d_in, float *d_out, int n);
void om_softplus(const float *d_in, float *d_out, int n);

/* ── Hadamard (element-wise product) ─────────────────────────────── */

void om_hadamard(const float *d_a, const float *d_b, float *d_out, int n);

/* ── GEMV : y = A * x  (A : m x n, x : n, y : m) ───────────────── */

void om_gemv(const float *d_A, const float *d_x, float *d_y, int m, int n);

/* ── GEMM : C = A * B  (A : m x k, B : k x n, C : m x n) ───────── */

void om_gemm(const float *d_A, const float *d_B, float *d_C,
             int m, int n, int k);

/* ── Conv1D depthwise causale ────────────────────────────────────── */
/*    input  : L x D                                                  */
/*    kernel : K x D                                                  */
/*    bias   : D (peut etre NULL)                                     */
/*    output : L x D                                                  */

void om_conv1d(const float *d_input, const float *d_kernel,
               const float *d_bias, float *d_output,
               int L, int D, int K);

/* ── Selective Scan 1D ───────────────────────────────────────────── */
/*    Recurrence : h_t = exp(dt*A) * h_{t-1} + dt * B_t * x_t        */
/*                 y_t = C_t * h_t                                    */

void om_scan1d_forward(const float *d_x, const float *d_A,
                       const float *d_B, const float *d_C,
                       const float *d_dt, float *d_y, float *d_h,
                       int L, int D, int M);

/* ── Selective Scan 2D wavefront — 3 strategies ──────────────────── */
/*                                                                     */
/*  Signature commune :                                                */
/*    x  : [d1 * d2 * D]       entree                                 */
/*    A1 : [D * M]              transition dim 1 (haut)                */
/*    A2 : [D * M]              transition dim 2 (gauche)              */
/*    B  : [d1 * d2 * D * M]   selectif                               */
/*    C  : [d1 * d2 * D * M]   selectif                               */
/*    dt : [d1 * d2 * D]       pas de temps                           */
/*    y  : [d1 * d2 * D]       sortie                                 */
/*    h  : [d1 * d2 * D * M]   etats caches                           */

/* A — Naive : un kernel launch par diagonale                          */
void om_scan2d_naive(const float *d_x, const float *d_A1, const float *d_A2,
                     const float *d_B, const float *d_C, const float *d_dt,
                     float *d_y, float *d_h,
                     int d1, int d2, int D, int M);

/* B — Cooperative : persistent kernel + grid.sync()                   */
void om_scan2d_coop(const float *d_x, const float *d_A1, const float *d_A2,
                    const float *d_B, const float *d_C, const float *d_dt,
                    float *d_y, float *d_h,
                    int d1, int d2, int D, int M);

/* C — Tiled : wavefront par tuiles de TILE x TILE                    */
void om_scan2d_tiled(const float *d_x, const float *d_A1, const float *d_A2,
                     const float *d_B, const float *d_C, const float *d_dt,
                     float *d_y, float *d_h,
                     int d1, int d2, int D, int M);

/* A' — Naive vectorisee : 1 thread par (pos, d, m) + warp reduction  */
/*      Impossible sur CPU : pas de exp vectoriel ni de warp shuffle   */
void om_scan2d_naive_vec(const float *d_x, const float *d_A1, const float *d_A2,
                         const float *d_B, const float *d_C, const float *d_dt,
                         float *d_y, float *d_h,
                         int d1, int d2, int D, int M);

#ifdef __cplusplus
}
#endif

#endif /* OPTIMATRIX_H */
