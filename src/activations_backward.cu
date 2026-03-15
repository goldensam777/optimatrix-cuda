/*
 * activations_backward.cu — Derivees des activations non-lineaires
 *
 * SiLU(x)     = x * sigma(x)
 * SiLU'(x)    = sigma(x) * (1 + x * (1 - sigma(x)))
 *             = sigma(x) + x * sigma(x) * sigma(-x)
 *
 * Softplus(x) = ln(1 + exp(x))
 * Softplus'(x)= sigma(x) = 1 / (1 + exp(-x))
 *
 * Convention : d_x est l'entree du forward (sauvegardee).
 *   d_dx[i] = d_dy[i] * f'(d_x[i])
 */

#include "optimatrix.h"
#include <math.h>

/* ── SiLU backward ───────────────────────────────────────────────── */
/*
 * sigma(x)  = 1 / (1 + exp(-x))
 * SiLU'(x)  = sigma(x) * (1 + x * (1 - sigma(x)))
 *
 * Note : on recompute sigma depuis x pour eviter de stocker un buffer
 * supplementaire au forward. Cout : 1 expf par element (meme que forward).
 */

__global__ void silu_backward_kernel(const float *x, const float *dy,
                                     float *dx, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float xv  = x[tid];
    float sig = 1.0f / (1.0f + expf(-xv));
    float dsilu = sig * (1.0f + xv * (1.0f - sig));
    dx[tid] = dy[tid] * dsilu;
}

/* ── Softplus backward ───────────────────────────────────────────── */
/*
 * Softplus'(x) = sigma(x)
 */

__global__ void softplus_backward_kernel(const float *x, const float *dy,
                                         float *dx, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float sig = 1.0f / (1.0f + expf(-x[tid]));
    dx[tid] = dy[tid] * sig;
}

/* ── API ─────────────────────────────────────────────────────────── */

static inline int grid(int n) { return (n + 255) / 256; }

void om_silu_backward(const float *d_x, const float *d_dy, float *d_dx, int n)
{
    silu_backward_kernel<<<grid(n), 256>>>(d_x, d_dy, d_dx, n);
}

void om_softplus_backward(const float *d_x, const float *d_dy, float *d_dx, int n)
{
    softplus_backward_kernel<<<grid(n), 256>>>(d_x, d_dy, d_dx, n);
}
