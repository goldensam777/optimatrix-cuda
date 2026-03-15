/*
 * conv1d_backward.cu — Backward de la convolution 1D depthwise causale
 *
 * Forward : output[t, d] = sum_{k=0}^{K-1} weight[k, d] * input[t-k, d]
 *           (causale : ignore t-k < 0)
 *
 * Backward (donnee dy[L,D] = dL/d output) :
 *
 *   dL/d input[t, d]    = sum_{k=0}^{K-1} weight[k, d] * dy[t+k, d]
 *                          (ou t+k < L)
 *
 *   dL/d weight[k, d]   = sum_{t=k}^{L-1} dy[t, d] * input[t-k, d]
 *
 *   dL/d bias[d]        = sum_{t=0}^{L-1} dy[t, d]
 *
 * Parallelisme :
 *   dinput  : 1 thread = (t, d), boucle sur K en avant
 *   dweight : 1 thread = (k, d), boucle sur t
 *   dbias   : 1 thread = d, boucle sur t
 *
 * Symetrie avec le forward (conv1d.cu) :
 *   Le forward fait output[t] = sum_{k} w[k] * input[t-k]
 *   Le backward dinput[t] = sum_{k} w[k] * dy[t+k]
 *   -> meme noyau, direction opposee dans le temps.
 */

#include "optimatrix.h"

#define BLOCK_SIZE 256

/* ── Kernel dinput ───────────────────────────────────────────────── */
/*
 * 1 thread = (t, d).
 * dinput[t, d] = sum_{k=0}^{K-1} weight[k, d] * dy[t+k, d]
 * (t+k doit etre < L pour rester causal)
 */

__global__ void conv1d_backward_dinput_kernel(
    const float *dy,      /* [L x D] */
    const float *weight,  /* [K x D] */
    float *dinput,        /* [L x D] */
    int L, int D, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * D) return;

    int d = tid % D;
    int t = tid / D;

    float grad = 0.0f;
    for (int k = 0; k < K; k++) {
        int t_k = t + k;
        if (t_k < L)
            grad += weight[k * D + d] * dy[t_k * D + d];
    }
    dinput[t * D + d] = grad;
}

/* ── Kernel dweight ──────────────────────────────────────────────── */
/*
 * 1 thread = (k, d).
 * dweight[k, d] = sum_{t=k}^{L-1} dy[t, d] * input[t-k, d]
 */

__global__ void conv1d_backward_dweight_kernel(
    const float *dy,      /* [L x D] */
    const float *input,   /* [L x D] */
    float *dweight,       /* [K x D] */
    int L, int D, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= K * D) return;

    int d = tid % D;
    int k = tid / D;

    float grad = 0.0f;
    for (int t = k; t < L; t++)
        grad += dy[t * D + d] * input[(t - k) * D + d];

    dweight[k * D + d] = grad;
}

/* ── Kernel dbias ────────────────────────────────────────────────── */
/*
 * 1 thread = d.
 * dbias[d] = sum_{t} dy[t, d]
 */

__global__ void conv1d_backward_dbias_kernel(
    const float *dy,   /* [L x D] */
    float *dbias,      /* [D] */
    int L, int D)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    float grad = 0.0f;
    for (int t = 0; t < L; t++)
        grad += dy[t * D + d];

    dbias[d] = grad;
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_conv1d_backward(
    const float *d_dy,
    const float *d_input, const float *d_weight,
    float *d_dinput, float *d_dweight, float *d_dbias,
    int L, int D, int K)
{
    /* dinput */
    int blocks_LD = (L * D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conv1d_backward_dinput_kernel<<<blocks_LD, BLOCK_SIZE>>>(
        d_dy, d_weight, d_dinput, L, D, K);

    /* dweight */
    int blocks_KD = (K * D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conv1d_backward_dweight_kernel<<<blocks_KD, BLOCK_SIZE>>>(
        d_dy, d_input, d_dweight, L, D, K);

    /* dbias (optionnel) */
    if (d_dbias != NULL) {
        int blocks_D = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
        conv1d_backward_dbias_kernel<<<blocks_D, BLOCK_SIZE>>>(
            d_dy, d_dbias, L, D);
    }
}
