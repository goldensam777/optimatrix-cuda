/*
 * conv1d.cu — Convolution 1D depthwise causale
 *
 * Formule :
 *   output[t, d] = sum_{k=0}^{K-1} weight[k, d] * input[t-k, d]
 *   Causale : seuls les indices t-k >= 0 contribuent.
 *
 * Layout memoire (row-major) :
 *   input  : [L * D]     position t, canal d -> input[t*D + d]
 *   weight : [K * D]     noyau k, canal d    -> weight[k*D + d]
 *   bias   : [D]         optionnel (NULL = pas de biais)
 *   output : [L * D]
 *
 * Parallelisme :
 *   1 thread = (t, d). Il boucle sur K en interne.
 *   K est petit (typiquement 2-4), donc la boucle est negligeable.
 *   Total threads : L * D.
 *
 * Comparaison avec CPU (ASM) :
 *   CPU : AVX2 vectorise sur D (8 canaux a la fois), boucle sur K et t
 *   GPU : parallelisme massif sur (t, d), K reste une boucle courte
 *
 * Optimization possible :
 *   Charger weight en shared memory (lu par tous les threads du bloc).
 *   Pour K=4, D=128 : 4*128*4 = 2 Ko — tient largement en shared mem.
 */

#include "optimatrix.h"

#define BLOCK_SIZE 256

/* ── Kernel ──────────────────────────────────────────────────────── */

__global__ void conv1d_kernel(
    const float *input,   /* [L * D] */
    const float *weight,  /* [K * D] */
    const float *bias,    /* [D] ou NULL */
    float *output,        /* [L * D] */
    int L, int D, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * D) return;

    int d = tid % D;
    int t = tid / D;

    float acc = 0.0f;

    /* Boucle sur le noyau — causale : ignore t-k < 0 */
    for (int k = 0; k < K; k++) {
        int t_k = t - k;
        if (t_k >= 0)
            acc += weight[k * D + d] * input[t_k * D + d];
    }

    if (bias != NULL)
        acc += bias[d];

    output[t * D + d] = acc;
}

/* ── Kernel avec weight en shared memory ─────────────────────────── */
/*
 * Pour les petits K et D, on charge tout le noyau en shared memory.
 * Chaque thread du bloc participe au chargement (acces coalescent).
 *
 * Quand utiliser cette version ?
 *   -> D * K <= 4096 (limite confortable de shared mem)
 *   -> Le meme bloc traite plusieurs positions t (economie de charge)
 */

#define MAX_K  8
#define MAX_D  128  /* taille max du noyau en shared mem */

__global__ void conv1d_smem_kernel(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int L, int D, int K)
{
    /* Charger le noyau en shared memory */
    __shared__ float sw[MAX_K * MAX_D];

    int tid_local = threadIdx.x;
    int wsize = K * D;

    /* Chargement cooperatif : tous les threads du bloc */
    for (int i = tid_local; i < wsize && i < MAX_K * MAX_D; i += blockDim.x)
        sw[i] = weight[i];

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + tid_local;
    if (tid >= L * D) return;

    int d = tid % D;
    int t = tid / D;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        int t_k = t - k;
        if (t_k >= 0)
            acc += sw[k * D + d] * input[t_k * D + d];
    }

    if (bias != NULL)
        acc += bias[d];

    output[t * D + d] = acc;
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_conv1d(
    const float *d_input, const float *d_weight,
    const float *d_bias,  float *d_output,
    int L, int D, int K)
{
    int total  = L * D;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Utiliser la version shared memory si le noyau est assez petit */
    if (K <= MAX_K && D <= MAX_D)
        conv1d_smem_kernel<<<blocks, BLOCK_SIZE>>>(
            d_input, d_weight, d_bias, d_output, L, D, K);
    else
        conv1d_kernel<<<blocks, BLOCK_SIZE>>>(
            d_input, d_weight, d_bias, d_output, L, D, K);
}
