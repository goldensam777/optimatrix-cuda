/*
 * scan2d_naive_vec.cu — Strategie A' : naive VECTORISEE sur M
 *
 * Difference avec naive :
 *   naive     : 1 thread = (pos, d),     boucle sur M
 *   naive_vec : 1 thread = (pos, d, m),  reduction warp pour y
 *
 * Pourquoi c'est impossible sur CPU :
 *   - AVX2 n'a pas d'instruction exp vectorielle native
 *   - On ne peut pas facilement reduire sur M avec SIMD
 *   - Le warp shuffle n'existe pas sur CPU
 *
 * Pourquoi c'est possible sur GPU :
 *   - expf() est calcule par les SFU (Special Function Units)
 *     en ~20 cycles, independamment par thread
 *   - Le warp shuffle (__shfl_down_sync) permet une reduction
 *     en O(log M) sans passer par la shared memory
 *   - On multiplie le parallelisme par M (typiquement x8 ou x16)
 *
 * Reduction warp pour y :
 *   y(i,j,d) = sum_m C(i,j,d,m) * h(i,j,d,m)
 *
 *   Les M threads d'un meme (pos, d) forment un "groupe".
 *   Chaque thread calcule sa contribution C*h, puis on reduit :
 *
 *     Etape 1 : thread m=0 recoit de m=8  (si M=16)
 *     Etape 2 : thread m=0 recoit de m=4
 *     Etape 3 : thread m=0 recoit de m=2
 *     Etape 4 : thread m=0 recoit de m=1
 *     -> thread m=0 a la somme totale, ecrit y
 *
 *   Cout : log2(M) = 4 cycles au lieu de M = 16 iterations.
 */

#include "optimatrix.h"
#include <math.h>

/* ── Helpers ─────────────────────────────────────────────────────── */

static inline __host__ int diag_i_start(int d, int d2)
{
    return (d - d2 + 1 > 0) ? d - d2 + 1 : 0;
}

static inline __host__ int diag_len(int d, int d1, int d2)
{
    int is = (d - d2 + 1 > 0) ? d - d2 + 1 : 0;
    int ie = (d < d1 - 1) ? d : d1 - 1;
    return ie - is + 1;
}

/* ── Arrondir M a la puissance de 2 superieure ───────────────────── */
/*    Necessaire pour que le warp shuffle fonctionne correctement.     */
/*    M=8 -> 8, M=12 -> 16, M=16 -> 16                               */

static inline __host__ __device__ int next_pow2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/* ── Kernel vectorise ────────────────────────────────────────────── */
/*
 * Mapping des threads :
 *   tid -> (pos, d, m)
 *   total = dlen * D * M_pad   (M_pad = next_pow2(M))
 *
 * Chaque thread :
 *   1. Calcule exp(dt*A1) et exp(dt*A2)          -- SFU, ~20 cycles
 *   2. Lit h_top et h_left depuis global memory
 *   3. Calcule h = a1*h_top + a2*h_left + dt*B*x -- 1 FMA
 *   4. Ecrit h en global memory
 *   5. Calcule partial = C * h
 *   6. Reduction warp sur M -> y
 */

__global__ void scan2d_naive_vec_kernel(
    const float *x,   const float *A1,  const float *A2,
    const float *B,   const float *C,   const float *dt,
    float *y,         float *h,
    int d1, int d2, int D, int M, int M_pad,
    int diag, int i_start, int dlen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dlen * D * M_pad;
    if (tid >= total) return;

    /* Decomposition : tid = pos * (D * M_pad) + d * M_pad + m */
    int m   = tid % M_pad;
    int d   = (tid / M_pad) % D;
    int pos = tid / (D * M_pad);

    int i = i_start + pos;
    int j = diag - i;

    /* Thread fantome : m >= M, participe a la reduction mais val = 0 */
    float partial = 0.0f;

    if (m < M) {
        int ij    = i * d2 + j;
        int ij_d  = ij * D + d;
        int dm    = d * M + m;
        int ij_dm = ij * D * M + dm;

        float dt_val = dt[ij_d];

        /* SFU : chaque thread calcule son propre exp() en parallele */
        float a1 = expf(dt_val * A1[dm]);
        float a2 = expf(dt_val * A2[dm]);

        float h_top  = (i > 0) ?
            h[((i - 1) * d2 + j) * D * M + dm] : 0.0f;
        float h_left = (j > 0) ?
            h[(i * d2 + (j - 1)) * D * M + dm] : 0.0f;

        float h_val = a1 * h_top + a2 * h_left
                    + dt_val * B[ij_dm] * x[ij_d];
        h[ij_dm] = h_val;

        partial = C[ij_dm] * h_val;
    }

    /* ── Reduction warp shuffle sur M ────────────────────────────── */
    /*                                                                 */
    /*  Les M_pad threads d'un meme (pos, d) sont contigus en memoire. */
    /*  On utilise __shfl_down_sync pour sommer sans shared memory.    */
    /*                                                                 */
    /*  __shfl_down_sync(mask, val, offset) :                          */
    /*    Le thread t recoit la valeur du thread t+offset              */
    /*    dans le meme warp (32 threads).                              */
    /*                                                                 */
    /*  Pour M_pad=16, on fait 4 etapes (log2(16)) :                  */
    /*    offset=8 : t0 += t8,  t1 += t9,  ...                        */
    /*    offset=4 : t0 += t4,  t1 += t5,  ...                        */
    /*    offset=2 : t0 += t2,  t1 += t3,  ...                        */
    /*    offset=1 : t0 += t1                                          */
    /*    -> t0 contient la somme totale                               */

    unsigned mask = 0xFFFFFFFF;
    for (int offset = M_pad / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(mask, partial, offset);

    /* Seul le thread m=0 ecrit y */
    if (m == 0) {
        int ij   = i * d2 + j;
        int ij_d = ij * D + d;
        y[ij_d]  = partial;
    }
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_scan2d_naive_vec(
    const float *d_x,  const float *d_A1, const float *d_A2,
    const float *d_B,  const float *d_C,  const float *d_dt,
    float *d_y,        float *d_h,
    int d1, int d2, int D, int M)
{
    int M_pad = next_pow2(M);
    int num_diags = d1 + d2 - 1;

    for (int diag = 0; diag < num_diags; diag++) {
        int is   = diag_i_start(diag, d2);
        int dlen = diag_len(diag, d1, d2);

        int threads = dlen * D * M_pad;
        int blocks  = (threads + 255) / 256;

        scan2d_naive_vec_kernel<<<blocks, 256>>>(
            d_x, d_A1, d_A2, d_B, d_C, d_dt, d_y, d_h,
            d1, d2, D, M, M_pad, diag, is, dlen);
    }

    cudaDeviceSynchronize();
}
