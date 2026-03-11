/*
 * scan2d_naive.cu — Strategie A : un kernel launch par diagonale
 *
 * Principe :
 *   La recurrence 2D impose un DAG : h(i,j) depend de h(i-1,j) et h(i,j-1).
 *   Les positions sur une meme anti-diagonale (i+j = d) sont independantes.
 *
 *   On lance un kernel par diagonale d = 0, 1, ..., d1+d2-2.
 *   Chaque thread traite une paire (position_sur_diag, canal_d).
 *   Le thread boucle sur M (dimension de l'etat cache) en interne.
 *
 * Layout memoire (row-major) :
 *   x  : [d1 * d2 * D]         position (i,j) canal d  -> x[(i*d2+j)*D + d]
 *   A1 : [D * M]               par dimension 1 (haut)
 *   A2 : [D * M]               par dimension 2 (gauche)
 *   B  : [d1 * d2 * D * M]     selectif (depend de l'entree)
 *   C  : [d1 * d2 * D * M]     selectif
 *   dt : [d1 * d2 * D]         pas de temps adaptatif
 *   h  : [d1 * d2 * D * M]     etats caches (sortie, stockes pour backward)
 *   y  : [d1 * d2 * D]         sortie
 *
 * Recurrence :
 *   h(i,j,d,m) = exp(dt*A1[d,m]) * h(i-1,j,d,m)
 *              + exp(dt*A2[d,m]) * h(i,j-1,d,m)
 *              + dt * B(i,j,d,m) * x(i,j,d)
 *   y(i,j,d)   = sum_m C(i,j,d,m) * h(i,j,d,m)
 */

#include "optimatrix.h"
#include <math.h>

/* ── Helpers ─────────────────────────────────────────────────────── */

/* Nombre de positions sur la diagonale d d'une grille d1 x d2 */
static inline __host__ int diag_len(int d, int d1, int d2)
{
    int i_start = (d - d2 + 1 > 0) ? d - d2 + 1 : 0;
    int i_end   = (d < d1 - 1) ? d : d1 - 1;
    return i_end - i_start + 1;
}

static inline __host__ int diag_i_start(int d, int d2)
{
    return (d - d2 + 1 > 0) ? d - d2 + 1 : 0;
}

/* ── Kernel ──────────────────────────────────────────────────────── */
/*
 * Un thread = une paire (position, d_canal).
 * Il boucle sur M pour calculer h et y.
 *
 * Pourquoi boucler sur M plutot qu'un thread par (pos, d, m) ?
 *   -> Evite atomicAdd sur y (plusieurs m ecrivent au meme y[i,j,d])
 *   -> M est petit (8-16), la boucle est rapide
 */

__global__ void scan2d_diag_kernel(
    const float *x,   const float *A1,  const float *A2,
    const float *B,   const float *C,   const float *dt,
    float *y,         float *h,
    int d1, int d2, int D, int M,
    int diag, int i_start, int dlen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dlen * D) return;

    int d   = tid % D;
    int pos = tid / D;

    int i = i_start + pos;
    int j = diag - i;

    /* Indices lineaires */
    int ij   = i * d2 + j;
    int ij_d = ij * D + d;

    float dt_val = dt[ij_d];
    float y_val  = 0.0f;

    for (int m = 0; m < M; m++) {
        int dm    = d * M + m;
        int ij_dm = ij * D * M + dm;

        /* Exponentielles des transitions */
        float a1 = expf(dt_val * A1[dm]);
        float a2 = expf(dt_val * A2[dm]);

        /* Predecesseurs : haut (i-1,j) et gauche (i,j-1) */
        float h_top  = (i > 0) ? h[((i - 1) * d2 + j) * D * M + dm] : 0.0f;
        float h_left = (j > 0) ? h[(i * d2 + (j - 1)) * D * M + dm] : 0.0f;

        /* Recurrence */
        float h_val = a1 * h_top + a2 * h_left + dt_val * B[ij_dm] * x[ij_d];
        h[ij_dm] = h_val;

        /* Accumulation sortie */
        y_val += C[ij_dm] * h_val;
    }

    y[ij_d] = y_val;
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_scan2d_naive(
    const float *d_x,  const float *d_A1, const float *d_A2,
    const float *d_B,  const float *d_C,  const float *d_dt,
    float *d_y,        float *d_h,
    int d1, int d2, int D, int M)
{
    int num_diags = d1 + d2 - 1;

    for (int diag = 0; diag < num_diags; diag++) {
        int is   = diag_i_start(diag, d2);
        int dlen = diag_len(diag, d1, d2);

        int threads = dlen * D;
        int blocks  = (threads + 255) / 256;

        scan2d_diag_kernel<<<blocks, 256>>>(
            d_x, d_A1, d_A2, d_B, d_C, d_dt, d_y, d_h,
            d1, d2, D, M, diag, is, dlen);
    }

    /* Attendre que tous les kernels finissent */
    cudaDeviceSynchronize();
}
