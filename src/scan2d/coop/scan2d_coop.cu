/*
 * scan2d_coop.cu — Strategie B : persistent kernel + cooperative groups
 *
 * Principe :
 *   Au lieu de lancer un kernel par diagonale (overhead ~5-10us chacun),
 *   on lance UN SEUL kernel qui vit pendant tout le scan.
 *   Entre chaque diagonale, on fait un grid.sync() — synchronisation
 *   globale de tous les blocs du GPU.
 *
 * Avantage :
 *   Zero overhead de lancement entre diagonales.
 *   Le kernel reste "chaud" sur le GPU.
 *
 * Contrainte :
 *   Le nombre de blocs ne doit pas depasser ce que le GPU peut
 *   executer simultanement (sinon deadlock sur grid.sync).
 *   On utilise cudaOccupancyMaxActiveBlocksPerMultiprocessor pour
 *   calculer cette limite.
 *
 * Lancement :
 *   cudaLaunchCooperativeKernel() au lieu de <<<blocks, threads>>>
 *   Necessite compute capability >= 6.0
 */

#include "optimatrix.h"
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

/* ── Kernel persistent ───────────────────────────────────────────── */
/*
 * Mapping des threads :
 *   tid = blockIdx.x * blockDim.x + threadIdx.x
 *   On mappe tid -> (pos_sur_diag, d_canal)
 *   total_threads = max_diag_len * D
 *
 *   A chaque diagonale, seuls les threads avec tid < diag_len * D
 *   ont du travail. Les autres attendent au grid.sync().
 */

__global__ void scan2d_coop_kernel(
    const float *x,   const float *A1,  const float *A2,
    const float *B,   const float *C,   const float *dt,
    float *y,         float *h,
    int d1, int d2, int D, int M)
{
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int num_diags = d1 + d2 - 1;

    for (int diag = 0; diag < num_diags; diag++) {
        /* Bornes de la diagonale */
        int i_start = (diag - d2 + 1 > 0) ? diag - d2 + 1 : 0;
        int i_end   = (diag < d1 - 1) ? diag : d1 - 1;
        int dlen    = i_end - i_start + 1;

        if (tid < dlen * D) {
            int d   = tid % D;
            int pos = tid / D;
            int i   = i_start + pos;
            int j   = diag - i;

            int ij   = i * d2 + j;
            int ij_d = ij * D + d;

            float dt_val = dt[ij_d];
            float y_val  = 0.0f;

            for (int m = 0; m < M; m++) {
                int dm    = d * M + m;
                int ij_dm = ij * D * M + dm;

                float a1 = expf(dt_val * A1[dm]);
                float a2 = expf(dt_val * A2[dm]);

                float h_top  = (i > 0) ?
                    h[((i - 1) * d2 + j) * D * M + dm] : 0.0f;
                float h_left = (j > 0) ?
                    h[(i * d2 + (j - 1)) * D * M + dm] : 0.0f;

                float h_val = a1 * h_top + a2 * h_left
                            + dt_val * B[ij_dm] * x[ij_d];
                h[ij_dm] = h_val;

                y_val += C[ij_dm] * h_val;
            }

            y[ij_d] = y_val;
        }

        /* Synchronisation globale — TOUS les blocs attendent ici */
        grid.sync();
    }
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_scan2d_coop(
    const float *d_x,  const float *d_A1, const float *d_A2,
    const float *d_B,  const float *d_C,  const float *d_dt,
    float *d_y,        float *d_h,
    int d1, int d2, int D, int M)
{
    int block_size = 256;

    /* Combien de threads au max sur une diagonale ? */
    int max_dlen = (d1 < d2) ? d1 : d2;
    int total    = max_dlen * D;
    int num_blocks = (total + block_size - 1) / block_size;

    /* Limiter au nombre de blocs residants sur le GPU */
    int dev;
    cudaGetDevice(&dev);

    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, scan2d_coop_kernel, block_size, 0);

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_blocks = max_blocks_per_sm * num_sms;
    if (num_blocks > max_blocks)
        num_blocks = max_blocks;

    /* Lancement cooperatif */
    void *args[] = {
        (void *)&d_x,  (void *)&d_A1, (void *)&d_A2,
        (void *)&d_B,  (void *)&d_C,  (void *)&d_dt,
        (void *)&d_y,  (void *)&d_h,
        (void *)&d1,   (void *)&d2,   (void *)&D, (void *)&M
    };

    cudaLaunchCooperativeKernel(
        (void *)scan2d_coop_kernel,
        dim3(num_blocks), dim3(block_size),
        args);

    cudaDeviceSynchronize();
}
