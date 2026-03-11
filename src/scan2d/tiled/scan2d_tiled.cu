/*
 * scan2d_tiled.cu — Strategie C : wavefront par tiles
 *
 * Principe :
 *   On decoupe la grille d1 x d2 en tiles de TILE x TILE.
 *   Au lieu de d1+d2-1 kernel launches (une par diag de positions),
 *   on fait ceil(d1/T)+ceil(d2/T)-1 launches (une par diag de TILES).
 *
 *   Exemple : grille 64x64, TILE=8 -> 8x8 tiles -> 15 launches
 *   au lieu de 127. Reduction ~8x de l'overhead de lancement.
 *
 *   Chaque thread prend en charge un (tile, canal_d) :
 *   - il parcourt les positions du tile en wavefront interne
 *   - les predecesseurs hors-tile sont en global memory
 *     (deja calcules grace a l'ordre wavefront des tiles)
 *   - les predecesseurs intra-tile viennent d'etre calcules
 *     par le meme thread (pas de race condition)
 *
 * Mapping des threads :
 *   tid -> (tile_index_sur_diag, d_canal)
 *   Un thread = un (tile, d). Il boucle sur les positions du tile
 *   et sur M en interne.
 *
 *                     tile-diag 0    tile-diag 1    tile-diag 2
 *                     T(0,0)         T(0,1)         T(0,2)
 *                                    T(1,0)         T(1,1)
 *                                                   T(2,0)
 */

#include "optimatrix.h"
#include <math.h>

#define TILE 8

/* ── Kernel ──────────────────────────────────────────────────────── */

__global__ void scan2d_tiled_kernel(
    const float *x,   const float *A1,  const float *A2,
    const float *B,   const float *C,   const float *dt,
    float *y,         float *h,
    int d1, int d2, int D, int M,
    int tile_diag,              /* indice de la diagonale de tiles */
    int num_tile_rows,          /* ceil(d1 / TILE) */
    int num_tile_cols)          /* ceil(d2 / TILE) */
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Decomposer en (tile_index, d_canal) */
    int d        = tid % D;
    int tile_idx = tid / D;

    /* Convertir tile_idx en (tile_row, tile_col) sur cette diag */
    int tr_start = tile_diag - num_tile_cols + 1;
    if (tr_start < 0) tr_start = 0;
    int tr = tr_start + tile_idx;
    int tc = tile_diag - tr;

    /* Hors limites ? */
    if (tr >= num_tile_rows || tc < 0 || tc >= num_tile_cols) return;

    /* Bornes du tile en coordonnees globales */
    int i0 = tr * TILE;
    int j0 = tc * TILE;
    int i1 = i0 + TILE;  if (i1 > d1) i1 = d1;  /* bord bas   */
    int j1 = j0 + TILE;  if (j1 > d2) j1 = d2;  /* bord droit */

    int tile_h = i1 - i0;
    int tile_w = j1 - j0;

    /* Wavefront interne : parcourir les micro-diagonales du tile */
    int num_micro_diags = tile_h + tile_w - 1;

    for (int md = 0; md < num_micro_diags; md++) {
        /* Positions sur la micro-diagonale md : li + lj = md */
        int li_start = md - tile_w + 1;  if (li_start < 0) li_start = 0;
        int li_end   = md;               if (li_end >= tile_h) li_end = tile_h - 1;

        for (int li = li_start; li <= li_end; li++) {
            int lj = md - li;
            int i  = i0 + li;
            int j  = j0 + lj;

            int ij   = i * d2 + j;
            int ij_d = ij * D + d;

            float dt_val = dt[ij_d];
            float y_val  = 0.0f;

            for (int m = 0; m < M; m++) {
                int dm    = d * M + m;
                int ij_dm = ij * D * M + dm;

                float a1 = expf(dt_val * A1[dm]);
                float a2 = expf(dt_val * A2[dm]);

                /* Predecesseurs — peuvent etre hors-tile ou intra-tile,
                 * dans les deux cas ils sont deja en global memory */
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
    }
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_scan2d_tiled(
    const float *d_x,  const float *d_A1, const float *d_A2,
    const float *d_B,  const float *d_C,  const float *d_dt,
    float *d_y,        float *d_h,
    int d1, int d2, int D, int M)
{
    int ntr = (d1 + TILE - 1) / TILE;   /* nombre de tile-rows */
    int ntc = (d2 + TILE - 1) / TILE;   /* nombre de tile-cols */
    int num_tile_diags = ntr + ntc - 1;

    for (int td = 0; td < num_tile_diags; td++) {
        /* Nombre de tiles sur cette diag */
        int tr_start = td - ntc + 1;  if (tr_start < 0) tr_start = 0;
        int tr_end   = td;            if (tr_end >= ntr) tr_end = ntr - 1;
        int tiles_on_diag = tr_end - tr_start + 1;

        int threads = tiles_on_diag * D;
        int blocks  = (threads + 255) / 256;

        scan2d_tiled_kernel<<<blocks, 256>>>(
            d_x, d_A1, d_A2, d_B, d_C, d_dt, d_y, d_h,
            d1, d2, D, M, td, ntr, ntc);
    }

    cudaDeviceSynchronize();
}
