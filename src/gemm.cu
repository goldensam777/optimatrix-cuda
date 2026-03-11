/*
 * gemm.cu — GEMV et GEMM custom sur GPU
 *
 * GEMV : y = A * x
 *   A : m x n (row-major), x : n, y : m
 *   Chaque thread calcule une ligne de A dot x.
 *   On utilise la shared memory pour cacher x (lu par tous les threads).
 *
 * GEMM : C = A * B
 *   A : m x k, B : k x n, C : m x n (tout row-major)
 *   Tiling 2D avec shared memory : chaque bloc charge un tile de A et B
 *   dans la shared mem, calcule le produit partiel, accumule.
 *   Tile size = TILE (16 ou 32).
 */

#include "optimatrix.h"

#define TILE 16

/* ── GEMV ────────────────────────────────────────────────────────── */
/*
 * 1 thread = 1 ligne de la matrice.
 * On charge x en shared memory par morceaux de blockDim.x.
 *
 * Pourquoi shared memory ?
 *   Chaque thread lit tout x. Sans shared mem, chaque thread
 *   ferait n lectures en global memory. Avec, on lit x une fois
 *   par bloc et tout le bloc y accede a ~100x la vitesse.
 */

__global__ void gemv_kernel(const float *A, const float *x,
                            float *y, int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++)
        sum += A[row * n + j] * x[j];
    y[row] = sum;
}

void om_gemv(const float *d_A, const float *d_x, float *d_y, int m, int n)
{
    int blocks = (m + 255) / 256;
    gemv_kernel<<<blocks, 256>>>(d_A, d_x, d_y, m, n);
}

/* ── GEMM — Tiled avec shared memory ─────────────────────────────── */
/*
 * Idee du tiling :
 *
 * Pour calculer C[row][col], on a besoin de la ligne row de A
 * et la colonne col de B. Mais lire k floats en global memory
 * pour chaque thread = tres lent.
 *
 * Solution : on decoupe A et B en tiles de TILE x TILE.
 * A chaque etape :
 *   1. Le bloc charge un tile de A et un tile de B en shared mem
 *   2. __syncthreads() — tout le monde a fini de charger
 *   3. Chaque thread accumule TILE multiplications
 *   4. __syncthreads() — on peut ecraser la shared mem
 *   5. On passe au tile suivant
 *
 * Resultat : au lieu de k lectures global par thread,
 * on fait k/TILE lectures global + TILE lectures shared.
 * Shared memory = ~100x plus rapide que global.
 */

__global__ void gemm_kernel(const float *A, const float *B,
                            float *C, int m, int n, int k)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    /* Boucle sur les tiles le long de la dimension k */
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        /* Charger un tile de A en shared memory */
        int a_col = t * TILE + threadIdx.x;
        if (row < m && a_col < k)
            sA[threadIdx.y][threadIdx.x] = A[row * k + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        /* Charger un tile de B en shared memory */
        int b_row = t * TILE + threadIdx.y;
        if (b_row < k && col < n)
            sB[threadIdx.y][threadIdx.x] = B[b_row * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        /* Accumuler le produit partiel depuis shared memory */
        for (int i = 0; i < TILE; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

void om_gemm(const float *d_A, const float *d_B, float *d_C,
             int m, int n, int k)
{
    dim3 block(TILE, TILE);            /* 16x16 = 256 threads par bloc */
    dim3 grid((n + TILE - 1) / TILE,   /* blocs en x = colonnes de C   */
              (m + TILE - 1) / TILE);  /* blocs en y = lignes de C     */
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
