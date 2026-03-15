/*
 * gemm_backward.cu — Backward du GEMM : C = A @ B
 *
 * Pour C = A @ B  (A : m x k,  B : k x n,  C : m x n) :
 *
 *   dL/dA = dL/dC @ B^T   ->  om_gemm_ABt(dC, B, dA,  m, k, n)
 *   dL/dB = A^T @ dL/dC   ->  om_gemm_AtB(A,  dC, dB, k, n, m)
 *
 * ─────────────────────────────────────────────────────────────────
 * om_gemm_ABt(A, B, C, m, n, k) : C[m,n] = A[m,k] @ B^T
 *   B est stocke en [n x k].   C[i,j] = sum_l A[i,l] * B[j,l]
 *
 * Tiling shared memory (TILE x TILE) :
 *   sA[ty][tx] = A[row * k + (t*TILE + tx)]       -- chargement standard
 *   sB[ty][tx] = B[(blockIdx.x*TILE + tx) * k + (t*TILE + ty)]
 *             -> thread (ty,tx) lit B[col_tx * k + b_col] ou col_tx = col du thread tx
 *   Produit : sum_i sA[ty][i] * sB[i][tx]
 *           = sum_l A[row,l] * B[col,l] = C[row,col] ✓
 *
 * ─────────────────────────────────────────────────────────────────
 * om_gemm_AtB(A, B, C, m, n, k) : C[m,n] = A^T @ B
 *   A est stocke en [k x m].   C[i,j] = sum_l A[l,i] * B[l,j]
 *
 * Tiling :
 *   sA[ty][tx] = A[(t*TILE + tx) * m + row]       -- colonne de A
 *   sB[ty][tx] = B[(t*TILE + ty) * n + col]       -- standard
 *   Produit : sum_i sA[ty][i] * sB[i][tx]
 *           = sum_l A[l,row] * B[l,col] = C[row,col] ✓
 */

#include "optimatrix.h"

#define TILE 16

/* ── C = A @ B^T   (B stocke en [n x k]) ────────────────────────── */

__global__ void gemm_ABt_kernel(const float *A, const float *B,
                                float *C, int m, int n, int k)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row    = blockIdx.y * TILE + threadIdx.y;
    int col    = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        /* A[row][a_col] — chargement standard */
        int a_col = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < m && a_col < k)
            ? A[row * k + a_col] : 0.0f;

        /* B^T : on lit B[col_of_tx][b_col] ou col_of_tx varie en tx */
        int col_tx = blockIdx.x * TILE + threadIdx.x;
        int b_col  = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (col_tx < n && b_col < k)
            ? B[col_tx * k + b_col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

/* ── C = A^T @ B   (A stocke en [k x m]) ────────────────────────── */

__global__ void gemm_AtB_kernel(const float *A, const float *B,
                                float *C, int m, int n, int k)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   /* colonne de A = ligne de A^T */
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        /* A^T : on lit A[a_l * m + row] ou a_l varie en tx */
        int a_l = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (a_l < k && row < m)
            ? A[a_l * m + row] : 0.0f;

        /* B : chargement standard */
        int b_l = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (b_l < k && col < n)
            ? B[b_l * n + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_gemm_ABt(const float *d_A, const float *d_B, float *d_C,
                 int m, int n, int k)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    gemm_ABt_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}

void om_gemm_AtB(const float *d_A, const float *d_B, float *d_C,
                 int m, int n, int k)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    gemm_AtB_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
