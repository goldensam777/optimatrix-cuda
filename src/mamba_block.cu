/*
 * mamba_block.cu — MambaBlock complet (forward 1D)
 *
 * Pipeline :
 *
 *   x_in [L, dim]
 *      │
 *      ├─ W_in [2D, dim] ──GEMM──► [x, z] [L, 2D] ──split──► x[L,D]  z[L,D]
 *      │                                                          │         │
 *      │                                                       SiLU(x)     │
 *      │                                                          │         │
 *      │                                                       Conv1D       │
 *      │                                                          │         │
 *      ├─ W_dt [D, dim]  ──GEMM──► dt [L, D] ──softplus──►       │         │
 *      ├─ W_B  [D*M,dim] ──GEMM──► B  [L, D, M]                  │         │
 *      ├─ W_C  [D*M,dim] ──GEMM──► C  [L, D, M]                  │         │
 *      │                                                          │         │
 *      │                              A [D, M] (fixe) ──────► scan1d        │
 *      │                                                          │         │
 *      │                                               y_scan [L, D]        │
 *      │                                                          │         │
 *      │                                               y_scan * SiLU(z) ◄───┘
 *      │                                                          │
 *      └─ W_out [dim, D] ──GEMM──► y_out [L, dim]
 *
 * Tous les GEMM ici sont des "batched GEMV" :
 *   Pour chaque position t, on fait y[t] = W @ x_in[t].
 *   Equivalent a GEMM(W, X^T) avec X = x_in.
 *
 * Parametres :
 *   L    : longueur de sequence
 *   dim  : dimension entree/sortie
 *   D    : canaux SSM (state_size)
 *   M    : dimension etat cache
 *   K    : taille noyau conv1d
 */

#include <stdio.h>
#include "optimatrix.h"
#include <math.h>

/* ── Struct de parametres ────────────────────────────────────────── */

typedef struct {
    int L, dim, D, M, K;

    /* Poids (pointeurs device) */
    const float *W_in;    /* [2*D, dim]   projection entree -> x, z    */
    const float *W_dt;    /* [D,   dim]   projection -> delta           */
    const float *W_B;     /* [D*M, dim]   projection -> B               */
    const float *W_C;     /* [D*M, dim]   projection -> C               */
    const float *A;       /* [D,   M]     transition SSM (log-domaine)  */
    const float *conv_w;  /* [K,   D]     noyau conv1d                  */
    const float *conv_b;  /* [D]          biais conv1d (peut etre NULL) */
    const float *W_out;   /* [dim, D]     projection sortie             */
} MambaParams;

/* ── Workspace intermediaire ─────────────────────────────────────── */

typedef struct {
    float *xz;     /* [L, 2*D]   sortie W_in avant split        */
    float *x;      /* [L, D]     x = xz[:,  :D] apres SiLU+conv */
    float *z;      /* [L, D]     z = xz[:, D:] avant gating     */
    float *dt;     /* [L, D]     delta apres softplus            */
    float *B;      /* [L, D, M]  B selectif                     */
    float *C;      /* [L, D, M]  C selectif                     */
    float *y_scan; /* [L, D]     sortie du scan                  */
    float *h;      /* [L, D, M]  etats caches scan               */
    float *y_gate; /* [L, D]     y_scan * SiLU(z)               */
} MambaWorkspace;

/* ── Kernel : split xz en x et z ─────────────────────────────────── */
/*
 * xz [L, 2D] -> x [L, D] et z [L, D]
 * x reçoit SiLU directement ici pour economiser un kernel.
 */

__global__ void split_silu_kernel(
    const float *xz, float *x, float *z,
    int L, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * D) return;

    int d = tid % D;
    int t = tid / D;

    float xv = xz[t * 2 * D + d];
    float zv = xz[t * 2 * D + D + d];

    /* x = SiLU(x) = x * sigmoid(x) */
    x[t * D + d] = xv / (1.0f + expf(-xv));
    z[t * D + d] = zv;
}

/* ── Kernel : gating y = y_scan * SiLU(z) ───────────────────────── */

__global__ void gate_kernel(
    const float *y_scan, const float *z,
    float *y_out,
    int L, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * D) return;

    float zv = z[tid];
    float gz = zv / (1.0f + expf(-zv));   /* SiLU(z) */
    y_out[tid] = y_scan[tid] * gz;
}

/* ── Alloc/free workspace ─────────────────────────────────────────── */

static MambaWorkspace alloc_workspace(int L, int D, int M)
{
    MambaWorkspace ws;
    OM_CHECK(cudaMalloc(&ws.xz,     L * 2 * D * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.x,      L * D     * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.z,      L * D     * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.dt,     L * D     * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.B,      L * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.C,      L * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.y_scan, L * D     * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.h,      L * D * M * sizeof(float)));
    OM_CHECK(cudaMalloc(&ws.y_gate, L * D     * sizeof(float)));
    return ws;
}

static void free_workspace(MambaWorkspace *ws)
{
    cudaFree(ws->xz);     cudaFree(ws->x);  cudaFree(ws->z);
    cudaFree(ws->dt);     cudaFree(ws->B);  cudaFree(ws->C);
    cudaFree(ws->y_scan); cudaFree(ws->h);  cudaFree(ws->y_gate);
}

/* ── Forward ─────────────────────────────────────────────────────── */
/*
 * x_in  : [L, dim]   entree
 * y_out : [L, dim]   sortie
 * p     : parametres (poids sur device)
 */

void om_mamba_block_forward(
    const float *d_x_in, float *d_y_out,
    const MambaParams *p)
{
    int L   = p->L;
    int dim = p->dim;
    int D   = p->D;
    int M   = p->M;
    int K   = p->K;

    MambaWorkspace ws = alloc_workspace(L, D, M);

    int blocks_LD  = (L * D     + 255) / 256;
    int blocks_LDM = (L * D * M + 255) / 256;

    /* ── Etape 1 : projection entree x_in -> xz ─────────────────── */
    /*
     * xz = x_in @ W_in^T
     * x_in : [L, dim]    W_in : [2D, dim]   xz : [L, 2D]
     * = GEMM(x_in, W_in^T) avec m=L, n=2D, k=dim
     */
    om_gemm(d_x_in, p->W_in, ws.xz, L, 2 * D, dim);

    /* ── Etape 2 : split + SiLU(x), garder z ────────────────────── */
    split_silu_kernel<<<blocks_LD, 256>>>(ws.xz, ws.x, ws.z, L, D);

    /* ── Etape 3 : Conv1D depthwise sur x ───────────────────────── */
    om_conv1d(ws.x, p->conv_w, p->conv_b, ws.x, L, D, K);

    /* ── Etape 4 : projections selectifs dt, B, C ───────────────── */
    /*
     * dt = softplus(x_in @ W_dt^T)  [L, D]
     * B  = x_in @ W_B^T              [L, D*M]
     * C  = x_in @ W_C^T              [L, D*M]
     */
    om_gemm(d_x_in, p->W_dt, ws.dt, L, D,     dim);
    om_softplus(ws.dt, ws.dt, L * D);

    om_gemm(d_x_in, p->W_B,  ws.B,  L, D * M, dim);
    om_gemm(d_x_in, p->W_C,  ws.C,  L, D * M, dim);

    /* ── Etape 5 : Selective scan 1D ────────────────────────────── */
    /*
     * Inputs : x[L,D], A[D,M], B[L,D,M], C[L,D,M], dt[L,D]
     * Output : y_scan[L,D], h[L,D,M]
     *
     * NOTE : c'est ici que la sequentialite est incontournable.
     * Voir RESEARCH.md pour les pistes futures.
     */
    om_scan1d_forward(ws.x, p->A, ws.B, ws.C, ws.dt,
                      ws.y_scan, ws.h, L, D, M);

    /* ── Etape 6 : Gating y = y_scan * SiLU(z) ──────────────────── */
    gate_kernel<<<blocks_LD, 256>>>(ws.y_scan, ws.z, ws.y_gate, L, D);

    /* ── Etape 7 : projection sortie -> dim ─────────────────────── */
    /*
     * y_out = y_gate @ W_out^T
     * y_gate : [L, D]   W_out : [dim, D]   y_out : [L, dim]
     */
    om_gemm(ws.y_gate, p->W_out, d_y_out, L, dim, D);

    cudaDeviceSynchronize();
    free_workspace(&ws);
}
