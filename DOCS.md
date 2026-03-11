# DOCS.md — Documentation Technique

## Notations

```
N    — nombre de dimensions spatiales
d_k  — taille de la dimension k  (k = 1..N)
D    — dimension du modele (d_model / canaux)
M    — dimension de l'etat cache (d_state)
K    — taille du noyau de convolution par axe
P    — nombre d'elements total : P = d_1 x d_2 x ... x d_N
```

---

## 1. ConvND (Convolution N-dimensionnelle separable)

### Description
Chaine de Conv1D depthwise causale axe par axe (du dernier au premier),
puis biais. Le noyau ASM AVX2 (`conv1d_depthwise_avx2`) fait le calcul brut,
l'orchestrateur C (`convnd.c`) gere le gather/scatter pour les axes strides.

### API unifiee

```c
void convnd(ConvNDParams *p, ConvNDMode mode, ConvNDWorkspace *ws);
```

**Modes :**

| Mode | ws fourni | Comportement |
|------|-----------|-------------|
| `CONVND_FORWARD` | NULL | Forward pur, pas d'overhead |
| `CONVND_FORWARD` | oui | Forward + sauvegarde des intermediaires dans ws |
| `CONVND_BACKWARD` | oui | Backward, utilise les intermediaires sauves |
| `CONVND_BACKWARD` | NULL | Backward, recompute les intermediaires en interne |
| `CONVND_COMPLETE` | NULL | Forward + Backward enchaines |

**Workspace :** stocke les N+1 tenseurs intermediaires de la chaine forward
pour que le backward n'ait pas a les recomputer.

```c
ConvNDWorkspace *ws = convnd_workspace_create(&p);
convnd(&p, CONVND_FORWARD, ws);
convnd(&p, CONVND_BACKWARD, ws);
convnd_workspace_free(ws);
```

### Forward

```
inter[0] = input
inter[1] = apres conv sur axe ndims-1
inter[2] = apres conv sur axe ndims-2
...
inter[ndims] = output (avant biais)
output = inter[ndims] + biais
```

### Backward

Ordre inverse du forward : backprop de l'axe 0 vers l'axe ndims-1.
A chaque etape k :
- Axe = k
- Input du step forward = inter[ndims-1-k]
- Produit dy_suivant et accumule dkernel[axe]

Le backward 1D (conv1d_backward_c) calcule :
```
dkernel[j,d] += sum_t dy[t,d] * input[t-K+1+j, d]
dinput[s,d]  += sum_t dy[t,d] * kernel[t-s+K-1, d]
```

### Complexite

```
Forward  : O(P * D * K * ndims)    — K taps par axe, ndims axes
Backward : O(P * D * K * ndims)    — meme complexite + recompute forward si pas de ws
Memoire  : O(P * D)                — tenseurs
         + O(ndims * K * D)        — noyaux
         + O((ndims+1) * P * D)    — workspace (si utilise)
```

---

## 2. GEMV — Produit Matrice x Vecteur

### Description
Projection lineaire : `y = W * x` avec `W in R^{D_out x D_in}`, `x in R^{D_in}`
Utilise pour : calcul de B(n), C(n), delta(n).

### Complexite
```
Calcul  : O(P * D_in * D_out)
Memoire : O(D_in * D_out)
```

### Operations ASM
- Produit scalaire W * x (AVX2 : 8 float32 en parallele)

---

## 3. GEMM — Produit Matrice x Matrice

### Description
`C = A * B` avec `A in R^{m x k}`, `B in R^{k x n}`
Utilise pour : in_proj, out_proj (batch de projections).

### Complexite
```
Calcul  : O(m * n * k)    [2mnk flops]
Memoire : O(m*k + k*n + m*n)
```

### Operations ASM
- Blocking (tiling) pour cache L1/L2
- FMA vectorise (AVX2)

---

## 4. Hadamard ND — Produit element par element

### Description
`Z = X * Y` — multiplication terme a terme sur tenseurs de meme forme.
Utilise dans la gate de Mamba : `y = x * silu(z)`

### Complexite
```
Calcul  : O(P * D)
Memoire : O(P * D)
```

---

## 5. Selective Scan ND — Coeur du calcul

### Description
Resout la recurrence sur le DAG N-dimensionnel :

```
h(n) = sum_{k=1}^{N} A_k(n) * h(n - e_k)  +  B(n) * x(n)
y(n) = C(n) * h(n)
```

### Complexite
```
Calcul  : O(P * N * M^2) + O(P * D * M)
Memoire : O(P * M) + O(N * M^2)
```

### Ordonnancement wavefront (2D)
```
Diagonale 0 : h(0,0)
Diagonale 1 : h(1,0), h(0,1)           — independantes
Diagonale 2 : h(2,0), h(1,1), h(0,2)   — independantes
...
```

### Backward 1D
Trois specialisations :
- `scan1d_backward` generique sur `[L, D, M]`
- `M=1` specialise
- `M=1` avec `B/C` partages et `delta[t]` scalaire (chemin chaud Mamba)

---

## 6. MambaBlock — Bloc SSM complet

### Description
Implementation complete du bloc Mamba :
- Projections W_in (dim -> state_size) et W_out (state_size -> dim)
- A diagonal (log-parametrise), B/C partages
- Delta input-dependant via delta_proj + softplus
- Scan selectif 1D ou 2D (wavefront ASM)
- Backward complet avec accumulation des gradients
- Optimiseur MUONCLIP integre

### Lifecycle
```c
MambaBlock *b = mamba_block_create(&config);
mamba_block_init(b);                    // Xavier init, A_log spacing

// Entrainement
mamba_attach_optimizer(b, &opt_cfg);
mamba_block_forward(b, out, in, batch); // ou forward_2d
mamba_backward(b, dY, in, din, 0);     // ou backward_2d
mamba_optimizer_step(b, &opt_cfg);

mamba_block_free(b);
```

### Forward 1D
```
x_t -> W_in -> SiLU -> u_t (controleur)
x_t -> delta_proj -> softplus -> clamp -> dt_t
scan1d(u, A_log, B, C, delta) -> h_t
h_t -> W_out -> y_t
```

### Forward 2D
```
Meme pipeline, mais scan2d (wavefront) au lieu de scan1d.
Positions P = d1 * d2 traitees par diagonales anti.
```

### Backward
- Recompute le forward (store intermediaires)
- Backprop W_out -> scan backward -> SiLU backward -> W_in
- Accumule gradients dans MBOptimState

### OpenMP
Les boucles du forward store (`selective_scan_forward_store`) sont
parallelisables via `#pragma omp parallel for` (protege par `#ifdef _OPENMP`).

---

## 7. Activations

### SiLU (Swish)
```
silu(x) = x * sigmoid(x)
```

### Softplus
```
softplus(x) = log(1 + exp(x))
```
Utilise pour delta (doit etre positif).

---

## 8. Format des tenseurs en memoire

Stockage **row-major** (C-order), continu :

```
T[n_1][n_2]...[n_N][d] = base + (n_1*s_1 + n_2*s_2 + ... + n_N*s_N + d) * sizeof(float)
```

Alignement **32 bytes** requis pour AVX2.
