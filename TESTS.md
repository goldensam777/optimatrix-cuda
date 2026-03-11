# TESTS.md — Rapport de tests optimatrix-cuda

Plateforme : Google Colab — GPU NVIDIA T4 (sm_75, 16 Go HBM2)
Date       : 2026-03-11
Commit     : 348c139 (branche `secours-demo`)
Commande   : `make clean && make tests ARCH=sm_75`

---

## Résultats globaux

| Phase | Module | Tests | Statut |
|-------|--------|-------|--------|
| 1 | Activations (ReLU, Sigmoid, SiLU, Softplus) | 4/4 | OK |
| 1 | Hadamard | 1/1 | OK |
| 2 | GEMV + GEMM | 7/7 | OK |
| 3 | Conv1D depthwise causale | 5/5 | OK |
| 3 | Selective Scan 1D (Blelloch) | 5/5 | OK |
| 4 | Selective Scan 2D — 4 stratégies | 16/16 | OK |
| 5 | MambaBlock forward complet | 3/3 | OK |
| **Total** | | **41/41** | **OK** |

---

## Phase 1 — Activations + Hadamard

```
=== Test Activations (N=1024) ===
  ReLU       : max_err = 0.00e+00  OK
  Sigmoid    : max_err = 1.19e-07  OK
  SiLU       : max_err = 1.19e-07  OK
  Softplus   : max_err = 2.38e-07  OK
4/4 tests OK

=== Test Hadamard (N=2048) ===
  Hadamard : max_err = 0.00e+00  OK
1/1 tests OK
```

Erreurs nulles ou inférieures à l'epsilon machine float32 (≈ 1.19e-07). Normal pour des opérations
élément par élément sans accumulation.

---

## Phase 2 — GEMV + GEMM

```
=== Test GEMV / GEMM ===
  GEMV 16x16   : max_err = 1.19e-07  OK
  GEMV 64x64   : max_err = 9.54e-07  OK
  GEMV 37x53   : max_err = 1.19e-06  OK
  GEMM 16x16x16  : max_err = 3.58e-07  OK
  GEMM 64x64x64  : max_err = 1.91e-06  OK
  GEMM 33x47x25  : max_err = 7.15e-07  OK
  GEMM 128x128x128 : max_err = 2.86e-06  OK
7/7 tests OK
```

L'erreur croît légèrement avec la taille (accumulation sur k termes). Reste largement
dans la tolérance float32 (< 1e-5).

---

## Phase 3 — Conv1D depthwise causale

```
=== Test Conv1D depthwise causale ===
  Conv1D L=4    D=3    K=2 bias=0 : max_err=2.98e-08  OK
  Conv1D L=128  D=64   K=4 bias=1 : max_err=4.77e-07  OK
  Conv1D L=256  D=128  K=3 bias=0 : max_err=2.38e-07  OK
  Conv1D L=512  D=16   K=4 bias=1 : max_err=4.77e-07  OK
  Conv1D L=64   D=64   K=8 bias=0 : max_err=3.58e-07  OK
5/5 tests OK
```

Convolution causale (pas de fuite depuis le futur). Testé avec et sans biais, noyaux K=2 à 8.

---

## Phase 3 — Selective Scan 1D

### Correctness

```
=== Correctness — Scan 1D ===
  Scan1D L=16   D=8   M=4  : max_err=2.98e-08  OK
  Scan1D L=64   D=16  M=8  : max_err=1.19e-07  OK
  Scan1D L=128  D=32  M=8  : max_err=1.94e-07  OK
  Scan1D L=512  D=64  M=16 : max_err=8.94e-07  OK
  Scan1D L=1024 D=128 M=16 : max_err=1.91e-06  OK
5/5 tests OK
```

Implémentation Blelloch (parallel prefix scan, O(log L) depth) vérifiée contre
référence CPU séquentielle. L=16..1024, D=8..128, M=4..16.

### Benchmark (temps moyen, 100 répétitions)

```
  Config                Temps
  L=64   D=32  M=8       0.026 ms
  L=128  D=64  M=8       0.047 ms
  L=256  D=128 M=16      0.451 ms
  L=512  D=128 M=16      0.713 ms
  L=1024 D=128 M=16      1.344 ms
```

---

## Phase 4 — Selective Scan 2D (4 stratégies)

### Correctness (vs référence CPU séquentielle)

```
  naive        4x4   D=8  M=4  : max_err=2.98e-08  OK
  naive_vec    4x4   D=8  M=4  : max_err=2.98e-08  OK
  coop         4x4   D=8  M=4  : max_err=2.98e-08  OK
  tiled        4x4   D=8  M=4  : max_err=2.98e-08  OK
  naive        8x8   D=16 M=8  : max_err=4.77e-07  OK
  naive_vec    8x8   D=16 M=8  : max_err=7.15e-07  OK
  coop         8x8   D=16 M=8  : max_err=4.77e-07  OK
  tiled        8x8   D=16 M=8  : max_err=4.77e-07  OK
  naive        6x10  D=8  M=4  : max_err=1.19e-07  OK
  naive_vec    6x10  D=8  M=4  : max_err=1.19e-07  OK
  coop         6x10  D=8  M=4  : max_err=1.19e-07  OK
  tiled        6x10  D=8  M=4  : max_err=1.19e-07  OK
  naive        16x16 D=32 M=8  : max_err=3.81e-05  OK
  naive_vec    16x16 D=32 M=8  : max_err=4.58e-05  OK
  coop         16x16 D=32 M=8  : max_err=3.81e-05  OK
  tiled        16x16 D=32 M=8  : max_err=3.81e-05  OK
Correctness : 16/16 OK
```

### Benchmark (temps moyen, ms)

```
  Config              naive    naive_vec    coop      tiled     best
  8x8  D16 M8         0.055    0.046        0.045     0.081     <- coop
  16x16 D32 M8        0.141    0.084        0.119     0.236     <- naive_vec
  32x32 D64 M8        0.337    0.198        0.297     1.029     <- naive_vec
  64x64 D128 M16      2.548    0.920        2.458    11.347     <- naive_vec
```

### Analyse des stratégies

| Stratégie | Principe | Comportement observé |
|-----------|----------|----------------------|
| `naive` | 1 kernel/diagonale, 1 thread/(pos,d), boucle M | Référence correcte, vitesse modérée |
| `naive_vec` | 1 thread/(pos,d,m), warp shuffle sur M | **Meilleure à grande taille** — 2.7× plus rapide que naive (64×64) |
| `coop` | Persistent kernel + `grid.sync()` | Meilleure à petite taille (8×8), dégradation à grande taille |
| `tiled` | Wavefront par tuiles 8×8 | Overhead synchronisation dominant — lent à grande taille |

`naive_vec` tire parti des **warp shuffle** (`__shfl_down_sync`) pour réduire M en
log₂(M) étapes sans passer par la mémoire partagée. Impossible sur CPU (pas de warp
en AVX2).

---

## Phase 5 — MambaBlock forward complet

```
=== TEST MambaBlock forward complet ===
  MambaBlock L=8    dim=16  D=8   M=4  K=2 : max_err=2.17e-06  OK
  MambaBlock L=16   dim=32  D=16  M=8  K=4 : max_err=4.58e-05  OK
  MambaBlock L=32   dim=64  D=32  M=8  K=4 : max_err=4.04e-04  OK
3/3 tests OK
```

Pipeline complet (7 étapes) vérifié contre référence CPU :
W_in → SiLU → Conv1D → dt/B/C → scan1D → gating → W_out.

L'erreur augmente avec la taille (accumulation sur 7 étapes), mais reste sous la
tolérance fixée à 1e-2. Normal pour un pipeline en float32.

---

## Notes techniques

- Tolérance `TOL = 1e-2` pour MambaBlock (pipeline multi-étapes), `1e-4` pour les
  modules individuels.
- Valeurs de A contraintes à [-3.0, -1.4] pour les tests 2D (stabilité numérique :
  garantit `a1 + a2 < 1` dans la récurrence 2D).
- Scan 1D : bascule automatique Blelloch (L ≤ 1024) / séquentiel (L > 1024).
- GPU : T4 — Turing, sm_75, 2560 CUDA cores, 320 Tensor cores (non utilisés ici).
