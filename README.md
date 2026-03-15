# optimatrix-cuda

Implémentation CUDA custom de primitives de calcul :
activations, GEMM/GEMV, Conv1D depthwise causale, selective scan 1D et scan 2D wavefront.
Port GPU du projet [optimatrix](https://github.com/goldensam777/optimatrix) (ASM x86-64 / AVX2).

> Projet de recherche — Samuel Yevi, IFRI-UAC, L1 Systèmes Embarqués

---

## Aperçu

| Module | Implémentation | Statut |
|--------|---------------|--------|
| Activations (ReLU, Sigmoid, SiLU, Softplus) | 1 thread/élément | ✅ |
| Hadamard | 1 thread/élément | ✅ |
| GEMM / GEMV | Tiling shared memory (16×16) | ✅ |
| Conv1D depthwise causale | Shared memory pour le noyau | ✅ |
| Selective Scan 1D | Parallel prefix scan (Blelloch) | ✅ |
| Selective Scan 2D — naive | 1 kernel par diagonale | ✅ |
| Selective Scan 2D — naive\_vec | Vectorisé sur M + warp shuffle | ✅ |
| Selective Scan 2D — coop | Persistent kernel + `grid.sync()` | ✅ |
| Selective Scan 2D — tiled | Wavefront par tuiles 8×8 | ✅ |

---

## Prérequis

- CUDA Toolkit ≥ 11.0
- GPU NVIDIA compute capability ≥ 7.5 (Turing)
- `make` (Linux) ou PowerShell (Windows)

---

## Compilation et tests

```bash
# Ajuster ARCH selon la carte GPU :
# sm_75 → RTX 20xx / MX450 / T4
# sm_86 → RTX 30xx
# sm_89 → RTX 40xx

make tests ARCH=sm_75
```

**Sur Windows (PowerShell) :**
```powershell
.\run_tests.ps1 -Arch sm_75
# Rapport sauvegardé dans results/report_YYYYMMDD_HHMMSS.txt
```

**Sur Kaggle / Google Colab :**
```bash
git clone https://github.com/goldensam777/optimatrix-cuda.git
cd optimatrix-cuda
make tests   # T4 = sm_75 par défaut
```

---

## Structure

```
src/
  activations.cu          # ReLU, Sigmoid, SiLU, Softplus
  hadamard.cu             # Produit élément par élément
  gemm.cu                 # GEMV + GEMM tiled
  conv1d.cu               # Conv1D depthwise causale
  scan1d.cu               # Selective scan 1D (Blelloch)
  scan2d/
    naive/                # A  — 1 kernel/diagonale
    naive_vec/            # A' — vectorisé sur M + warp reduction
    coop/                 # B  — persistent kernel + grid.sync()
    tiled/                # C  — wavefront par tuiles 8×8
include/
  optimatrix.h            # API publique
tests/
  test_activations.cu
  test_hadamard.cu
  test_gemm.cu
  test_conv1d.cu
  test_scan1d.cu
  test_scan2d.cu          # Correctness + benchmark des 4 stratégies
results/                  # Rapports de tests générés
RESEARCH.md               # Pistes de recherche : scan 2D en O(log N)
```

---

## Ce qui est nouveau par rapport au projet CPU

| Aspect | CPU (ASM/AVX2) | GPU (CUDA) |
|--------|---------------|------------|
| `exp()` | Scalaire, non vectorisable | Parallèle sur SFU (~20 cycles/thread) |
| Réduction sur M | Boucle séquentielle | Warp shuffle `__shfl_down_sync` en log₂(M) |
| Scan 1D | O(L) depth | O(log L) depth — Blelloch |
| Parallélisme GEMM | AVX2 ×6.9 | Tiling shared memory |
| Scan 2D | 1 stratégie (wavefront ASM) | 4 stratégies comparées |

---

## Limitation connue

Le scan 2D reste borné par `O(d1 + d2)` étapes séquentielles (diagonales du DAG).
C'est une limite **théorique**, pas d'implémentation — voir [`RESEARCH.md`](RESEARCH.md)
pour les pistes vers un scan 2D en O(log(d1·d2)).
