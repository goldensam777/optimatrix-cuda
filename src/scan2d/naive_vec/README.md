# Strategie A' — Naive vectorisee sur M

## Ce qui change par rapport a A (naive)

```
naive     : 1 thread = (pos, d)     -> boucle sur M      -> ecrit y directement
naive_vec : 1 thread = (pos, d, m)  -> calcul parallele  -> reduction warp -> y
```

## Pourquoi c'etait impossible sur CPU (ASM x86-64)

1. **exp() non vectorisable** : AVX2 n'a pas d'instruction `vexpps`.
   On devait appeler exp() une par une, ou ecrire une approximation
   polynomiale (roadmap du projet original).

2. **Pas de reduction hardware** : Sur CPU, reduire M valeurs =
   M additions sequentielles ou un arbre SIMD penible a ecrire.

3. **Parallelisme limite** : AVX2 = 8 floats max.
   Pour M=16, on ne peut meme pas tout mettre dans un registre.

## Pourquoi c'est naturel sur GPU

1. **SFU (Special Function Units)** : Le GPU a des unites dediees
   qui calculent exp(), sin(), cos() en ~20 cycles. Chaque thread
   a acces a sa propre SFU. 1000 threads = 1000 exp() en parallele.

2. **Warp shuffle** : `__shfl_down_sync()` permet de sommer M valeurs
   en log2(M) etapes sans passer par la memoire.
   Pour M=16 : 4 etapes au lieu de 16 additions.

3. **Parallelisme massif** : On passe de (dlen * D) threads
   a (dlen * D * M) threads. Pour une diagonale de 64 positions,
   D=128, M=16 : 64*128*16 = 131072 threads au lieu de 8192.

## Impact mesurable

- **x M** plus de threads actifs sur le GPU
- **Meilleure occupation** des SMs (plus de warps a scheduler)
- **Reduction en log2(M)** au lieu de M (lineaire)
- Potentiellement **x2 a x4** sur les configurations ou le GPU
  etait sous-utilise avec la version naive

## Contrainte

M_pad doit etre une puissance de 2 (pour le warp shuffle).
M=8 -> ok, M=16 -> ok, M=12 -> on pad a 16 (threads fantomes).
