# RESEARCH.md — Limites fondamentales et pistes futures

## Limitation identifiée : séquentialité du scan 2D

**Date** : 2026-03-11
**Auteur** : Samuel Yevi (IFRI-UAC, L1 Systèmes Embarqués)
**Contexte** : Projet optimatrix-cuda — implémentation GPU du selective scan 2D

---

### Le problème

La récurrence du selective scan 2D est :

```
h(i,j) = exp(dt·A1) · h(i-1,j) + exp(dt·A2) · h(i,j-1) + dt·B·x(i,j)
```

Cela induit un **DAG (Directed Acyclic Graph)** de dépendances :
- `h(i,j)` dépend de `h(i-1,j)` (haut) et `h(i,j-1)` (gauche)
- Les positions sur la même anti-diagonale `d = i + j` sont indépendantes
- Les diagonales elles-mêmes sont **strictement séquentielles**

```
Profondeur du DAG = d1 + d2 - 1  (nombre de diagonales)
Parallélisme max  = min(d1, d2)  (taille de la plus grande diagonale)

Pour une image 64x64 : 127 étapes séquentielles obligatoires
Pour une image 512x512 : 1023 étapes séquentielles obligatoires
```

**Conséquence** : même avec un GPU infini, le temps d'exécution est borné
inférieurement par `O(d1 + d2)`, pas `O(1)`.

---

### Pourquoi le scan 1D s'en sort (Blelloch)

En 1D, la récurrence `h_t = a_t · h_{t-1} + b_t` a une structure
d'opérateur **associatif** :

```
(a₁, b₁) ⊗ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)
```

Le **prefix scan parallèle** (Blelloch, 1990) exploite cette associativité
pour réduire la profondeur de O(L) à O(log L).

En 2D, l'opérateur de composition existe aussi :

```
(A₁, b₁) ⊗ (A₂, b₂) = ?
```

Mais `h(i,j)` dépend de **deux prédécesseurs**, pas un seul.
Cela brise la structure de chaîne linéaire nécessaire au prefix scan classique.

---

### Pistes de recherche futures

#### Piste 1 — Décomposition en scans 1D indépendants
**Idée** : reformuler le scan 2D comme une composition de scans 1D.
Ex : scan horizontal puis scan vertical, avec correction des termes croisés.

**Problème** : les termes croisés `h(i-1,j-1)` apparaissent et créent
des dépendances supplémentaires. La correction exacte n'est pas triviale.

**Référence à explorer** : méthodes de décomposition d'opérateurs pour
équations aux différences partielles (PDE splitting, Strang splitting).

#### Piste 2 — Approximation polynomiale du DAG
**Idée** : tronquer les dépendances longue-portée.
Si `exp(dt·A)` est proche de 0 pour des `dt` typiques, alors
`h(i,j)` dépend peu de `h(i-k, j)` pour `k` grand.

On pourrait limiter la fenêtre de dépendance à `W` étapes :
```
h(i,j) ≈ sum_{k=0}^{W} sum_{l=0}^{W} coeff(k,l) · x(i-k, j-l)
```
Ce qui ramène à une convolution 2D — parallélisable à 100%.

**Coût** : approximation, pas exact. Pertinent pour l'inférence, pas l'entraînement.

#### Piste 3 — Opérateur associatif 2D généralisé
**Idée** : trouver un opérateur `⊗₂D` tel que le scan 2D puisse s'exprimer
comme un prefix scan sur un treillis (lattice).

Sur un treillis 2D, le "prefix" en `(i,j)` est défini par l'ensemble
`{(i',j') : i'≤i, j'≤j}`. Un tel opérateur devrait satisfaire :

```
h(i,j) = ⊗₂D_{(i',j') ≤ (i,j)} (a(i',j'), b(i',j'))
```

**Difficulté** : l'associativité sur un treillis 2D est un problème ouvert
pour les récurrences à deux prédécesseurs.

**Référence à explorer** :
- Ladner & Fischer (1980) — parallel prefix computation
- Kogge & Stone (1973) — scan sur DAGs généraux
- Blelloch (1990) — prefix sums and their applications

#### Piste 4 — Parallélisme stochastique
**Idée** : pendant l'entraînement, perturber légèrement les dépendances
pour autoriser le calcul parallèle, puis corriger par gradient.

Similaire aux techniques de **parallel tempering** ou **asynchronous SGD**.

**Statut** : très spéculatif, aucune garantie de convergence connue.

#### Piste 5 — Hardware spécialisé (non-GPU)
**Idée** : concevoir un accélérateur matériel avec une topologie de mémoire
correspondant exactement au DAG du scan 2D wavefront.

Sur un chip systolique 2D (comme les TPU Google), chaque cellule
communique avec ses voisins directs — parfaitement adapté au wavefront.

**Référence** : Kung & Leiserson (1980) — systolic arrays.

---

### Ce que cette recherche apporterait

Si un scan 2D **exact** en O(log(d1+d2)) était trouvé :

| Métrique | Actuel (wavefront) | Théorique (prefix 2D) |
|----------|-------------------|----------------------|
| Profondeur | O(d1 + d2) | O(log(d1 · d2)) |
| Travail | O(d1 · d2 · D · M) | O(d1 · d2 · D · M) |
| Speedup potentiel | 1x | (d1+d2) / log(d1·d2) |

Pour 512x512 : speedup potentiel de **~93x** sur la profondeur.

---

### Note de suivi

Ce fichier sera mis à jour au fil des lectures et expériences.
Toute implémentation d'une piste doit d'abord passer par une preuve
de correction sur un exemple petit (d1=d2=4, D=2, M=2).
