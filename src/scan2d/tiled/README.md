# Strategie C — Wavefront par tuiles (tiled)

## Principe

On decoupe la grille d1 x d2 en tuiles de TILE x TILE (defaut: 8).
Le wavefront opere au niveau des tuiles : les tuiles sur une meme
anti-diagonale de tuiles sont independantes.

Dans chaque tuile, un seul thread traite toutes les positions
en wavefront interne (micro-diagonales).

## Avantages

- Beaucoup moins de kernel launches :
  ceil(d1/T) + ceil(d2/T) - 1 au lieu de d1+d2-1
  Ex: 64x64, T=8 -> 15 launches au lieu de 127
- Meilleure localite : les h intra-tile restent en cache L1/L2
- Bonne scalabilite sur les grandes grilles

## Inconvenients

- Plus de travail sequentiel par thread (toutes les positions du tile)
- Les bords (tiles partielles) necessitent des gardes
- Un peu plus de code que A

## Parametre TILE

- TILE=8 : bon compromis (15 launches pour 64x64)
- TILE=16 : moins de launches mais plus de travail sequentiel par thread
- A ajuster experimentalement selon le GPU

## Quand l'utiliser

- Grandes grilles (64x64 et au-dela)
- Quand l'overhead de A devient le goulot
- Meilleur compromis simplicite/performance en general
