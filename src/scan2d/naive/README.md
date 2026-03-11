# Strategie A — Un kernel par diagonale

## Principe

La plus directe. On lance `d1+d2-1` kernels sequentiellement,
un par anti-diagonale. Les positions sur une meme diagonale
sont independantes et traitees en parallele par le GPU.

## Avantages

- Simple a ecrire et debugger
- Correctness evidente (meme logique que le CPU)
- Pas de contrainte sur le GPU (fonctionne partout)

## Inconvenients

- Overhead de lancement : ~5-10us par kernel x (d1+d2-1) diagonales
- Pour 64x64 : 127 launches = ~1ms perdu juste en overhead
- Faible occupation sur les petites diagonales (coins)

## Quand l'utiliser

- Comme reference de correctness
- Petites grilles ou le overhead est negligeable
- Prototypage rapide
