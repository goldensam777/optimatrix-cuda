# Strategie B — Persistent kernel + cooperative groups

## Principe

Un seul kernel lance, qui reste actif pendant tout le scan.
Entre chaque diagonale, tous les blocs se synchronisent via
`cooperative_groups::this_grid().sync()`.

## Avantages

- Zero overhead de lancement entre diagonales
- Le kernel reste "chaud" : registres, caches, tout est en place
- Code relativement simple (meme structure que A, juste la boucle est dans le kernel)

## Inconvenients

- Necessite compute capability >= 6.0
- Le nombre de blocs est limite a ce que le GPU peut executer
  simultanement (sinon deadlock sur grid.sync)
- Necessite `cudaLaunchCooperativeKernel()` au lieu de <<<>>>
- Compilation avec `-rdc=true` requise

## Contraintes techniques

- Verifier `cudaDeviceProp::cooperativeLaunch` avant utilisation
- Le grid size ne doit pas depasser `maxBlocksPerSM * numSMs`

## Quand l'utiliser

- Grilles moyennes ou l'overhead de A devient visible
- GPU modernes avec bon support cooperative launch
