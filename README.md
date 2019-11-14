# Koi

Simulated koi fish learn to eat food.

## Prerequisites

### [Pytorch](https://pytorch.org/)

This project only uses the CPU since pytorch is not being used for training.

```
conda install pytorch-cpu -c pytorch
```

### [Pygame](https://www.pygame.org/)

```
pip install pygame
```

### [DEAP](https://deap.readthedocs.io/)

```
pip install deap
```

## Running

```
python koi.py
```

Optional arguments:

```
  --fish N_FISH   Number of fish on the screen at once.
  --food N_FOOD   Number of food on the screen at once.
  --pop N_POP     Genetic algorithm queue and graveyard size.
  --draw {0,1,2}  Draw mode. 0: Do not draw, 1: Simple shapes, 2: Full graphics.
```

Example:

```
python koi.py --fish 8 --food 8 --pop 64 --draw 2
```
