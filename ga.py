
import random
import numpy as np
from deap import base, creator, tools
from nn import Brain

class GeneticAlgorithm(base.Toolbox):
    def __init__(self, input_size, n_pop=100, p_cx=0.7, p_mut=0.7, d_mut=0.05, decay=0.01):
        super(GeneticAlgorithm, self).__init__()
        self.n_pop, self.decay = n_pop, decay
        self.p_cx, self.p_mut, self.d_mut = p_cx, p_mut, d_mut
        creator.create('Fitness', base.Fitness, weights=(1.0,))
        creator.create('Individual', Brain, fitness=creator.Fitness)
        self.register('individual', creator.Individual, input_size)
        self.register('population', tools.initRepeat, list, self.individual)
        self.queue = self.population(n=self.n_pop)
        self.graveyard = tools.HallOfFame(maxsize=self.n_pop)

    def crossover(self, ind1, ind2):
        params1, params2 = ind1.parameters, ind2.parameters
        bools = [np.random.random(i.shape) < 0.5 for i in params1]
        ind1.params = [np.where(b, i, j) for b, i, j in zip(bools, params1, params2)]
        ind2.params = [np.where(b, j, i) for b, i, j in zip(bools, params1, params2)]

    def mutate(self, ind):
        params = ind.parameters
        rands = [np.random.random(i.shape) for i in params]
        bools = [np.random.random(i.shape) < self.d_mut for i in params]
        ind.parameters = [np.where(b, i, j) for b, i, j in zip(bools, rands, params)]

    def next(self):
        for ind in self.graveyard:
            ind.fitness.values = [i*(1-self.decay) for i in ind.fitness.values]

        if len(self.queue) > self.n_pop:
            return self.queue.pop(0)

        elif len(self.graveyard) == self.n_pop:
            parent1 = self.clone(random.choice(self.graveyard))

            if random.random() < self.p_mut:
                self.mutate(parent1)

            if random.random() < self.p_cx:
                parent2 = self.clone(random.choice(self.graveyard))

                if random.random() < self.p_mut:
                    self.mutate(parent2)

                self.crossover(parent1, parent2)
                self.queue.append(parent2)

            self.queue.append(parent1)
            return self.queue.pop(0)

        else:
            self.queue.append(self.individual())
            return self.queue.pop(0)

def test():
    pass

if __name__ == '__main__':
    test()
