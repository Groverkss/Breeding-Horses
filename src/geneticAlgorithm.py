from typing import List, Callable, Tuple
from pprint import pprint
import numpy as np
from nptyping import NDArray

Individual = NDArray
Population = NDArray


class GeneticAlgorithm:
    """Definition for genetic algorithm used"""

    def __init__(self, populationSize: int, iterationsLeft: int) -> None:
        """Constructor for genetic algorithm"""
        self.vectorSize = 11
        self.scalingFactor = 10
        self.rng = np.random.default_rng()
        self.populationSize = populationSize
        self.iterationsLeft = iterationsLeft

    def runEvolution(self, steps: int) -> None:
        """Run step iterations of genetic algorithm"""

        for iteration in range(steps):
            population = self.initializePopulation()

            for iteration in range(steps):
                population = self.initializePopulation()

                self.checkTermination()

                # Gurantee that top two will be selected without any mutation
                # or crossover
                nextGeneration = population[:2, :]

                for crossoverIteration in range((self.populationSize // 2) - 1):

                    # Select two parents from population
                    parent_a, parent_b = self.selectTwo(population)

                    # Cross them
                    offspring_a, offspring_b = self.crossOver(
                        parent_a, parent_b
                    )

                    # Mutate
                    offspring_a = self.mutateOffspring(offspring_a)
                    offspring_b = self.mutateOffspring(offspring_b)

                    # Add to next generation
                    nextGeneration += [offspring_a, offspring_b]

                population = nextGeneration

    def initializePopulation(self) -> Population:
        """Initialize a population randomly"""
        population = self.rng.random((self.populationSize, self.vectorSize))

        # Scale and normalize population to be in range -10 to 10
        population = population * 2 * self.scalingFactor
        population = population - self.scalingFactor

        return population

    def checkTermination(self) -> bool:
        """Check if termination condition is satisfied"""
        return self.iterationsLeft > 0

    def getPopulation(self):
        pass

    def selectTwo(
        self, population: Population
    ) -> Tuple[Individual, Individual]:
        """Selects to random individuals from a given population"""

        indices = np.random.choice(population.shape[0], 2, replace=False)
        return population[indices]

    def crossOver(self, parent_a, parent_b) -> Tuple[Individual, Individual]:
        """Crosses two parents to give a two new offsprings"""
        # TODO
        return parent_a, parent_b

    def mutateOffspring(self, offspring) -> Individual:
        """Mutates an individual"""
        # TODO
        return offspring
