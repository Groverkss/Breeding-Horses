from os import popen
from typing import List, Callable, Tuple
from nptyping import NDArray

from pprint import pprint
import numpy as np

from scipy import stats as sciStats

Individual = NDArray
Population = NDArray
Fitness = NDArray


class GeneticAlgorithm:
    """Definition for genetic algorithm used"""

    def __init__(
        self,
        populationSize: int,
        iterationsLeft: int,
        mutationProbability: float = 1,
        mutationStandardDev: float = 0.5,
    ) -> None:

        """Constructor for genetic algorithm"""
        self.vectorSize = 11
        self.scalingFactor = 10
        self.rng = np.random.default_rng()
        self.populationSize = populationSize
        self.iterationsLeft = iterationsLeft
        self.mutationProbability = mutationProbability
        self.mutationStandardDev = mutationStandardDev

    def runEvolution(self, steps: int) -> None:
        """Run step iterations of genetic algorithm"""
        population = self.initializePopulation()

        for iteration in range(steps):
            fitness = self.calculateFitness(population)

            self.checkTermination()
            if self.iterationsLeft % 10 == 0:
                print(f"Enter new mutation parameters:")
                self.mutationProbability = float(input())
                self.mutationStandardDev = float(input())

            sortedIndices = fitness.argsort()

            population = population[sortedIndices]
            fitness = fitness[sortedIndices]

            print("Fitness: ", fitness[0], flush=True)
            print("Vector: ", population[0])
            with open("output.txt", "a") as outfile:
                pprint(
                    f"---Iterations Left: {self.iterationsLeft}---",
                    stream=outfile,
                )
                pprint(population, stream=outfile)
                pprint(fitness, stream=outfile)

            # Gurantee that top two will be selected without any mutation
            # or crossover
            nextGeneration = population[:2, :]

            for crossoverIteration in range((self.populationSize // 2) - 1):

                # Select two parents from population
                parent_a, parent_b = self.selectTwo(population)

                # Cross them
                offspring_a, offspring_b = self.crossOver(parent_a, parent_b)

                # Mutate
                offspring_a = self.mutateOffspring(offspring_a)
                offspring_b = self.mutateOffspring(offspring_b)

                # Add to next generation
                nextGeneration = np.concatenate(
                    (nextGeneration, [offspring_a, offspring_b]), axis=0
                )

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
        self.iterationsLeft -= 1
        return self.iterationsLeft > 0

    def selectTwo(
        self, population: Population
    ) -> Tuple[Individual, Individual]:
        """Selects to random individuals from a given population"""

        indices = np.random.choice(population.shape[0], 2, replace=False)
        return population[indices]

    def crossOver(
        self, parent_a: Individual, parent_b: Individual
    ) -> Tuple[Individual, Individual]:

        """Crosses two parents to give a two new offsprings"""
        # sliceIndex = np.random.randint(0, self.vectorSize)
        sliceIndex = self.rng.choice(
            np.arange(self.vectorSize), self.vectorSize, replace=False
        )

        # offspring_a, offspring_b = (
        #     np.concatenate((parent_a[:sliceIndex], parent_b[sliceIndex:])),
        #     np.concatenate((parent_b[:sliceIndex], parent_b[sliceIndex:])),
        # )

        offspring_a = np.copy(parent_a)
        offspring_b = np.copy(parent_b)

        offspring_a[sliceIndex], offspring_b[sliceIndex] = (
            offspring_b[sliceIndex],
            offspring_a[sliceIndex],
        )

        return offspring_a, offspring_b

    def mutateOffspring(self, offspring: Individual) -> Individual:
        """
        Mutates an individual
            Current Algo: Select some indices randomly and choose a new number from
        a gaussian distribution with mean as the number at that index
        """
        flagArray = sciStats.bernoulli.rvs(  # type:ignore
            p=self.mutationProbability, size=offspring.shape
        )

        generateGaus = lambda x: np.clip(
            np.random.normal(loc=x, scale=self.mutationStandardDev),
            -self.scalingFactor,
            self.scalingFactor,
        )
        vectorizedGenerateGaus = np.vectorize(generateGaus)
        gausArray = vectorizedGenerateGaus(offspring)

        offspring[flagArray == 1] = gausArray[flagArray == 1]
        return offspring

    def calculateFitness(self, population: Population) -> Fitness:
        """Returns fitness array for the population"""
        return np.sum(population ** 2, axis=1)

        # TODO Implement fitness function based on errors


test = GeneticAlgorithm(10, 100, 1, 1)
test.runEvolution(100)
