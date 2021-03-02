from typing import List, Callable, Tuple
from nptyping import NDArray

from pprint import pprint
import numpy as np

from scipy import stats as sciStats

from client import getErrors

Individual = NDArray
Population = NDArray
Fitness = NDArray


class GeneticAlgorithm:
    """Definition for genetic algorithm used"""

    def __init__(
        self,
        populationSize: int,
        mutationProbability: float = 1,
    ) -> None:

        """Constructor for genetic algorithm"""
        self.vectorSize = 5
        self.rng = np.random.default_rng()
        self.scalingFactor = 10
        self.populationSize = populationSize
        self.mutationProbability = mutationProbability

    def runEvolution(self, steps: int) -> None:
        """Run step iterations of genetic algorithm"""
        population = self.initializePopulation()

        for iteration in range(steps):
            fitness, testFitness = self.calculateFitness(population)

            sortedIndices = fitness.argsort()

            population = population[sortedIndices]
            fitness = fitness[sortedIndices]
            testFitness = testFitness[sortedIndices]

            # self.mutationStandardDev = self.getNewStandardDev(fitness[0])

            print("Fitness: ", fitness)
            print("Test Fitness: ", testFitness)
            print("Vector: ", population)

            # if iteration % 10 == 0:
            #     input()

            input()

            with open("output.txt", "a") as outfile:
                pprint(
                    f"---Generation: {iteration}---",
                    stream=outfile,
                )
                pprint(population, stream=outfile)
                pprint(fitness, stream=outfile)

            # Gurantee that top two will be selected without any mutation or
            # crossover: 10 = 8 + 2
            nextGeneration = population[:1]

            for crossoverIteration in range((self.populationSize // 2)):

                # Select two parents from population
                parent_a, parent_b = self.selectTwo(
                    population[: self.populationSize - 3]
                )

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
        return np.array(
            [
                [
                    #-1.83669770e-15,
                    #2.29423303e-05,
                    #-2.04721003e-06,
                    #-1.59792834e-08,
                    #9.98214034e-10,
                    -1.93e-15,
                    2.29e-05,
                    -2.049e-06,
                    -1.6e-08,
                    9.98e-10,
                ]
            ]
            * 10
        )

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
            np.arange(self.vectorSize), self.vectorSize // 2, replace=False
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
        Mutates an individual Current Algo: Select some indices randomly and
        choose a new number from a gaussian distribution with mean as the
        number at that index
        """
        flagArray = sciStats.bernoulli.rvs(  # type:ignore
            p=self.mutationProbability, size=offspring.shape
        )

        generateGaus = lambda x: np.clip(
            np.random.normal(loc=x, scale=abs(x) / 1e5),
            -self.scalingFactor,
            self.scalingFactor,
        )
        vectorizedGenerateGaus = np.vectorize(generateGaus)
        gausArray = vectorizedGenerateGaus(offspring)

        offspring[flagArray == 1] = gausArray[flagArray == 1]
        return offspring

    def calculateFitness(self, population: Population):
        """Returns fitness array for the population"""
        # return np.mean(population ** 2, axis=1) ** 0.5

        errorList = [getErrors(indi) for indi in population.tolist()]
        return (
            np.array([error[0] for error in errorList]),
            np.array([error[1] for error in errorList]),
        )


test = GeneticAlgorithm(11, 1)
test.runEvolution(50000000)
