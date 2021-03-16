from typing import List, Callable, Tuple
from nptyping import NDArray

from pprint import pprint
import numpy as np

from scipy import stats as sciStats

from client import getErrors, submit

# Constituents
Individual = NDArray
Population = NDArray
Fitness = NDArray

# Hyper-parameters
POPULATION_SIZE = 9
MUTATION_PROBABILITY = 0.5
BETA = 0.75


class GeneticAlgorithm:
    """Definition for genetic algorithm used"""

    def __init__(
        self,
        populationSize: int,
        mutationProbability: float = 1,
        beta: float = 0.7,
    ) -> None:

        """Constructor for genetic algorithm"""
        self.vectorSize = 5
        self.rng = np.random.default_rng()
        self.scalingFactor = 10
        self.populationSize = populationSize
        self.mutationProbability = mutationProbability
        self.beta = beta

    def runEvolution(self, steps: int) -> None:
        """Run step iterations of genetic algorithm"""
        population = self.initializePopulation()

        for iteration in range(steps):
            fitness, testFitness = self.calculateFitness(population)

            sortedIndices = (abs(fitness + testFitness)).argsort()

            population = population[sortedIndices]
            fitness = fitness[sortedIndices]
            testFitness = testFitness[sortedIndices]

            print("Fitness: ", np.array([fitness[0]]))
            print("Test Fitness: ", np.array([testFitness[0]]))
            print("Vector: ", population[0])
            # print("\n", population[0])
            submit(population.tolist()[0])

            with open("generations.txt", "a") as outfile:
                pprint(
                    f"-----------GENERATION: {iteration}--------------",
                    stream=outfile,
                )
                pprint(population, stream=outfile)
                pprint(
                    f"---Train Fitness: {iteration}---",
                    stream=outfile,
                )
                pprint(fitness, stream=outfile)
                pprint(
                    f"---Test Fitness: {iteration}---",
                    stream=outfile,
                )
                pprint(testFitness, stream=outfile)

            input()

            # Gurantee that top two will be selected without any mutation or
            # crossover: 9 = 8 + 1
            nextGeneration = population[:1]
            nonMutatedNextGen = population[:1]

            for crossoverIteration in range(self.populationSize):

                # Select two parents from population
                index_a, index_b = self.selectTwo(population)

                # Cross them
                offspring_a, offspring_b = self.crossOver(
                    population[index_a],
                    population[index_b],
                    fitness[index_a],
                    fitness[index_b],
                )

                nonMutatedNextGen = np.concatenate(
                    (nonMutatedNextGen, [offspring_a, offspring_b]), axis=0
                )

                # Mutate
                offspring_a = self.mutateOffspring(offspring_a)
                offspring_b = self.mutateOffspring(offspring_b)

                # Add to next generation
                nextGeneration = np.concatenate(
                    (nextGeneration, [offspring_a, offspring_b]), axis=0
                )

            population = nextGeneration[:self.populationSize]

            with open("generations.txt", "a") as outfile:
                pprint(
                    f"---After crossover population: {iteration}---",
                    stream=outfile,
                )
                pprint(nonMutatedNextGen, stream=outfile)
                pprint(
                    f"---After mutation population: {iteration}---",
                    stream=outfile,
                )
                pprint(nextGeneration, stream=outfile)
                pprint(
                    f"---After Selection population: {iteration}---",
                    stream=outfile,
                )
                pprint(population, stream=outfile)

    def initializePopulation(self):
        """Initialize a population randomly"""
        return np.array(
            [
                [
                    0.0, 0.0, 0.0, 0.0, 0.0, -5.74726809e-16, 0.0,
                    1.52226779e-05, -1.04623995e-06, -5.47552982e-09, 3.8344835e-10
                ]
            ]
            * self.populationSize
        )

    def selectTwo(self, population):
        """Selects to random individuals from a given population"""

        indices = np.random.choice(population.shape[0], 2, replace=False)
        return indices

    def crossOver(
        self,
        parent_a: Individual,
        parent_b: Individual,
        fitness_a: float,
        fitness_b: float,
    ) -> Tuple[Individual, Individual]:
        """Crosses two parents to give a two new offsprings"""

        offspring_a = (
            ((1 + self.beta) * parent_a) + ((1 - self.beta) * parent_b)
        ) / 2

        offspring_b = (
            ((1 - self.beta) * parent_a) + ((1 + self.beta) * parent_b)
        ) / 2

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
            np.random.normal(loc=x, scale=abs(x) / 5e2),
            -self.scalingFactor,
            self.scalingFactor,
        )
        vectorizedGenerateGaus = np.vectorize(generateGaus)
        gausArray = vectorizedGenerateGaus(offspring)

        offspring[flagArray == 1] = gausArray[flagArray == 1]
        return offspring

    def calculateFitness(self, population):
        """Returns fitness array for the population"""
        # return np.mean(population ** 2, axis=1) ** 0.5

        errorList = [getErrors(indi, False) for indi in population.tolist()]
        return (
            np.array([error[0] for error in errorList]),
            np.array([error[1] for error in errorList]),
        )


test = GeneticAlgorithm(POPULATION_SIZE, MUTATION_PROBABILITY, BETA)
test.runEvolution(50000000)
