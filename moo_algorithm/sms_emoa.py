import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population
from utils import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph

class SMSEMOAPopulation(Population):
    def __init__(self, pop_size, reference_point):
        super().__init__(pop_size)
        self.external_pop = []
        self.reference_point = np.array(reference_point)
        self.objectives_tuple = set()

    def hypervolume_contribution(self, individuals):
        contributions = []
        for i in range(len(individuals)):
            remaining = individuals[:i] + individuals[i+1:]
            hv_total = cal_hv_front(individuals, self.reference_point)
            hv_without = cal_hv_front(remaining, self.reference_point)
            contributions.append(hv_total - hv_without)
        return contributions

    def natural_selection(self): #remove only 1 individual
        if len(self.indivs) <= self.pop_size:
            return
        hv_contributions = self.hypervolume_contribution(self.indivs)
        min_index = np.argmin(hv_contributions)
        del self.indivs[min_index]

    def update_external(self):
        non_dominated = []
        for indi in self.indivs:
            if not any(other.dominates(indi) for other in self.external_pop):
                non_dominated.append(indi)
        self.external_pop = non_dominated

def run_sms_emoa(problem, pop_size, max_gen, indi_list, reference_point, crossover_operator, mutation_operator, cal_fitness):
    np.random.seed(0)
    population = SMSEMOAPopulation(pop_size, reference_point)
    population.indivs = indi_list

    # Evaluate initial fitness
    for indi in population.indivs:
        indi.objectives = cal_fitness(problem, indi)

    population.update_external()

    for gen in range(max_gen):
        offspring = []
        for _ in range(pop_size):
            parent1, parent2 = np.random.choice(pop_size, 2, replace=False)
            child1, _ = crossover_operator(problem, population.indivs[parent1], population.indivs[parent2])
            if np.random.rand() < 0.15:
                child1 = mutation_operator(problem, child1)
            child1.objectives = cal_fitness(problem, child1)
            offspring.append(child1)

        population.indivs.extend(offspring)
        population.update_external()
        population.natural_selection()

        print(f"Generation {gen + 1}: Hypervolume = {cal_hv_front(population.external_pop, reference_point)}")
        for i in population.external_pop:
            print(cal_fitness(problem, i))

    return population.external_pop

if __name__ == "__main__":
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    indi_list = [create_individual_pickup(graph) for _ in range(100)]
    run_sms_emoa(graph, 100, 100, indi_list, np.array([1, 1, 1]), crossover_operator, mutation_operator, calculate_fitness)
