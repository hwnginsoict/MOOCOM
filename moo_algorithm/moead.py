import multiprocessing
import sys
import os
import numpy as np
# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population, Individual
from utils import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph

def init_weight_vectors_2d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        wvs.append([i, 1 - i])
    return np.array(wvs)

def init_weight_vectors_3d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        for j in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
            if i + j <= 1:
                wvs.append([i, j, 1 - i - j])
    return np.array(wvs)

def init_weight_vectors_4d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        for j in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
            for k in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
                if i + j + k <= 1:
                    wvs.append([i, j, k, 1 - i - j - k])
    return np.array(wvs)


class MOEADPopulation(Population):
    def __init__(self, pop_size,  neighborhood_size, init_weight_vectors):
        super().__init__(pop_size)
        self.neighborhood_size = neighborhood_size
        self.external_pop = []
        self.weights = init_weight_vectors(self.pop_size)
        self.neighborhoods = self.init_neighborhood()

    def init_neighborhood(self):
        B = np.empty([self.pop_size, self.neighborhood_size], dtype=int)
        for i in range(self.pop_size):
            wv = self.weights[i]
            euclidean_distances = np.empty([self.pop_size], dtype=float)
            for j in range(self.pop_size):
                euclidean_distances[j] = np.linalg.norm(wv - self.weights[j])
            B[i] = np.argsort(euclidean_distances)[:self.neighborhood_size]
        return B

    def reproduction(self, problem, crossover_operator, mutation_operator, mutation_rate):
        offspring = []
        for i in range(self.pop_size):
            parent1, parent2 = np.random.choice(self.neighborhoods[i].tolist(), 2, replace=False)
            off1, off2 = crossover_operator(problem, self.indivs[parent1], self.indivs[parent2])
            if np.random.rand() < mutation_rate:
                off1 = mutation_operator(problem, off1)
            offspring.append(off1)
        return offspring
    
    # def mutation(self, problem, mutation_operator):
    #     for i in range(self.pop_size):
    #         if np.random.rand() < 0.1:
    #             self.indivs[i] = mutation_operator(problem, self.indivs[i])
    

    def natural_selection(self):
        self.indivs, O = self.indivs[:self.pop_size], self.indivs[self.pop_size:]
        for i in range(self.pop_size):
            indi = O[i]
            wv = self.weights[i]
            value_indi = np.sum(wv * indi.objectives)
            for j in self.neighborhoods[i]:
                if value_indi < np.sum(wv * self.indivs[j].objectives):
                    self.indivs[j] = indi

    def update_external(self, indivs: list):
        for indi in indivs:
            old_size = len(self.external_pop)
            self.external_pop = [other for other in self.external_pop
                                 if not indi.dominates(other)]
            if old_size > len(self.external_pop):
                self.external_pop.append(indi)
                continue
            for other in self.external_pop:
                if other.dominates(indi):
                    break
            else:
                self.external_pop.append(indi)
    
    # def update_weights(self, problem, indivs: list):
    #     for i in range(self.pop_size):
    #         wv = self.weights[i]
    #         self.indivs[i].objectives = problem.evaluate(indivs[i].chromosome)
    #         value_indi = np.sum(wv * self.indivs[i].objectives)
    #         for j in self.neighborhoods[i]:
    #             if value_indi < np.sum(wv * self.indivs[j].objectives):
    #                 self.indivs[j] = self.indivs[i]


def run_moead(processing_number, problem, indi_list, pop_size, max_gen, neighborhood_size, 
              init_weight_vectors, crossover_operator, cal_fitness):
    moead_pop = MOEADPopulation(pop_size, neighborhood_size, init_weight_vectors)
    moead_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in moead_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(moead_pop.indivs, result):
        individual.objectives = fitness
    
    moead_pop.update_external(moead_pop.indivs)
    # moead_pop.update_weights(problem, moead_pop.indivs)
    print("Generation 0: ", cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1])))

    for gen in range(max_gen):
        offspring = moead_pop.reproduction(problem, crossover_operator)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        moead_pop.update_external(offspring)
        moead_pop.indivs.extend(offspring)
        # moead_pop.update_weights(problem, offspring)
        print("Generation {}: ".format(gen + 1), cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1])))
        moead_pop.natural_selection()
    pool.close()

    # for i in moead_pop.external_pop:
    #     print(i.objectives)
    return moead_pop.external_pop


import time, json
if __name__ == "__main__":
    from util_bi_tsp import GetData, crossover, mutation, tour_cost, create_individual

    num = 1
    size = 50
    data = GetData(num,size)
    problems = data.generate_instances()

    ref_point = np.array([35, 35])

    hv_list = []
    time_list = []

    print(ref_point)

    obj_json = []
    
    for problem in problems:
        start = time.time()
        indi_list = [create_individual(size) for _ in range(100)]
        pareto_store = run_moead(4, problem, indi_list, 100, 50, 3, init_weight_vectors_3d, crossover, tour_cost)
        end = time.time()
        time_list.append(end - start)

        temp = []

        for indi in pareto_store:
            temp.append(indi.objectives)
        obj_json.append(temp)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj

    # Prepare the data
    serializable_obj_json = convert_to_serializable(obj_json)

    # Save to JSON
    with open("pareto_objectives.json", "w") as f:
        json.dump(serializable_obj_json, f, indent=2)

    print("HV LIST", hv_list)
    print("AVG HV: ", sum(hv_list)/len(hv_list))
    print("AVG TIME", sum(time_list)/len(time_list))
 

