import multiprocessing
import sys
import os
import numpy as np
# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population, Individual
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


def g_te(indi: Individual, wv_j, z_star):
    return max([wv_j[i]*np.abs(indi.objectives[i] - z_star[i]) for i in range(len(wv_j))])


class MOEADPopulation(Population):
    def __init__(self, pop_size,  neighborhood_size, init_weight_vectors):
        super().__init__(pop_size)
        self.neighborhood_size = neighborhood_size
        self.external_pop = []
        self.weights = init_weight_vectors(self.pop_size)
        self.neighborhoods = self.init_neighborhood()
        self.z_star = None

    def init_neighborhood(self):
        B = np.empty([self.pop_size, self.neighborhood_size], dtype=int)
        for i in range(self.pop_size):
            wv = self.weights[i]
            euclidean_distances = np.empty([self.pop_size], dtype=float)
            for j in range(self.pop_size):
                euclidean_distances[j] = np.linalg.norm(wv - self.weights[j])
            B[i] = np.argsort(euclidean_distances)[:self.neighborhood_size]
        return B
    
    def initialize_z_star(self):
        self.z_star = [float('inf') for _ in range(len(self.indivs[0].objectives))]
        for i in range(len(self.z_star)):
            temp_i = min([indiv.objectives[i] for indiv in self.indivs])
            self.z_star[i] = min(self.z_star[i], temp_i)


    def update_z_star(self, individual: Individual):
        for i in range(len(self.z_star)):
            self.z_star[i] = min(self.z_star[i], individual.objectives[i])

    def reproduction(self, problem, crossover_operator, mutation_operator, mutation_rate):
        offspring = []
        for i in range(self.pop_size):
            parent1, parent2 = np.random.choice(self.neighborhoods[i].tolist(), 2, replace=False)
            off1, off2 = crossover_operator(problem, self.indivs[parent1], self.indivs[parent2])
            if np.random.rand() < mutation_rate:
                off1 = mutation_operator(problem,off1)
            offspring.append(off1)
        return offspring
    

    # def natural_selection(self):
    #     self.indivs, O = self.indivs[:self.pop_size], self.indivs[self.pop_size:]
    #     for i in range(self.pop_size):
    #         indi = O[i]
    #         wv = self.weights[i]
    #         value_indi = np.sum(wv * indi.objectives)
    #         for j in self.neighborhoods[i]:
    #             if value_indi < np.sum(wv * self.indivs[j].objectives):
    #                 self.indivs[j] = indi
    def natural_selection(self):
        self.indivs, O = self.indivs[:self.pop_size], self.indivs[self.pop_size:]
        for i in range(self.pop_size):
            indi = O[i]
            wv = self.weights[i]
            value_g_te = g_te(indi, wv, self.z_star)
            for j in self.neighborhoods[i]:
                if value_g_te < g_te(self.indivs[j], wv, self.z_star):
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

    def filter_external(self):
        objectives = set()
        new_external_pop = []
        for indi in self.external_pop:
            if tuple(indi.objectives) not in objectives:
                new_external_pop.append(indi)
                objectives.add(tuple(indi.objectives))
        self.external_pop = new_external_pop
    
    # def update_weights(self, problem, indivs: list):
    #     for i in range(self.pop_size):
    #         wv = self.weights[i]
    #         self.indivs[i].objectives = problem.evaluate(indivs[i].chromosome)
    #         value_indi = np.sum(wv * self.indivs[i].objectives)
    #         for j in self.neighborhoods[i]:
    #             if value_indi < np.sum(wv * self.indivs[j].objectives):
    #                 self.indivs[j] = self.indivs[i]


def run_moead(processing_number, problem, indi_list, pop_size, max_gen, neighborhood_size, 
              init_weight_vectors, crossover_operator, mutation_operator, mutation_rate, cal_fitness, ref_point):
    np.random.seed(0)
    moead_pop = MOEADPopulation(pop_size, neighborhood_size, init_weight_vectors)
    moead_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in moead_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(moead_pop.indivs, result):
        individual.objectives = fitness
    
    moead_pop.initialize_z_star()
    moead_pop.update_external(moead_pop.indivs)
    # moead_pop.update_weights(problem, moead_pop.indivs)

    # print("Generation 0: ", cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1])))

    history = {}
    Pareto_store = []   
    for indi in moead_pop.external_pop:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store

    for gen in range(max_gen):
        offspring = moead_pop.reproduction(problem, crossover_operator, mutation_operator, mutation_rate)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
            moead_pop.update_z_star(individual)
        moead_pop.update_external(offspring)
        moead_pop.filter_external()
        moead_pop.indivs.extend(offspring)
        # moead_pop.update_weights(problem, offspring)

        # print("Generation {}: ".format(gen + 1), cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1])))

        # print("Generation {}: ".format(gen + 1))
        moead_pop.natural_selection()

        Pareto_store = []   
        for indi in moead_pop.external_pop:
            Pareto_store.append(list(indi.objectives))
        history[gen+1] = Pareto_store
        print(gen, cal_hv_front(moead_pop.external_pop, ref_point)/np.prod(ref_point))

    pool.close()

    print("Last:",  cal_hv_front(moead_pop.external_pop, ref_point)/np.prod(ref_point))
    # return cal_hv_front(moead_pop.external_pop, ref_point)/np.prod(ref_point)

    # for i in moead_pop.external_pop:
    #     print(i.objectives)


    # return cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1]))
    return moead_pop.external_pop

    # list = []
    # for i in moead_pop.external_pop:
    #     list.append(i.objectives)
    # return list

    # print(history)
    # return history

# import time

# if __name__ == "__main__":
#     from util_bi_tsp import GetData, crossover, mutation, tour_cost, create_individual

#     num = 20


#     size = 50 #doi
#     data = GetData(num,size)
#     problems = data.generate_instances()

#     ref_point = np.array([35, 35]) #doi

#     hv_list = []
#     time_list = []

#     for problem in problems:
#         time_start = time.time()
#         indi_list = [create_individual(size) for _ in range(500)]
#         result = run_moead(4, problem[0], indi_list, 500, 500, 10, init_weight_vectors_2d, crossover, mutation, 
#                 0.1, tour_cost, ref_point)
#         time_end = time.time()
#         time_list.append(time_end - time_start)
#         hv_list.append(result)

#     print("AVG", sum(hv_list)/len(hv_list))
#     print("AVG TIME", sum(time_list)/len(time_list))
 

import time, json
if __name__ == "__main__":
    from util_bi_cvrp import GetData, crossover, mutation, tour_cost, create_individual

    num = 8

    size = 50 # 20, 50, 100

    if size == 20:
        ref_point = np.array([30, 8])
    elif size == 50:
        ref_point = np.array([45, 8])
    elif size == 100:
        ref_point = np.array([80, 8])
    data = GetData(num,size)
    problems = data.generate_instances()

    print(f"moead bi cvrp {size}")

    hv_list = []
    time_list = []

    print(ref_point)

    obj_json = []
    
    for problem in problems:
        start = time.time()
        indi_list = [create_individual(problem,size) for _ in range(500)]
        pareto_store = run_moead(4, problem, indi_list, 500, 500, 10, init_weight_vectors_2d, crossover, mutation, 
                0.1, tour_cost, ref_point)
        end = time.time()
        time_list.append(end - start)
        hv = cal_hv_front(pareto_store, ref_point) / np.prod(ref_point)
        hv_list.append(hv)
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