import multiprocessing
import sys
import os
import numpy as np
import time
# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population, Individual
from graph.graph import Graph
from pymoo.indicators.hv import HV 

class NSGAIIPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []
    

    def fast_nondominated_sort_crowding_distance(self, indi_list):
        ParetoFront = [[]]
        for individual in indi_list:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in indi_list:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                ParetoFront[0].append(individual)
        i = 0
        while len(ParetoFront[i]) > 0:
            temp = []
            for individual in ParetoFront[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            ParetoFront.append(temp)
        for front in ParetoFront:
            self.calculate_crowding_distance(front)
        return ParetoFront

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    # Crowding Operator
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def natural_selection(self):
        self.ParetoFront = self.fast_nondominated_sort_crowding_distance(self.indivs)
        new_indivs = []
        new_fronts = []
        front_num = 0
        while len(new_indivs) + len(self.ParetoFront[front_num]) <= self.pop_size:
            new_indivs.extend(self.ParetoFront[front_num])
            new_fronts.append(self.ParetoFront[front_num])
            if len(new_indivs) == self.pop_size:
                break
            front_num += 1
        self.calculate_crowding_distance(self.ParetoFront[front_num])
        self.ParetoFront[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
        number_remain = self.pop_size - len(new_indivs)
        new_indivs.extend(self.ParetoFront[front_num][0:number_remain])
        new_fronts.append(self.ParetoFront[front_num][0:number_remain])
        self.ParetoFront = new_fronts
        self.indivs = new_indivs

def filter_external(pareto):
    objectives = set()
    new_external_pop = []
    for indi in pareto:
        if tuple(indi.objectives) not in objectives:
            new_external_pop.append(indi)
            objectives.add(tuple(indi.objectives))
    return new_external_pop

def run_nsga_ii(processing_number, problem, indi_list, pop_size, max_gen, crossover_operator, mutation_operator, 
                crossover_rate, mutation_rate, cal_fitness, ref_point):
    history = {}
    nsga_ii_pop = NSGAIIPopulation(pop_size)
    nsga_ii_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in nsga_ii_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(nsga_ii_pop.indivs, result):
        individual.objectives = fitness

    history_hv = []
    nsga_ii_pop.natural_selection()
    # history_hv.append(cal_hv_front(nsga_ii_pop.ParetoFront[0], np.array([1, 1, 1])))

    Pareto_store = []
    for indi in nsga_ii_pop.ParetoFront[0]:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store


    for gen in range(max_gen):
        # print("Bắt đầu gen")
        time_start = time.time()
        offspring = nsga_ii_pop.gen_offspring(problem, crossover_operator, mutation_operator, crossover_rate, mutation_rate)
        # print("Done gen off")
        # print("Tạo cá thể xong")
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        # print("Tính fitness xong")
        nsga_ii_pop.indivs.extend(offspring)
        nsga_ii_pop.natural_selection()
        # history_hv.append(cal_hv_front(nsga_ii_pop.ParetoFront[0], np.array([1, 1, 1])))

       
        Pareto_store = []
        for indi in nsga_ii_pop.ParetoFront[0]:
            Pareto_store.append(list(indi.objectives))
        history[gen+1] = Pareto_store
        # print("Lưu cá thể")

        print(gen, cal_hv_front(nsga_ii_pop.ParetoFront[0], ref_point) / np.prod(ref_point ))
    # print(Pareto_store)
    print(cal_hv_front(nsga_ii_pop.ParetoFront[0], ref_point=ref_point) / np.prod(ref_point))
    pool.close()
    return nsga_ii_pop.ParetoFront[0]
    
import json
if __name__ == "__main__":
    from util_bi_tsp import GetData, crossover, mutation, tour_cost, create_individual


    num = 20

    size = 50
    ref_point = np.array([35, 35])

    print("bi tsp 50")
    print(ref_point)

    data = GetData(num,size)
    problems = data.generate_instances()

    hv_list = []

    print(ref_point)

    obj_json = []

    for problem in problems:
        indi_list = [create_individual(size) for _ in range(300)]
        Pareto_store = run_nsga_ii(4, problem[0], indi_list, 300, 300, crossover, mutation, 0.5, 0.1, tour_cost, ref_point)
        hv  = cal_hv_front(Pareto_store, ref_point) / np.prod(ref_point)
        hv_list.append(hv)
        print(hv)

        temp = []
        for indi in Pareto_store:
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



# if __name__ == "__main__":
#     from util_bi_tsp import GetData, crossover, mutation, tour_cost, create_individual

#     num = 20
#     size = 50
#     data = GetData(num,size)
#     problems = data.generate_instances()

#     hv_list = []

#     ref_point = np.array([35, 35])

#     for problem in problems:
#         indi_list = [create_individual(size) for _ in range(300)]
#         Pareto_store = run_nsga_ii(4, problem[0], indi_list, 300, 300, crossover, mutation, 0.5, 0.1, tour_cost)
#         hv  = cal_hv_front(Pareto_store, ref_point) / np.prod(ref_point)
#         hv_list.append(hv)
#         print(hv)
#     print("HV LIST", hv_list)
#     print("AVG HV: ", sum(hv_list)/len(hv_list))


