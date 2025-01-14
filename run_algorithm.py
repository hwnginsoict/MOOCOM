from utils_new import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph
import time
from moo_algorithm.nsga_ii import run_nsga_ii
from moo_algorithm.pfg_moea import run_pfgmoea
from moo_algorithm.moead_plus import run_moead_plus, init_weight_vectors_3d_plus
from moo_algorithm.moead_paper import run_moead
from algorithm.fair_ga import FairGA
import json
import numpy as np

def main(number = 8, type = "LC2", index = 1, seed = 0, num = 100, max_gen = 100):
    # 1) Prepare data and create an initial population of individuals

    # filepath = f'.\\data\\dpdptw\\{number}00\\{type}_{number}_{index}.csv'
    filepath = f'./data/dpdptw/{number}00/{type}_{number}_{index}.csv'

    #graph.num_vehicle = number*5

    graph = Graph(filepath)
    indi_list = [create_individual_pickup(graph) for _ in range(num)]

    np.random.seed(seed)

    # 3) Run PFG-EMOA
    start_time_pfg = time.time()
    pfg_results = run_pfgmoea(4, graph, indi_list, num, max_gen, 5, 0.01, crossover_operator, mutation_operator, 0.9, 0.1, calculate_fitness)
    pfg_time = time.time() - start_time_pfg
    pfg_results["time"] = pfg_time

    # 2) Run NSGA-II
    start_time_nsga2 = time.time()
    nsga2_results = run_nsga_ii(4, graph, indi_list, num, max_gen, crossover_operator, mutation_operator, 0.5, 0.1, calculate_fitness)
    nsga2_time = time.time() - start_time_nsga2
    nsga2_results["time"] = nsga2_time

    start_time_moead_plus = time.time()
    moead_plus_results = run_moead_plus(4, graph, indi_list, num, max_gen, 10, init_weight_vectors_3d_plus, crossover_operator, mutation_operator, calculate_fitness)
    moead_plus_time = time.time() - start_time_moead_plus
    moead_plus_results["time"] = moead_plus_time

    start_time_moead_paper = time.time()
    moead_paper_results = run_moead(4, graph, indi_list, num, max_gen, 10, init_weight_vectors_3d_plus, crossover_operator, mutation_operator, 0.1 , calculate_fitness)
    moead_paper_time = time.time() - start_time_moead_paper
    moead_paper_results["time"] = moead_paper_time


    
    # 4) Combine and store both results in the same JSON
    final_results = {
        "NSGA-II": nsga2_results,
        "PFG-EMOA": pfg_results,
        "MOEA/D+": moead_plus_results,
        "MOEA/D": moead_paper_results

    }

    name_store = f"Result/{type}_{number}_{index}_{seed}.json"
    with open(name_store, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results stored in {name_store}")

if __name__ == "__main__":
    
    for num in [2, 4, 8]:
        # for type in ["LC1", "LC2", "LR1", "LR2", "LRC1", "LRC2"]:
        for type in ["LRC1", "LRC2"]:
            for i in range(1, 11):
                main(num, type, i, 0, 100, 150)
                print(f"Done {num}_{type}_{i}")