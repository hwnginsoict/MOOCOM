from utils_new import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph
import time
from moo_algorithm.nsga_ii import run_nsga_ii
from moo_algorithm.pfg_moea import run_pfgmoea  # <-- make sure this import is correct
import json
import numpy as np

def main(number = 8, type = "LC2", index = 1, seed = 0):
    # 1) Prepare data and create an initial population of individuals

    # filepath = f'.\\data\\dpdptw\\{number}00\\{type}_{number}_{index}.csv'
    filepath = f'.\\data\\dpdptw\\800\\LC2_8_1.csv'

    graph = Graph(filepath)
    indi_list = [create_individual_pickup(graph) for _ in range(100)]

    np.random.seed(0)   

    # 3) Run PFG-EMOA
    start_time_pfg = time.time()
    pfg_results = run_pfgmoea(4, graph, indi_list, 100, 100, 100, 0.01, crossover_operator, mutation_operator, 0.9, 0.1, calculate_fitness)
    pfg_time = time.time() - start_time_pfg
    pfg_results["time"] = pfg_time

    # 2) Run NSGA-II
    start_time_nsga2 = time.time()
    nsga2_results = run_nsga_ii(4, graph, indi_list, 100, 100, crossover_operator, mutation_operator, 0.5, 0.1, calculate_fitness)
    nsga2_time = time.time() - start_time_nsga2
    nsga2_results["time"] = nsga2_time

    # 4) Combine and store both results in the same JSON
    final_results = {
        "NSGA-II": nsga2_results,
        "PFG-EMOA": pfg_results
    }

    name_store = f"Result/{type}_{number}_{index}_{seed}.json"
    with open(name_store, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results stored in {name_store}")

if __name__ == "__main__":
    main()