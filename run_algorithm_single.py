from utils_new import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph
import time
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

    start_time_fair_ga = time.time()
    fair_ga_results = FairGA(4, graph, indi_list, num, max_gen, 100, 0.01, crossover_operator, mutation_operator, 0.9, 0.1, calculate_fitness)
    fair_ga_time = time.time() - start_time_fair_ga
    fair_ga_results["time"] = fair_ga_time
    
    # 4) Combine and store both results in the same JSON
    final_results = {
        "Fair-GA": fair_ga_results
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
                main(num, type, i, 0, 100, 100)
                print(f"Done {num}_{type}_{i}")