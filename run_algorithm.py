from utils_new import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph
import time
from moo_algorithm.nsga_ii import run_nsga_ii
import json





if __name__ == "__main__":
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    indi_list = [create_individual_pickup(graph) for _ in range(100)]
    time_start = time.time()
    Pareto_store = run_nsga_ii(4, graph, indi_list, 100, 100, crossover_operator, mutation_operator, 0.5, 0.1, calculate_fitness)
    time_end = time.time()
    name_store = "Result/example.json"
    Pareto_store["time"] = time_end - time_start
    with open(name_store, "w") as f:
        json.dump(Pareto_store, f)
