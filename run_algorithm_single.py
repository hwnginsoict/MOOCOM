from utils_new import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph
import time
from algorithm.fair_ga import FairGA
import json
import numpy as np
import csv

def main(number=8, type="LC2", index=1, seed=0, num=100, max_gen=100):
    # 1) Prepare data and create an initial population of individuals
    filepath = f'./data/dpdptw/{number}00/{type}_{number}_{index}.csv'
    graph = Graph(filepath)
    indi_list = [create_individual_pickup(graph) for _ in range(num)]
    np.random.seed(seed)

    start_time_fair_ga = time.time()
    fair_ga_results = FairGA(graph, indi_list, max_iteration=100, population_size=100, cross_size=30, localRate=0.3, mutateRate=0.1, elitistRate=0.1, crossRate=0.5)
    fair_ga_time = time.time() - start_time_fair_ga
    fair_ga_results.append(fair_ga_time)

    fair_ga_results = [float(i) for i in fair_ga_results]

    fair_ga_results.insert(0, f"{type}_{number}_{index}.csv")
    # Extract objectives from fair_ga_results

    return fair_ga_results

if __name__ == "__main__":
    header = ["Instance","EC", "CF", "CF", "MWT", "Time"]

    # Open the CSV file once before the loops begin
    with open("FairGA.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header

        for num in [2, 4, 8]:
            for type in ["LC1", "LC2", "LR1", "LR2", "LRC1", "LRC2"]:
                for i in range(1, 11):
                    result = main(num, type, i, 0, 100, 100)
                    # Ensure the result contains four objectives before writing
                    if result and len(result) == 6:
                        writer.writerow(result)
                    else:
                        raise Exception(f"Invalid result: {result}")
                    print(f"Done {type}_{num}_{i}")

    print("All objectives have been saved to FairGA.csv")