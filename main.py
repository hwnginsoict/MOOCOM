import json
import os
import numpy as np

def append_results_to_json(filename, algo_name, solutions, tolerance=1e-6):
    """
    Append results to a JSON file under the key `algo_name`, while removing 
    duplicates (up to the given tolerance). If the file doesn't exist, it will be created.

    :param filename: The name of the JSON file to write or append to.
    :param algo_name: A string representing the name of the algorithm 
                      (e.g. "NSGA_II", "MOEAD_PLUS", etc.).
    :param solutions: A list of NumPy arrays, each array typically shape (3,) 
                      (objectives).
    :param tolerance: Tolerance for comparing two solutions as duplicates.
    """
    
    # 1) Read existing data if the file exists, otherwise start with an empty dict
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # 2) Make sure there is an empty list for the algorithm if not present
    if algo_name not in data:
        data[algo_name] = []

    # 3) Deduplicate new solutions with respect to what’s already in the JSON
    #    Convert existing solutions (Python lists) to NumPy arrays for comparison
    existing_solutions = [np.array(sol) for sol in data[algo_name]]

    unique_new_solutions = []
    for sol in solutions:
        if not any(np.allclose(sol, es, atol=tolerance) for es in existing_solutions):
            unique_new_solutions.append(sol)

    # 4) Convert newly filtered solutions to lists for JSON-serialization
    unique_new_solutions_as_lists = [sol for sol in unique_new_solutions]

    # Append to the existing solutions
    data[algo_name].extend(unique_new_solutions_as_lists)

    # 5) Optionally, deduplicate the entire set under algo_name again, just in case
    data[algo_name] = _deduplicate_solution_list(data[algo_name], tolerance)

    # 6) Write the updated dictionary back to the JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def _deduplicate_solution_list(solution_list, tolerance=1e-6):
    """
    Given a list of solutions (as Python lists), deduplicate them using 
    np.allclose with the specified tolerance.
    """
    unique_solutions = []
    for sol in solution_list:
        arr_sol = np.array(sol)
        if not any(np.allclose(arr_sol, np.array(us), atol=tolerance) for us in unique_solutions):
            unique_solutions.append(sol)
    return unique_solutions



if __name__ == "__main__":
    from algorithm.ga import GA
    from graph.graph import Graph
    from moo_algorithm.moead_plus import init_weight_vectors_3d, run_moead_plus
    from moo_algorithm.moead_paper import run_moead
    from moo_algorithm.nsga_ii import run_nsga_ii
    from moo_algorithm.pfg_moea import run_pfgmoea
    from moo_algorithm.sms_emoa import run_sms_emoa
    from utils import create_individual_pickup, crossover_operator, mutation_operator, calculate_fitness
    
    # Adjust as needed
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    np.random.seed(0)
    
    # Create a large population of individuals

    pop_size = 100
    indi_list = [create_individual_pickup(graph) for _ in range(pop_size)]
    
    # -------------------------
    # 1) Run NSGA-II
    # -------------------------
    nsga_ii_results = run_nsga_ii(
        4,
        graph,
        indi_list,
        pop_size,  # e.g. population size
        100,   # e.g. number of generations
        crossover_operator,
        mutation_operator,
        0.5,   # crossover prob
        0.1,   # mutation prob
        calculate_fitness
    )
    # Immediately write/append NSGA-II results
    append_results_to_json("moo_results.json", "NSGA_II", nsga_ii_results)
    print("NSGA-II results appended to moo_results.json")
    

    # -------------------------
    # 2) Run MOEA/D+
    # -------------------------

    # moead_plus_results = run_moead_plus(
    #     4,
    #     graph,
    #     indi_list,
    #     pop_size,
    #     100,
    #     25,  # neighborhood size
    #     init_weight_vectors_3d,
    #     crossover_operator,
    #     mutation_operator,
    #     calculate_fitness
    # )
    # # Immediately write/append MOEA/D+ results
    # append_results_to_json("moo_results.json", "MOEAD_PLUS", moead_plus_results)
    # print("MOEA/D+ results appended to moo_results.json")


    # -------------------------
    # 3) Run MOEA/D
    # -------------------------
    
    # moead_paper_results = run_moead(
    #     4,
    #     graph,
    #     indi_list,
    #     pop_size,
    #     100,
    #     25,
    #     init_weight_vectors_3d,
    #     crossover_operator,
    #     mutation_operator,
    #     0.1,  # F or alpha param
    #     calculate_fitness
    # )
    # append_results_to_json("moo_results.json", "MOEAD_PAPER", moead_paper_results)
    # print("MOEA/D results appended to moo_results.json")


    # -------------------------
    # 4) Run PFG-MOEA
    # -------------------------
    pfg_moea_results = run_pfgmoea(
        4,
        graph,
        indi_list,
        pop_size,
        100,
        100,   # number of generations or other param
        0.01,  # some param
        crossover_operator,
        mutation_operator,
        0.9,   # crossover prob
        0.1,   # mutation prob
        calculate_fitness
    )
    append_results_to_json("moo_results.json", "PFG_MOEA", pfg_moea_results)
    print("PFG-MOEA results appended to moo_results.json")


    # -------------------------
    # 5) Run SMS-EMOA
    # -------------------------
    # msm_moead_results = run_sms_emoa(
    #     graph,
    #     100, 
    #     100, 
    #     np.array([1, 1, 1]),
    #     crossover_operator,
    #     mutation_operator,
    #     calculate_fitness
    # )
    # append_results_to_json("moo_results.json", "SMS_EMOA", msm_moead_results)
    # print("SMS-EMOA results appended to moo_results.json")

    # Now, after each algorithm has appended its results, you can optionally read and 
    # inspect "moo_results.json" to see all solutions (deduplicated) for each algorithm.
    print("All results have been saved to moo_results.json.")
