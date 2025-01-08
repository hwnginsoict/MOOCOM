from algorithm.ga import GA
from graph.graph import Graph
from moo_algorithm.moead_plus import init_weight_vectors_3d, run_moead_plus
from moo_algorithm.moead_paper import run_moead
from moo_algorithm.nsga_ii import run_nsga_ii
from utils import create_individual_pickup, crossover_operator, mutation_operator, calculate_fitness
import numpy as np
if __name__ == "__main__":
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    np.random.seed(0)
    indi_list = [create_individual_pickup(graph) for _ in range(1000)]
    ngga_ii = run_nsga_ii(4, graph, indi_list, 1000, 100, crossover_operator, mutation_operator, 0.5, 0.1, calculate_fitness)
    moead_plus = run_moead_plus(4, graph, indi_list, 1000, 100, 25, init_weight_vectors_3d, crossover_operator,mutation_operator, calculate_fitness)
    moead_paper = run_moead(4, graph, indi_list, 1000, 100, 25, init_weight_vectors_3d, crossover_operator, mutation_operator, 
              0.1, calculate_fitness)
    
    print("NSGA-II: ", ngga_ii)
    print("MOEA/D+: ", moead_plus)
    print("MOEA/D: ", moead_paper)