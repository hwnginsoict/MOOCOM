from algorithm.ga import GA
from graph.graph import Graph
from problem import Problem

if __name__ == "__main__":
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    problem = Problem(filepath)
    ga_lerk = GA(graph, problem, 10, 100, 50)
    ga_lerk.run()