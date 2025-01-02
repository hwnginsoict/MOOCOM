from algorithm.ga_lerk import GA_LERK
from graph.graph import Graph
from problem import Problem

if __name__ == "__main__":
    filepath = 'F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    problem = Problem(filepath)
    ga_lerk = GA_LERK(graph, problem, 10, 100, 50)
    ga_lerk.run()