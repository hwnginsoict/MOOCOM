from algorithm.ga_lerk import GA_LERK
from graph.graph import Graph
from problem import Problem

if __name__ == "__main__":
    graph = Graph('F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv')
    problem = Problem('F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv')
    ga_lerk = GA_LERK(graph, 10, 100, 50)
    ga_lerk.run()