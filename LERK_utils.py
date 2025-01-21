from graph.graph import Graph
import numpy as np
from population import Individual

class Element:
    def __init__(self, leader=None, id_request=None, value=None):
        self.leader = leader
        self.id_request = id_request
        self.value = value


def create_chromosome_LERK(graph: Graph):
    # Insert some random uniform values
    # For the "leader" elements, we do vehicle_num - 1
    chromosome = [
        Element(leader=True, id_request=None, value=np.random.rand())
        for _ in range(graph.vehicle_num - 1)
    ]
    # For the requests
    chromosome.extend(
        Element(leader=False, id_request=i, value=np.random.rand())
        for i in range(graph.num_pickup_nodes)
    )
    return chromosome


def create_individual_LERK(graph: Graph):
    indi = Individual()
    indi.chromosome = create_chromosome_LERK(graph)
    return indi


def decode_chromosome(graph: Graph, chromosome):
    chromosome.sort(key=lambda x: x.value)
    
    routes = [[] for _ in range(graph.vehicle_num)]
    curr_vehicle = 0
    for element in chromosome:
        if element.leader:
            curr_vehicle += 1
        else:
            routes[curr_vehicle].append(element.id_request)
    
    id_solution = []
    for v_idx, request_list in enumerate(routes, start=1):
        route_nodes = []
        # for id_req in request_list:
        #     nid_pickup = graph.pickup_nodes[id_req].nid
        #     pickup_node = graph.nodes[nid_pickup]
        #     delivery_node = graph.nodes[pickup_node.did]
        #     route_nodes.append(pickup_node)
        #     route_nodes.append(delivery_node)
        # route_nodes.sort(key=lambda x: x.due_time)

        for id_req in request_list:
            nid_pickup = graph.pickup_nodes[id_req].nid
            # pickup_node = graph.nodes[nid_pickup]
            # delivery_node = graph.nodes[pickup_node.did]
            nid_delivery = graph.nodes[nid_pickup].did
            route_nodes.append(nid_pickup)
            route_nodes.append(nid_delivery)
        route_nodes.sort(key=lambda x: graph.nodes[x].due_time)

        # route_nids = [v_idx] + [node.nid for node in route_nodes]
        route_nids = [v_idx] + route_nodes
        id_solution.append(route_nids)

    return id_solution


def crossover_LERK(graph: Graph, indi1: Individual, indi2: Individual):
    off1 = Individual()
    off2 = Individual()
    off1.chromosome = []
    off2.chromosome = []

    # SBX crossover
    eta = 2
    u = np.random.rand()
    if u <= 0.5:
        beta = (2 * u) ** (1 / (eta + 1))
    else:
        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    for i in range(len(indi1.chromosome)):
        off1.chromosome.append(Element(leader = indi1.chromosome[i].leader, id_request = indi1.chromosome[i].id_request, value = 0.5 * ((1 + beta) * indi1.chromosome[i].value + (1 - beta) * indi2.chromosome[i].value)))
        off2.chromosome.append(Element(leader = indi2.chromosome[i].leader, id_request = indi2.chromosome[i].id_request, value = 0.5 * ((1 + beta) * indi2.chromosome[i].value + (1 - beta) * indi1.chromosome[i].value)))
    
    for i in range(len(off1.chromosome)):
        if off1.chromosome[i].value > 1:
            off1.chromosome[i].value = 1
        if off1.chromosome[i].value < 0:
            off1.chromosome[i].value = 0
        if off2.chromosome[i].value > 1:
            off2.chromosome[i].value = 1
        if off2.chromosome[i].value < 0:
            off2.chromosome[i].value = 0
    return off1, off2

def mutation_LERK(graph: Graph, indi: Individual):
    off = Individual()
    eta = 2
    off.chromosome = []
    for i in range(len(indi.chromosome)):
        u = np.random.rand()
        if u <= 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        off.chromosome.append(Element(leader = indi.chromosome[i].leader, id_request = indi.chromosome[i].id_request, value = indi.chromosome[i].value + delta))
    for i in range(len(off.chromosome)):
        if off.chromosome[i].value > 1:
            off.chromosome[i].value = 1
        if off.chromosome[i].value < 0:
            off.chromosome[i].value = 0
    return off

from utils import cost_full
from moo_algorithm.nsga_ii import run_nsga_ii

def calculate_fitness_LERK(problem, individual):
    route = decode_chromosome(problem, individual.chromosome)
    total_distance, vehicle_fairness, customer_fairness, max_time = cost_full(problem, route)
    individual.objectives = [total_distance, vehicle_fairness, customer_fairness, max_time]
    return individual.objectives

if __name__ == "__main__":
    graph = Graph(".\data\dpdptw\\200\LC1_2_1.csv")
    # chromosome = create_chromosome(graph)
    # print(chromosome)
    # print(chromosome)
    # print(decode_chromosome(graph, chromosome))
    indi_list = [create_individual_LERK(graph) for _ in range(100)]
    result = run_nsga_ii(4, graph, indi_list, 100, 100, crossover_LERK, mutation_LERK, 0.5, 0.1, calculate_fitness_LERK)
    print(result)