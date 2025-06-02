import random
import numpy as np
from population import Individual

class GetData():
    def __init__(self,n_instance,n_cities):
        self.n_instance = n_instance
        self.n_cities = n_cities

    def generate_instances(self):
        np.random.seed(2025)
        instance_data = []
        for _ in range(self.n_instance):
            coordinates_1 = np.random.rand(self.n_cities, 2)
            coordinates_2 = np.random.rand(self.n_cities, 2)
            coordinates = np.concatenate((coordinates_1, coordinates_2), axis=1)
            distances_1 = np.linalg.norm(coordinates_1[:, np.newaxis] - coordinates_1, axis=2)
            distances_2 = np.linalg.norm(coordinates_2[:, np.newaxis] - coordinates_2, axis=2)
            instance_data.append((coordinates,distances_1, distances_2))
        return instance_data

def tour_cost(instance, route):

    solution = route.chromosome
    problem_size = len(solution)

    cost_1 = 0
    cost_2 = 0

    # print(instance)
    
    for j in range(problem_size - 1):
        node1, node2 = int(solution[j]), int(solution[j + 1])
        
        # print(node1, node2)
        # print(instance)
        # print(instance[node1])
        # print(instance[node2])

        coord_1_node1, coord_2_node1 = instance[node1][:2], instance[node1][2:]
        coord_1_node2, coord_2_node2 = instance[node2][:2], instance[node2][2:]

        cost_1 += np.linalg.norm(coord_1_node1 - coord_1_node2)
        cost_2 += np.linalg.norm(coord_2_node1 - coord_2_node2)
    
    node_first, node_last = int(solution[0]), int(solution[-1])
    
    coord_1_first, coord_2_first = instance[node_first][:2], instance[node_first][2:]
    coord_1_last, coord_2_last = instance[node_last][:2], instance[node_last][2:]

    cost_1 += np.linalg.norm(coord_1_last - coord_1_first)
    cost_2 += np.linalg.norm(coord_2_last - coord_2_first)

    return cost_1, cost_2  

def create_individual(n_cities):

    individual = list(range(n_cities))
    random.shuffle(individual)
    indi = Individual(individual)
    return indi


def crossover(instance, p1, p2):
    """
    PMX crossover cho 2 mảng permutation parent1, parent2 có cùng độ dài.
    Trả về 2 mảng con (child1, child2).
    """
    parent1 = p1.chromosome
    parent2 = p2.chromosome

    # print(parent1)
    # print(parent2)

    size = len(parent1)
    # Chọn 2 điểm cắt (cut1, cut2)
    cut1, cut2 = sorted(np.random.choice(range(size), 2, replace=False))
    
    # Khởi tạo con
    child1 = [-1] * size
    child2 = [-1] * size

    # Copy đoạn cắt từ parent
    child1[cut1:cut2+1] = parent1[cut1:cut2+1]
    child2[cut1:cut2+1] = parent2[cut1:cut2+1]

    # Đánh dấu đoạn đã copy để tạo mapping
    mapping_p1 = list(parent1[cut1:cut2+1])
    mapping_p2 = list(parent2[cut1:cut2+1])

    # Xử lý các phần tử ngoài đoạn cắt cho child1
    for i in range(size):
        if i < cut1 or i > cut2:
            val = parent2[i]
            # Nếu val đã nằm trong child1, ta thay thế dựa trên mapping
            while val in mapping_p1:
                idx = mapping_p1.index(val)  # Lỗi vì mapping_p1 là ndarray
                # idx = np.where(mapping_p1 == val)[0][0]
                val = mapping_p2[idx]
            child1[i] = val

    # Tương tự cho child2
    for i in range(size):
        if i < cut1 or i > cut2:
            val = parent1[i]
            while val in mapping_p2:
                idx = mapping_p2.index(val)
                # idx = np.where(mapping_p2 == val)[0][0]
                val = mapping_p1[idx]
            child2[i] = val

    c1 = Individual(np.array(child1))
    c2 = Individual(np.array(child2))

    # c1 = lin_kernighan(instance, c1)
    # c2 = lin_kernighan(instance, c2)

    return c1, c2

def mutation(indi):
    individual = indi.chromosome
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return Individual(individual)


def two_opt(instance, route):
    """
    Basic 2-opt local search to improve the route
    """
    best = list(route)
    # print(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue  # skip adjacent
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if sum(tour_cost(instance, Individual(new_route))) < sum(tour_cost(instance, Individual(best))):
                    best = new_route
                    improved = True
        route = best
    return best



def lin_kernighan(instance, route, max_iter=50):
    """
    Simplified Lin-Kernighan local search:
    - Applies 2-opt and further improvements iteratively
    - Mimics variable-depth search
    """

    current_route = route.chromosome[:]
    best_distance = tour_cost(instance, Individual(current_route))
    
    for _ in range(max_iter):
        improved = False
        new_route = two_opt(instance, current_route)
        new_distance = tour_cost(instance, Individual(new_route))
        if sum(new_distance) < sum(best_distance):
            current_route = new_route
            best_distance = new_distance
            improved = True
        if not improved:
            break

    return Individual(current_route)