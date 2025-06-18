import random
import numpy as np
from population import Individual

class GetData():
    def __init__(self, n_instance: int, n_customers: int):
        self.n_instance = n_instance
        self.n_customers = n_customers

    def generate_instances(self):
        np.random.seed(2025)
        instance_data = []

        for _ in range(self.n_instance):
            # Depot + customers
            coords = np.random.rand(self.n_customers + 1, 2)  # (x, y) positions in [0, 1]^2
            demands = np.random.randint(1, 10, size=self.n_customers+1)
            demands[0]  = 0  # Depot has no demand

            # calculate distance matrix
            distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)

            # Set vehicle capacity based on number of customers
            if 20 <= self.n_customers < 40:
                capacity = 30
            elif 40 <= self.n_customers < 70:
                capacity = 40
            elif 70 <= self.n_customers <= 100:
                capacity = 50
            else:
                raise ValueError("Number of customers must be between 20 and 100.")

            instance_data.append((coords, demands, distance_matrix, capacity))

        return instance_data


def compute_route_length(route: np.ndarray, distance_matrix: np.ndarray) -> float:
    if len(route) <= 1:
        return 0.0
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))


def tour_cost(instance, ind):

    routes = ind.chromosome
    distance_matrix = instance[2]
    
    total_distance = 0.0
    longest_route = 0.0
    for route in routes:
        d = compute_route_length(route, distance_matrix)
        total_distance += d
        longest_route = max(longest_route, d)
    return total_distance, longest_route 


def create_individual(instance, n_cities):
    n_vehicles = np.random.randint(6, 15)
    vehicle_capacity = instance[3]
    demands = instance[1]

    cities = list(range(1, n_cities))  # exclude depot 0
    random.shuffle(cities)

    routes = [[] for _ in range(n_vehicles)]
    route_loads = [0] * n_vehicles

    for city in cities:
        demand = demands[city]
        assigned = False
        for i in range(n_vehicles):
            if route_loads[i] + demand <= vehicle_capacity:
                routes[i].append(city)
                route_loads[i] += demand
                assigned = True
                break
        # If no feasible vehicle, assign randomly (fallback)
        if not assigned:
            idx = random.choice(range(n_vehicles))
            routes[idx].append(city)
            route_loads[idx] += demand  # May violate capacity

    return Individual(routes)

def flatten_routes(routes):
    return [city for route in routes for city in route]

def unflatten_routes_with_capacity(flat, n_vehicles, instance):
    vehicle_capacity = instance[3]
    demands = instance[1]

    routes = [[] for _ in range(n_vehicles)]
    route_loads = [0] * n_vehicles

    for city in flat:
        demand = demands[city]
        assigned = False
        for i in range(n_vehicles):
            if route_loads[i] + demand <= vehicle_capacity:
                routes[i].append(city)
                route_loads[i] += demand
                assigned = True
                break
        if not assigned:
            # Fallback: assign to the vehicle with minimum overload
            overloads = [
                (i, route_loads[i] + demand - vehicle_capacity)
                for i in range(n_vehicles)
            ]
            overloads.sort(key=lambda x: x[1])
            best = overloads[0][0]
            routes[best].append(city)
            route_loads[best] += demand
    return routes

def crossover(instance, p1, p2):
    parent1 = flatten_routes(p1.chromosome)
    parent2 = flatten_routes(p2.chromosome)
    size = len(parent1)

    cut1, cut2 = sorted(np.random.choice(range(size), 2, replace=False))

    child1 = [-1] * size
    child2 = [-1] * size

    child1[cut1:cut2+1] = parent1[cut1:cut2+1]
    child2[cut1:cut2+1] = parent2[cut1:cut2+1]

    mapping_p1 = parent1[cut1:cut2+1]
    mapping_p2 = parent2[cut1:cut2+1]

    for i in range(size):
        if i < cut1 or i > cut2:
            val = parent2[i]
            while val in mapping_p1:
                idx = mapping_p1.index(val)
                val = mapping_p2[idx]
            child1[i] = val

            val = parent1[i]
            while val in mapping_p2:
                idx = mapping_p2.index(val)
                val = mapping_p1[idx]
            child2[i] = val

    n_vehicles = len(p1.chromosome)
    routes1 = unflatten_routes_with_capacity(child1, n_vehicles, instance)
    routes2 = unflatten_routes_with_capacity(child2, n_vehicles, instance)

    return Individual(routes1), Individual(routes2)

def mutation(instance, indi):
    vehicle_capacity = instance[3]
    demands = instance[1]

    routes = indi.chromosome
    new_routes = [list(route) for route in routes]

    # Chọn 2 route khác nhau để swap
    r1, r2 = random.sample(range(len(new_routes)), 2)
    if new_routes[r1] and new_routes[r2]:
        i = random.randrange(len(new_routes[r1]))
        j = random.randrange(len(new_routes[r2]))
        # Swap
        new_routes[r1][i], new_routes[r2][j] = new_routes[r2][j], new_routes[r1][i]

    # Validate capacity after mutation
    flat = flatten_routes(new_routes)
    fixed_routes = unflatten_routes_with_capacity(flat, len(routes), instance)
    return Individual(fixed_routes)


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