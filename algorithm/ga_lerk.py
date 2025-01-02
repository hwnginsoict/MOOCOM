import numpy as np
import random

class GA_LERK:
    def __init__(self, graph, num_vehicles, population_size, num_generations):
        self.graph = graph
        self.num_nodes = graph.num_nodes
        self.num_vehicles = num_vehicles
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = []
    
    def initialize_population(self):
        """Generate initial population using random keys."""
        for _ in range(self.population_size):
            # Random keys for vehicles and nodes
            leader_keys = np.random.uniform(200, 300, self.num_vehicles)  # Leader keys in the range [200, 300]
            node_keys = np.random.uniform(0, 1, self.num_nodes - 1)  # Node keys in [0, 1]
            individual = np.concatenate((leader_keys, node_keys))
            self.population.append(individual)

    def decode_solution(self, individual):
        """Decode random keys to a vehicle routing solution."""
        leader_keys = individual[:self.num_vehicles]
        node_keys = individual[self.num_vehicles:]
        sorted_indices = np.argsort(node_keys)

        # Assign nodes to vehicles in a round-robin manner
        solution = [[] for _ in range(self.num_vehicles)]
        vehicle_capacity = (self.num_nodes - 1) // self.num_vehicles
        remaining_nodes = list(sorted_indices + 1)  # Adjust index to match node numbers

        for i, leader_key in enumerate(sorted(leader_keys)):
            solution[i].append(int(leader_key))  # Assign leader key
            assigned_nodes = remaining_nodes[:vehicle_capacity]  # Assign up to capacity
            solution[i].extend(assigned_nodes)
            remaining_nodes = remaining_nodes[vehicle_capacity:]  # Update remaining nodes

        # Distribute any remaining nodes to vehicles
        for idx, node in enumerate(remaining_nodes):
            solution[idx % self.num_vehicles].append(node)

        return solution

    def fitness(self, solution):
        """Evaluate the fitness of a decoded solution."""
        total_distance = 0
        for route in solution:
            leader_key, *nodes = route
            distance = sum(self.graph.dist[nodes[i - 1]][nodes[i]] for i in range(1, len(nodes)))
            total_distance += distance
        return 1 / total_distance if total_distance > 0 else float('inf')

    def select_parents(self):
        """Select parents using tournament selection."""
        tournament_size = 3
        parents = []
        for _ in range(2):
            competitors = random.sample(self.population, tournament_size)
            parents.append(min(competitors, key=lambda x: self.fitness(self.decode_solution(x))))
        return parents

    def crossover(self, parent1, parent2):
        """Apply uniform crossover to generate offspring."""
        mask = np.random.randint(0, 2, len(parent1))
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutate(self, individual):
        """Apply mutation to an individual."""
        mutation_rate = 0.1
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
        return np.clip(individual, 0, None)  # Ensure values remain non-negative

    def run(self):
        """Run the GA to optimize the VRP solution."""
        self.initialize_population()
        best_solution = None
        best_fitness = float('-inf')

        for generation in range(self.num_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                # Select parents
                parent1, parent2 = self.select_parents()
                # Crossover
                offspring1 = self.crossover(parent1, parent2)
                offspring2 = self.crossover(parent2, parent1)
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                # Add offspring to new population
                new_population.extend([offspring1, offspring2])
            self.population = new_population
            
            # Evaluate the best solution
            for individual in self.population:
                decoded_solution = self.decode_solution(individual)
                fitness = self.fitness(decoded_solution)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = decoded_solution

            # print(best_solution)          
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        self.print_solution(best_solution)
        
        return best_solution

    # def repair(self, solution):

    
    def print_solution(self, solution):
        """
        Prints a solution for a vehicle routing problem.

        Args:
            solution (list of lists): Each sublist represents a route with the first element being the leader key.
        """
        for i, route in enumerate(solution):
            # Convert np.int64 elements to standard Python int for easier reading
            route = [int(node) if isinstance(node, np.int64) else node for node in route]
            leader = route[0]
            path = route[1:]
            print(f"Route {i + 1}: Leader {leader} -> Path: {path}")
