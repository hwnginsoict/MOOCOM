import numpy as np
import random

class GA_LERK:
    def __init__(self, graph, problem, num_vehicles, population_size, num_generations):
        self.graph = graph
        self.problem = problem
        self.num_nodes = graph.num_nodes
        self.num_vehicles = num_vehicles
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = []
        
        # If you have a known capacity for each vehicle, set it here:
        self.vehicle_capacity = 999999  # or any suitable number

        # Pre-compute all (pickup, delivery) requests from the problem
        # (Assuming each node i has problem[i].pid, problem[i].did)
        # pickup node if pid=0 and did !=0, delivery node if pid!=0 and did=0
        self.requests = self.get_requests()

    def get_requests(self):
        """
        Identify all pickup and delivery nodes from 'problem'.
        Returns a dictionary mapping pickup_node -> delivery_node.
        
        For example, if node p has pid=0, did=d and node d has pid=p, did=0,
        we store requests[p] = d.
        """
        requests = {}
        for node in range(self.num_nodes):
            pid = self.problem.graph.nodes[node].pid
            did = self.problem.graph.nodes[node].did

            # Identify a pickup node: pid=0, did != 0
            if pid == 0 and did != 0:
                # 'did' is the matching delivery node index
                pickup = node
                delivery = did
                requests[pickup] = delivery
        return requests

    def initialize_population(self):
        """Generate initial population using random keys."""
        for _ in range(self.population_size):
            # Leader keys in the range [num_nodes, num_nodes + num_vehicles]
            leader_keys = np.random.uniform(
                self.num_nodes, 
                self.num_nodes + self.num_vehicles, 
                self.num_vehicles
            )
            # Node keys in [0, 1]
            node_keys = np.random.uniform(0, 1, self.num_nodes - 1)
            
            individual = np.concatenate((leader_keys, node_keys))
            self.population.append(individual)

    def decode_solution(self, individual):
        """Decode random keys to a vehicle routing solution."""
        leader_keys = individual[:self.num_vehicles]
        node_keys = individual[self.num_vehicles:]
        sorted_indices = np.argsort(node_keys)

        # Assign nodes to vehicles in a round-robin manner
        solution = [[] for _ in range(self.num_vehicles)]
        
        # Each route can hold (num_nodes - 1) // num_vehicles in the naive approach
        naive_capacity = (self.num_nodes - 1) // self.num_vehicles
        remaining_nodes = list(sorted_indices + 1)  # Shift index by 1 (since we skip node 0)

        # 1) Assign "leader" keys (just storing them as the first element).
        #    Then assign up to naive_capacity to each route.
        sorted_leader_indices = np.argsort(leader_keys)
        for i, leader_key in enumerate(sorted_leader_indices):
            # This route starts with the leader "id" (as an integer).
            solution[i].append(int(leader_key))  
            assigned_nodes = remaining_nodes[:naive_capacity]
            solution[i].extend(assigned_nodes)
            remaining_nodes = remaining_nodes[naive_capacity:]

        # 2) Distribute leftover nodes if there are any
        for idx, node in enumerate(remaining_nodes):
            solution[idx % self.num_vehicles].append(node)

        # 3) Repair the solution so that pickup-and-delivery constraints are satisfied
        solution = self.repair_pickup_delivery(solution)

        return solution

    def repair_pickup_delivery(self, solution):
        """
        Ensures that for each pickup node p (where p in self.requests),
        the corresponding delivery node d is in the same route,
        and p appears before d in that route.
        
        If a node or its matching node is missing from a route, we move them
        accordingly.
        """
        # Step 1: Build a quick lookup of which route each node belongs to
        # and the index of the node in that route.
        route_of_node = {}
        for route_idx, route in enumerate(solution):
            for seq_idx, node in enumerate(route):
                route_of_node[node] = (route_idx, seq_idx)

        # Step 2: For each pickup node p in our requests, ensure
        # that its delivery node d is in the same route and that p < d in order.
        for p, d in self.requests.items():
            if p not in route_of_node or d not in route_of_node:
                # If (p) or (d) is missing from the solution, you can decide to skip
                # or create your own logic to insert them.
                continue

            p_route, p_pos = route_of_node[p]
            d_route, d_pos = route_of_node[d]

            # Case A: pickup and delivery are in different routes -> fix that
            if p_route != d_route:
                # Move the delivery node 'd' to pickup's route
                # if d in solution[d_route]:
                solution[d_route].remove(d)
                solution[p_route].append(d)
                # Update route_of_node to reflect the removal
                for seq_idx, node in enumerate(solution[d_route]):
                    route_of_node[node] = (d_route, seq_idx)
                # Update route_of_node to reflect the addition
                new_d_pos = len(solution[p_route]) - 1
                route_of_node[d] = (p_route, new_d_pos)
                # Now p and d are in the same route but might be out of order
                
                # Optionally, check capacity if you have a vehicle capacity constraint
                if len(solution[p_route]) - 1 > self.vehicle_capacity:
                    # If over capacity, you'd implement logic to move extra nodes.
                    pass

            # Step 3: Ensure p_pos < d_pos (pickup index < delivery index)
            # Re-fetch the route/positions because they might have changed in step A
            p_route, p_pos = route_of_node[p]
            d_route, d_pos = route_of_node[d]

            if p_pos > d_pos:
                # We need to reorder p and d
                route = solution[p_route]
                if d in solution[d_route]:
                    solution[d_route].remove(d)
                # Insert d right after p
                route.insert(p_pos + 1, d)

                # Update route_of_node indexing for that route
                for seq_idx, node in enumerate(route):
                    route_of_node[node] = (p_route, seq_idx)

        return solution

    def fitness(self, solution):
        """Evaluate the fitness of a decoded solution."""
        total_distance = 0
        for route in solution:
            if len(route) <= 1:
                continue
            # The first element is the 'leader' key, skip it for distance calculation
            # route[0] is the leader "id", route[1], route[2], ... are nodes
            nodes = route[1:]
            for i in range(1, len(nodes)):
                prev_node = nodes[i - 1]
                curr_node = nodes[i]
                total_distance += self.graph.dist[prev_node][curr_node]
                
        return 1 / total_distance if total_distance > 0 else float('inf')

    def select_parents(self):
        """Select parents using tournament selection."""
        tournament_size = 3
        parents = []
        for _ in range(2):
            competitors = random.sample(self.population, tournament_size)
            # decode each competitor, compute fitness, pick best
            best_competitor = min(
                competitors,
                key=lambda x: self.fitness(self.decode_solution(x))
            )
            parents.append(best_competitor)
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
        # Clip negative values to 0
        return np.clip(individual, 0, None)

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
                fit = self.fitness(decoded_solution)
                if fit > best_fitness:
                    best_fitness = fit
                    best_solution = decoded_solution

            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        self.print_solution(best_solution)
        
        return best_solution

    def print_solution(self, solution):
        """
        Prints a solution for a vehicle routing problem.

        Args:
            solution (list of lists): Each sublist represents a route with
            the first element being the leader key (artificial).
        """
        for i, route in enumerate(solution):
            # Convert possible numpy types to int
            route = [int(node) for node in route]
            leader = route[0] if len(route) > 0 else -1
            path = route[1:] if len(route) > 1 else []
            print(f"Route {i+1}: Leader {leader} -> Path: {path}")
