from graph.request import Request
from graph.graph import Graph
import csv
class Problem:
    def __init__(self, filepath: str):
        self.graph = Graph(filepath)
    
    def total_distance(self, route: list):
        total_distance = 0
        for i in range(len(route) - 1):
            if route[i] >= self.graph.num_nodes and route[i + 1] >= self.graph.num_nodes:
                total_distance += 0 
            elif route[i] >= self.graph.num_nodes:
                total_distance += self.graph.dist[0][route[i + 1]]
            elif route[i + 1] >= self.graph.num_nodes:
                total_distance += self.graph.dist[route[i]][0]
            else:
                total_distance += self.graph.dist[route[i]][route[i + 1]]
        return total_distance
    
    def cost(self, route: list):
        distance = 0
        ve_fair = []
        cus_fair = []
        time = 0
        for i in range(1, len(route)):
            
            if route[i-1] >= self.graph.num_nodes and route[i] >= self.graph.num_nodes:
                ve_fair.append(0)
                distance = 0 
                time = 0
                continue
            elif route[i-1] >= self.graph.num_nodes:
                distance = 0
                time = 0
                distance += self.graph.dist[0][route[i]]
                time += self.graph.dist[0][route[i]] / self.graph.vehicle_speed
            elif route[i] >= self.graph.num_nodes:
                distance += self.graph.dist[route[i-1]][0]
                time += self.graph.dist[route[i-1]][0] / self.graph.vehicle_speed
                ve_fair.append(distance)
                distance = 0
                time = 0
                continue
            else:
                distance += self.graph.dist[route[i-1]][route[i]]
                time += self.graph.dist[route[i-1]][route[i]] / self.graph.vehicle_speed
            node = route[i]
            customer = self.graph.nodes[node]
            time = max(time, customer.ready_time)
            time += 0 #service time, set to 0
            if time > customer.due_time:
                cus_fair.append(time - customer.due_time)
            else:
                cus_fair.append(0)
        
        distance += self.graph.dist[route[-1]][0]
        ve_fair.append(distance)

        vehicle_fairness = self.variance(ve_fair)
        total_distance = sum(ve_fair)
        customer_fairness = self.variance(cus_fair)

        # print(len(ve_fair))
        # print(len(cus_fair))
        return total_distance, vehicle_fairness, customer_fairness

    
    def variance(self, list):
        mean = sum(list) / len(list)
        variance = sum((x - mean) ** 2 for x in list) / len(list)
        return variance
    
    # def customer_time_fairness(self, route: list):
    #     time = 0
    #     list = []
    #     for i in range(len(route) - 1):
    #         if route[i] >= self.graph.num_nodes and route[i + 1] >= self.graph.num_nodes:
                
    #         elif route[i] >= self.graph.num_nodes:
    #             time += self.graph.nodes[0].time
    #         elif route[i + 1] >= self.graph.num_nodes:
    #             time += self.graph.nodes[route[i]].time
    #         else:
    #             time += self.graph.nodes[route[i]].time
    #     return time

    def check_pdp(self, solution):
        """
        Checks if the given solution satisfies the pickup–delivery constraints.
        A 'solution' is assumed to be a list of routes, each route is a list of node indices.
        Some nodes may be 'leaders' if they exceed self.graph.num_nodes; 
        here we store them in the same dictionary but ignore them when checking 
        pickup–delivery feasibility.

        The constraints are:
        1) Pickup node p and delivery node d must be in the same route.
        2) p must appear before d in that route.

        Returns:
            bool: True if all pickup–delivery constraints are satisfied, False otherwise.
        """

        # --------------------------------------------------------
        # 1) Build a lookup from node -> (route_index, position_in_route).
        #    We store EVERY node in route_of_node, but we'll only check
        #    those < self.graph.num_nodes for feasibility constraints.
        # --------------------------------------------------------
        route_of_node = {}
        for route_idx, route in enumerate(solution):
            for seq_idx, node in enumerate(route):
                route_of_node[node] = (route_idx, seq_idx)

        # --------------------------------------------------------
        # 2) For each node in [0, 1, 2, ..., self.graph.num_nodes - 1],
        #    identify if it's a pickup node (pid=0, did != 0).
        #    Then check the constraints for that pickup–delivery pair.
        # --------------------------------------------------------
        for node_id in range(self.graph.num_nodes):
            pickup_id = self.graph.nodes[node_id].pid
            delivery_id = self.graph.nodes[node_id].did

            # If pid=0 and did!=0, this node is a pickup (node_id = p)
            # and 'delivery_id' is the matching delivery node = d
            if pickup_id == 0 and delivery_id != 0:
                p = node_id
                d = delivery_id

                # (A) Ensure both p and d are actually in the solution
                if p not in route_of_node:
                    print(f"Violation: pickup {p} not found in any route.")
                    return False
                if d not in route_of_node:
                    print(f"Violation: delivery {d} not found in any route.")
                    return False

                p_route, p_pos = route_of_node[p]
                d_route, d_pos = route_of_node[d]

                # (B) They must be in the same route
                if p_route != d_route:
                    print(f"Violation: pickup {p} is in route {p_route}, "
                        f"while delivery {d} is in route {d_route}.")
                    return False

                # (C) The pickup node must appear before its delivery node
                if p_pos > d_pos:
                    print(f"Violation: pickup {p} appears after delivery {d} "
                        f"in route {p_route}.")
                    return False

        # If we get here, all pickup–delivery constraints are satisfied
        return True




if __name__ == "__main__":
    problem = Problem('F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv')
    print(problem.graph.num_nodes)
    print(problem.cost([213, 2 ,4, 5, 214, 3,5,7]))