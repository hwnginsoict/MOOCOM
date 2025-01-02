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


if __name__ == "__main__":
    problem = Problem('F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv')
    print(problem.graph.num_nodes)
    print(problem.cost([213, 2 ,4, 5, 214, 3,5,7]))