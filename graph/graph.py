import numpy as np
from .node import Node
import csv

class Graph:
    def __init__(self, file_path):
        self.read_file(file_path)

    def read_file(self, file_path):
        node_list = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                node_list.append(row)

        num_nodes = len(node_list)
        nodes = [
            Node(
                int(item[0]),  # nid
                float(item[1]),  # x
                float(item[2]),  # y
                float(item[3]),  # ready_time
                float(item[4]),  # due_time
                float(item[5]),  # demand
                float(item[6]),  # service_time
                int(item[7]),  # pid 
                int(item[8]),   # did 
                float(item[9])  # time
            ) for item in node_list
        ]

        dist = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            node_a = nodes[i]
            for j in range(i + 1, num_nodes):
                node_b = nodes[j]
                dist[i][j] = Graph.calculate_dist(node_a, node_b)

        vehicle_num = nodes[0].demand 
        # print(vehicle_num)
        vehicle_cap = nodes[0].service_time
        vehicle_speed = nodes[0].time

        # print(node_list[0])

        self.num_nodes = num_nodes
        self.nodes = nodes
        self.dist = dist
        self.vehicle_num = vehicle_num
        self.vehicle_cap = vehicle_cap
        self.vehicle_speed = vehicle_speed

                
    @staticmethod
    def calculate_dist(node_a, node_b):
        dis = np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))
        if dis == 0:
            return float('inf')
        return dis
    

if __name__ == "__main__":
    graph = Graph('F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv')
    # print(graph.num_nodes)
    # print(graph.nodes)
    # print(graph.dist)
    print(graph.vehicle_num)
    print(graph.vehicle_cap)
    print(graph.vehicle_speed)