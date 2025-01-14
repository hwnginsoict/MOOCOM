import random
import copy
import numpy as np

import sys
import os
# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils import calculate_fitness, create_individual_pickup, crossover_operator, mutation_operator

from population import Population



def rank_population(graph, population, rank_type="total_distance"):
    """
    Ví dụ: Sắp xếp population dựa trên 1 tiêu chí (VD: tổng quãng đường, 
    hay 1 trọng số multi-objective).
    Ở đây demo: ta chỉ lấy objective[0] (chính là total_distance) để sort tăng dần.
    rank_type = "total_distance" hay "vehicle" hoặc "customer" 
    sẽ tuỳ bạn định nghĩa logic.
    """
    # Tính fitness trước
    for indiv in population.indivs:
        if not indiv.objectives:  # nếu chưa có objectives
            cal_fitness(graph, indiv)
    # Sort theo objective 0 (giả sử total_distance) - ascending

    if rank_type == "total_distance":
        population.indivs.sort(key=lambda x: x.objectives[0])
    if rank_type == "vehicle":
        population.indivs.sort(key=lambda x: x.objectives[1])
    if rank_type == "customer":
        population.indivs.sort(key=lambda x: x.objectives[2])
    
def assign_vehicle(chromosome, fairness_type):
    """
    Giả sử đây là bước "Assignvehicle(vehicle-fairness)" hay "Assignvehicle(customer-fairness)".
    """
    # TODO: logic gán vehicle/fairness
    pass

def select_cross_rate(population, cross_rate=0.5):
    """
    Chọn ra một phần trăm (cross_rate) cá thể từ population để ghép cặp crossover.
    Ở đây demo: lấy ngẫu nhiên cross_rate% top trong population.
    """
    pop_size = len(population.indivs)
    num_selected = int(pop_size * cross_rate)
    # Lấy từ đầu (sau khi rank) => best
    selected = population.indivs[:num_selected]
    return selected

def local_optimization(graph, individual):
    """
    Áp dụng local optimization cho cá thể individual.
    Demo: ta chỉ in ra "local optimization" (placeholder).
    Bạn có thể cài đặt 2-opt, 3-opt,... tuỳ ý.
    """
    # TODO: local search logic
    # ví dụ: 2-opt route, or shifting...
    pass

def MO_GA(graph, 
           max_iteration=100, 
           population_size=30, 
           cross_size=10, 
           localRate=0.3, 
           mutateRate=0.1, 
           elitistRate=0.1,
           crossRate=0.5):

    # (1) Khởi tạo quần thể S
    population = Population(population_size)
    population.indivs = [create_individual_pickup(graph) for _ in range(population_size)]
    
    # Đánh giá fitness ban đầu
    for indiv in population.indivs:
        calculate_fitness(graph, indiv)
    
    Sbest = None
    iteration = 0

    while iteration < max_iteration:
        # (3) rank(S, probability)
        rank_population(graph, population, rank_type="total_distance")
        
        
        current_best = population.indivs[0]  # sau sort ascending
        print(f"Iteration {iteration}, {current_best.objectives[0]}, {current_best.objectives[1]}, {current_best.objectives[2]}")
        if Sbest is None or current_best.objectives[0] < Sbest.objectives[0]:
            Sbest = copy.deepcopy(current_best)
        
        # (4) Elitist: Giữ nguyên top elitistRate * population_size
        num_elites = int(elitistRate * population_size)
        elites = population.indivs[:num_elites]  # copy top
        new_population = Population(len(elites[:]))  # sẽ build new_population
        new_population.indivs = elites[:]
        
        # (5) Tạo child bằng crossover
        for i in range(cross_size):
        
            p1, p2 = random.sample(population.indivs, 2)
            child1, child2 = crossover_operator(graph, p1, p2)
            
            if random.random() > localRate:
                local_optimization(graph, child1)
            
            # Tính fitness cho child1
            calculate_fitness(graph, child1)
            
            # (21) new_population thêm child1
            new_population.indivs.append(child1)
            
            # Tương tự child2:
            if len(new_population.indivs) < population_size:
                if random.random() > localRate:
                    local_optimization(graph, child2)
                calculate_fitness(graph, child2)
                new_population.indivs.append(child2)
            
        # (23) Mutate(mutateRate, S)
        # new_population còn thiếu => ta bổ sung cho đủ population_size
        while len(new_population.indivs) < population_size:
            new_population.indivs.append(create_individual_pickup(graph))

        # Đột biến trên new_population (trừ elites nếu bạn muốn)
        for idx in range(num_elites, len(new_population.indivs)):
            new_population.indivs[idx] = mutation_operator(graph, new_population.indivs[idx], mutation_rate=mutateRate)
            calculate_fitness(graph, new_population.indivs[idx])

        # Cập nhật population = new_population
        population = new_population
        
        iteration += 1
        
    return Sbest

# Test FairGA
from graph.graph import Graph

if __name__ == "__main__":
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    Sbest = GA(graph, max_iteration=100, population_size=30, cross_size=10, localRate=0.3, mutateRate=0.1, elitistRate=0.1, crossRate=0.5)
