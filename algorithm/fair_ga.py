import random
import copy
import numpy as np

import sys
import os
# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils import *

def rank_population(graph, population, rank_type="total_distance"):
    """
    Ví dụ: Sắp xếp population dựa trên 1 tiêu chí (VD: tổng quãng đường, 
    hay 1 trọng số multi-objective).
    Ở đây demo: ta chỉ lấy objective[0] (chính là total_distance) để sort tăng dần.
    rank_type = "total_distance" hay "vehicle" hoặc "customer" 
    sẽ tuỳ bạn định nghĩa logic.
    """
    # Tính fitness trước
    for indiv in population:
        if not indiv["objectives"]:  # nếu chưa có objectives
            cal_fitness(graph, indiv)
    # Sort theo objective 0 (giả sử total_distance) - ascending

    if rank_type == "total_distance":
        population.sort(key=lambda x: x["objectives"][0])
    if rank_type == "vehicle":
        population.sort(key=lambda x: x["objectives"][1])
    if rank_type == "customer":
        population.sort(key=lambda x: x["objectives"][2])
    
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
    pop_size = len(population)
    num_selected = int(pop_size * cross_rate)
    # Lấy từ đầu (sau khi rank) => best
    selected = population[:num_selected]
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

def FairGA(graph, 
           max_iteration=100, 
           population_size=30, 
           cross_size=10, 
           localRate=0.3, 
           mutateRate=0.1, 
           elitistRate=0.1,
           crossRate=0.5):
    """
    Cài đặt theo pseudo-code:
    1) Initialize S
    2) while iteration < max_iteration:
       3) rank(S, probability)
       4) remain a portion elitistRate of best S unchanged
       5) for i in range(cross_size):
          6) Deep copy S -> S1, S2
          7..9) for chromosome in S1 -> assign_vehicle(vehicle-fairness)
          10) rank(S1, service vehicle)
          11..13) for chromosome in S2 -> assign_vehicle(customer-fairness)
          14) rank(S2, customer)
          15) SmO = selectCrossRate(S1)
          16) SfA = selectCrossRate(S2)
          17) Schild = Cross(SmO, SfA)
          18..20) if uniform(...) > localRate -> local optimization
          21) S[i] = Schild
       23) Mutate(mutateRate, S)
    25) return Sbest
    """

    # (1) Khởi tạo quần thể S
    population = [create_solution(graph) for _ in range(population_size)]
    
    # Đánh giá fitness ban đầu
    for indiv in population:
        cal_fitness(graph, indiv)
    
    Sbest = None
    iteration = 0

    while iteration < max_iteration:
        # (3) rank(S, probability)
        rank_population(graph, population, rank_type="total_distance")
        
        # Cập nhật Sbest (cá thể tốt nhất hiện tại)
        # Giả sử tiêu chí chính: objective[0] = tổng quãng đường => min
        current_best = population[0]  # sau sort ascending
        print(f"Iteration {iteration}, {current_best['objectives'][0]}, {current_best['objectives'][1]}, {current_best['objectives'][2]}")
        if Sbest is None or current_best["objectives"][0] < Sbest["objectives"][0]:
            Sbest = copy.deepcopy(current_best)
        
        # (4) Elitist: Giữ nguyên top elitistRate * population_size
        num_elites = int(elitistRate * population_size)
        elites = population[:num_elites]  # copy top
        new_population = elites[:]  # sẽ build new_population
        
        # (5) Tạo child bằng crossover
        for i in range(cross_size):
            # (6) deep copy S => S1, S2
            S1 = copy.deepcopy(population)
            S2 = copy.deepcopy(population)
            
            # (7..9) for chromosome in S1 -> Assignvehicle(vehicle-fairness)
            for chrom in S1:
                assign_vehicle(chrom, fairness_type="vehicle-fairness")
            
            # (10) rank(S1, service vehicle)
            rank_population(graph, S1, rank_type="vehicle")
            
            # (11..13) for chromosome in S2 -> Assignvehicle(customer-fairness)
            for chrom in S2:
                assign_vehicle(chrom, fairness_type="customer-fairness")
            
            # (14) rank(S2, customer)
            rank_population(graph, S2, rank_type="customer")
            
            # (15) SmO = selectCrossRate(S1)
            SmO = select_cross_rate(S1, crossRate)
            
            # (16) SfA = selectCrossRate(S2)
            SfA = select_cross_rate(S2, crossRate)
            
            # (17) Schild = Cross(SmO, SfA)
            # Ở đây ta cần chọn 1 cặp từ SmO và SfA để crossover. 
            # Demo: lấy ngẫu nhiên 1 cặp
            if len(SmO) > 0 and len(SfA) > 0:
                p1 = random.choice(SmO)
                p2 = random.choice(SfA)
                child1, child2 = crossover_operator(graph, p1, p2)
                
                # Ta có thể chọn 1 child hoặc cả 2 child. 
                # Demo: chọn child1 để đưa vào new_population
                # (18..20) local optimization
                if random.random() > localRate:
                    local_optimization(graph, child1)
                
                # Tính fitness cho child1
                cal_fitness(graph, child1)
                
                # (21) new_population thêm child1
                new_population.append(child1)
                
                # Tương tự child2:
                if len(new_population) < population_size:
                    if random.random() > localRate:
                        local_optimization(graph, child2)
                    cal_fitness(graph, child2)
                    new_population.append(child2)
            
        # (23) Mutate(mutateRate, S)
        # new_population còn thiếu => ta bổ sung cho đủ population_size
        while len(new_population) < population_size:
            new_population.append(create_solution(graph))

        # Đột biến trên new_population (trừ elites nếu bạn muốn)
        for idx in range(num_elites, len(new_population)):
            new_population[idx] = mutation_operator(graph, new_population[idx], mutation_rate=mutateRate)
            cal_fitness(graph, new_population[idx])

        # Cập nhật population = new_population
        population = new_population
        
        iteration += 1

    return Sbest


# Test FairGA
from graph.graph import Graph

if __name__ == "__main__":
    filepath = 'F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    Sbest = FairGA(graph, max_iteration=100, population_size=30, cross_size=10, localRate=0.3, mutateRate=0.1, elitistRate=0.1, crossRate=0.5)
