import random
import numpy as np
from population import Individual

class GetData():
    def __init__(self, n_instance: int, n_items: int):
        self.n_instance = n_instance
        self.n_items = n_items

    def generate_instances(self):
        np.random.seed(2025)
        instance_data = []
        for _ in range(self.n_instance):
            weights = np.random.rand(self.n_items)
            values_obj1 = np.random.rand(self.n_items)
            values_obj2 = np.random.rand(self.n_items)
            if 50 <= self.n_items < 100:
                capacity = 12.5
            elif 100 <= self.n_items <= 200:
                capacity = 25
            else:
                raise ValueError("Number of items must be between 50 and 200.")

            instance_data.append((weights, values_obj1, values_obj2, capacity))
        return instance_data

# def tour_cost(instance, route):
def cost(solution: np.ndarray, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
    if np.sum(solution * weight_lst) > capacity:
        return 1e10, 1e10  # Penalize infeasible solutions
    total_val1 = np.sum(solution * value1_lst)
    total_val2 = np.sum(solution * value2_lst)
    return -total_val1, -total_val2  

def tour_cost(instance, route):
    weight_lst, value1_lst, value2_lst, capacity = instance
    return cost(route.chromosome, weight_lst, value1_lst, value2_lst, capacity)


def create_individual(instance, n_items):
    """
    Tạo một cá thể ngẫu nhiên cho bài toán knapsack.
    Mỗi gene là 0 hoặc 1, đại diện cho việc không chọn hoặc chọn một vật phẩm.
    """
    individual = [random.randint(0, 1) for _ in range(n_items)]
    individual = repair_solution(instance[0], instance[3], np.array(individual))
    return Individual(individual)

def crossover(instance, p1, p2):
    """
    Crossover một điểm (one-point crossover) cho knapsack.
    Trả về hai cá thể con.
    """
    parent1 = p1.chromosome
    parent2 = p2.chromosome
    size = len(parent1)
    
    # Chọn điểm cắt
    cut = random.randint(1, size - 1)
    
    # Tạo con bằng cách kết hợp phần đầu của parent1 với phần cuối của parent2 và ngược lại
    # print(parent1, parent2, cut)
    # print(type(parent1), type(parent2))
    child1 = np.concatenate((parent1[:cut], parent2[cut:]))
    child2 = np.concatenate((parent2[:cut], parent1[cut:]))

    child1 = repair_solution(instance[0], instance[3], np.array(child1))
    child2 = repair_solution(instance[0], instance[3], np.array(child2))    
    
    return Individual(child1), Individual(child2)

def mutation(problem, indi, mutation_rate=0.05):
    """
    Đột biến bằng cách lật một số bit với xác suất mutation_rate.
    """
    chromosome = indi.chromosome[:]
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip bit
    
    chromosome = repair_solution(problem[0], problem[3], np.array(chromosome))
    return Individual(chromosome)


def repair_solution(weight_lst, capacity, solution):
    while np.sum(solution * weight_lst) > capacity:
        idx = np.random.choice(np.where(solution == 1)[0])  # Chọn ngẫu nhiên một vật phẩm đã chọn
        solution[idx] = 0  # Bỏ chọn vật phẩm đó
    return solution