import numpy as np
from copy import deepcopy
from graph.request import Request
import pandas as pd
from population import Individual

class Element:
    def __init__(self, isLeader = None, root = None, left_child = None, right_child = None, 
                 request_id = None, vehicle_id = None):
        self.isLeader  = isLeader
        self.root = root
        self.left_child = left_child
        self.right_child = right_child
        self.request_id = request_id
        self.vehicle_id = None

    def copy_element(self, other_element):
        self.isLeader = other_element.isLeader
        self.root = other_element.root
        self.left_child = other_element.left_child
        self.right_child = other_element.right_child
        self.request_id = other_element.request_id
        self.vehicle_id = other_element.vehicle_id
    
    def set_value(self):
        self.root = np.random.rand()
        self.left_child = np.random.rand()
        self.right_child = np.random.rand()
        self.repair_element()

    def repair_element(self):
        if self.root < 0:
            self.root = 0
        if self.root > 1:
            self.root = 1
        if self.left_child < 0:
            self.left_child = 0
        if self.left_child > 1:
            self.left_child = 1
        if self.right_child < 0:
            self.right_child = 0
        if self.right_child > 1:
            self.right_child = 1
        if self.left_child > self.right_child:
            self.left_child, self.right_child = self.right_child, self.left_child

def create_chromosome(num_vehicle, num_request):
    chromosome = []
    for i in range(num_vehicle - 1):
        leader_element = Element(isLeader= True)
        leader_element.set_value()
        leader_element.vehicle_id = i + 1
        chromosome.append(leader_element)
    for i in range(num_request):
        request_element = Element(isLeader= False)
        request_element.set_value()
        request_element.request_id = i + 1
        chromosome.append(request_element)
    return chromosome


def decode_chromosome(chromosome, num_vehicle):
    chromosome_copy = deepcopy(chromosome)
    chromosome_copy.sort(key=lambda x: x.root)
    element_routes = [[] for i in range(num_vehicle)]
    i = 0
    for element in chromosome_copy:
        if element.isLeader:
            i = i + 1
        else:
            element_routes[i].append(element)

    vehicle_routes = [[] for i in range(num_vehicle)]
    
    for i in range(num_vehicle):
        vehicle_routes[i] = np.zeros(2*len(element_routes[i]), dtype=int)
        pickup_delivery_value = []
        for element in element_routes[i]:
            pickup_delivery_value.append((element.left_child, element.request_id))
            pickup_delivery_value.append((element.right_child, - element.request_id))
        pickup_delivery_value.sort(key=lambda x: x[0])
        for j in range(len(pickup_delivery_value)):
            vehicle_routes[i][j] = pickup_delivery_value[j][1]
    return vehicle_routes


#########################################
# Calculated objectives
#########################################
def cal_distance_two_request(request1: Request, request2: Request, current_state = "pickup", next_state = "pickup"):
    if current_state == "pickup" and next_state == "pickup":
        return np.sqrt((request1.px - request2.px)**2 + (request1.py - request2.py)**2)
    if current_state == "pickup" and next_state == "delivery":
        return np.sqrt((request1.px - request2.dx)**2 + (request1.py - request2.dy)**2)
    if current_state == "delivery" and next_state == "pickup":
        return np.sqrt((request1.dx - request2.px)**2 + (request1.dy - request2.py)**2)
    if current_state == "delivery" and next_state == "delivery":
        return np.sqrt((request1.dx - request2.dx)**2 + (request1.dy - request2.dy)**2)
    
class Problem:
    def __init__(self, cd=0.7, xi=1, kappa=44, p=1.2, A=3.192, mk=3.2, g=9.81, 
                 cr=0.01, psi=737, pi=0.2, R=165, eta=0.36):
        self.cd = cd
        self.p = p
        self.A = A 
        self.mk = mk
        self.g = g
        self.cr = cr
        self.xi = xi
        self.kappa = kappa
        self.psi = psi 
        self.pi = pi
        self.R = R 
        self.eta = eta
        self.num_vehicle = None
        self.num_request = None
        self.request_list = None
        self.speed_vehicle = None
        self.capacity_vehicle = None

    def read_file(self, file_path, num_vehicle, speed_vehicle, capacity_vehicle):
        self.num_vehicle = num_vehicle
        self.speed_vehicle = speed_vehicle
        self.capacity_vehicle = capacity_vehicle
        data = pd.read_csv(file_path)
        self.request_list = []
        depot = Request(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.request_list.append(depot)
        for i in range(len(data)):
            request = Request(data.iloc[i]['nid'], data.iloc[i]['px'], data.iloc[i]['py'], data.iloc[i]['dx'], data.iloc[i]['dy'], 
                              data.iloc[i]['pready'], data.iloc[i]['pdue'], data.iloc[i]['dready'], data.iloc[i]['ddue'], 
                              data.iloc[i]['demand'],data.iloc[i]['service_time'], data.iloc[i]['time'])
            self.request_list.append(request)
        self.num_request = len(self.request_list) - 1
        

def cal_objectives(vehicle_routes, problem: Problem):
    def cal_energy_consumption(problem: Problem, current_capacity, request1: Request, request2: Request, current_state = "pickup", next_state = "pickup"):
        p_ij = 0.5*problem.cd*problem.p*problem.A*problem.speed_vehicle**3 + (problem.mk + current_capacity)*problem.g*problem.cr*problem.speed_vehicle
        d_ij = cal_distance_two_request(request1, request2, current_state, next_state)
        L_ij = problem.xi/(problem.kappa*problem.psi)*(problem.pi*problem.R + p_ij/problem.eta)*d_ij/problem.speed_vehicle
        if next_state == "pickup":
            current_capacity = current_capacity + request2.demand
        elif next_state == "delivery":
            current_capacity = current_capacity - request2.demand
        return L_ij, current_capacity, d_ij
    
    EC_vehicle = np.zeros(problem.num_vehicle)
    DT_vehicle = np.zeros(problem.num_vehicle)
    WT_customer = np.zeros(problem.num_request + 1)
    for i in range(problem.num_vehicle):
        current_capacity = 0
        current_time = 0
        if len(vehicle_routes[i]) == 0:
            continue
        EC_vehicle[i], current_capacity, d_ij = cal_energy_consumption(problem, current_capacity, problem.request_list[0], problem.request_list[vehicle_routes[i][0]], "pickup", "pickup")
        current_time = max(current_time + d_ij/problem.speed_vehicle + problem.request_list[vehicle_routes[i][0]].service_time, problem.request_list[vehicle_routes[i][0]].pready + problem.request_list[vehicle_routes[i][0]].service_time)
        WT_customer[vehicle_routes[i][0]] = max(0, current_time - problem.request_list[vehicle_routes[i][0]].pdue)
        DT_vehicle[i] = DT_vehicle[i] + d_ij

        for j in range(len(vehicle_routes[i])-1):
            current_state = "pickup"
            next_state = "pickup"
            if vehicle_routes[i][j] < 0:
                current_state = "delivery"
            if vehicle_routes[i][j+1] < 0:
                next_state = "delivery"
            EC_ij, current_capacity, d_ij = cal_energy_consumption(problem, current_capacity, problem.request_list[abs(vehicle_routes[i][j])], problem.request_list[abs(vehicle_routes[i][j+1])], current_state, next_state)
            EC_vehicle[i] = EC_vehicle[i] + EC_ij
            tau_start = current_time + d_ij/problem.speed_vehicle
            if next_state == "pickup":
                tau_start = max(tau_start, problem.request_list[abs(vehicle_routes[i][j+1])].pready)
            else:
                tau_start = max(tau_start, problem.request_list[abs(vehicle_routes[i][j+1])].dready)
            current_time = tau_start + problem.request_list[abs(vehicle_routes[i][j+1])].service_time
            tau_due = problem.request_list[abs(vehicle_routes[i][j+1])].pdue
            if next_state == "delivery":
                tau_due = problem.request_list[abs(vehicle_routes[i][j+1])].ddue
            WT_customer[abs(vehicle_routes[i][j+1])] = max(0, current_time - tau_due)
            DT_vehicle[i] = DT_vehicle[i] + d_ij
        
        EC_ij, current_capacity, d_ij = cal_energy_consumption(problem, current_capacity, problem.request_list[abs(vehicle_routes[i][-1])], problem.request_list[0], "delivery", "pickup")
        EC_vehicle[i] = EC_vehicle[i] + EC_ij
        DT_vehicle[i] = DT_vehicle[i] + d_ij
    return sum(EC_vehicle), np.std(WT_customer[1:]), np.std(DT_vehicle)


def cal_fitness(problem: Problem, individual: Individual):
    vehicle_routes = decode_chromosome(individual.chromosome, problem.num_vehicle)
    EC, CF, VF = cal_objectives(vehicle_routes, problem)
    return [EC/20000, CF/20000, VF/20000]

def crossover_operator(problem: Problem, parent1: Individual, parent2: Individual):
    # SBX crossover
    eta_c = 2
    chromosome1 = deepcopy(parent1.chromosome)
    chromosome2 = deepcopy(parent2.chromosome)
    # u = np.random.rand()
    # if u <= 0.5:
    #     beta = 0.5*((1 + 2*u)**(1/(eta_c + 1)))
    # else:
    #     beta = 0.5*((1 + 2*(1 - u))**(-1/(eta_c + 1)))
    for i in range(len(chromosome1)):
        u = np.random.rand()
        if u <= 0.5:
            beta = 0.5*((1 + 2*u)**(1/(eta_c + 1)))
        else:
            beta = 0.5*((1 + 2*(1 - u))**(-1/(eta_c + 1)))
        chromosome1[i].root = 0.5*((1 + beta)*chromosome1[i].root + (1 - beta)*chromosome2[i].root)
        chromosome2[i].root = 0.5*((1 - beta)*chromosome1[i].root + (1 + beta)*chromosome2[i].root)
        chromosome1[i].left_child = 0.5*((1 + beta)*chromosome1[i].left_child + (1 - beta)*chromosome2[i].left_child)
        chromosome2[i].left_child = 0.5*((1 - beta)*chromosome1[i].left_child + (1 + beta)*chromosome2[i].left_child)
        chromosome1[i].right_child = 0.5*((1 + beta)*chromosome1[i].right_child + (1 - beta)*chromosome2[i].right_child)
        chromosome2[i].right_child = 0.5*((1 - beta)*chromosome1[i].right_child + (1 + beta)*chromosome2[i].right_child)
        chromosome1[i].repair_element()
        chromosome2[i].repair_element()
    return Individual(chromosome1), Individual(chromosome2)

def mutation_operator(problem: Problem, individual: Individual):
    # Polynomial mutation
    eta_m = 2
    chromosome = deepcopy(individual.chromosome)
    for i in range(len(chromosome)):
        u = np.random.rand()
        delta = min(chromosome[i].root, 1 - chromosome[i].root)
        if u <= 0.5:
            delta = delta*(2*u)**(1/(eta_m + 1)) - 1
        else:
            delta = 1 - (2*(1 - u))**(1/(eta_m + 1))
        chromosome[i].root = chromosome[i].root + delta
        u = np.random.rand()
        delta = min(chromosome[i].left_child, 1 - chromosome[i].left_child)
        if u <= 0.5:
            delta = delta*(2*u)**(1/(eta_m + 1)) - 1
        else:
            delta = 1 - (2*(1 - u))**(1/(eta_m + 1))
        chromosome[i].left_child = chromosome[i].left_child + delta
        u = np.random.rand()
        delta = min(chromosome[i].right_child, 1 - chromosome[i].right_child)
        if u <= 0.5:
            delta = delta*(2*u)**(1/(eta_m + 1)) - 1
        else:
            delta = 1 - (2*(1 - u))**(1/(eta_m + 1))
        chromosome[i].right_child = chromosome[i].right_child + delta
        chromosome[i].repair_element()
    return Individual(chromosome)


def create_individual(problem: Problem):
    chromosome = create_chromosome(problem.num_vehicle, problem.num_request)
    return Individual(chromosome)


from moo_algorithm.nsga_ii import run_nsga_ii
from moo_algorithm.pfg_moea_knee import run_pfgmoea
from moo_algorithm.moead_plus import run_moead_plus, init_weight_vectors_3d
from moo_algorithm.nsga_iii import run_nsga_iii
if __name__  == "__main__":
    # num_vehicle = 5
    # num_request = 4
    # chromosome = create_chromosome(num_vehicle, num_request)
    # vehicle_routes = decode_chromosome(chromosome, num_vehicle)
    # print(vehicle_routes)
    problem = Problem()
    problem.read_file(r"data\requests.csv", 10, 40, 1000)
    # chromosome = create_chromosome(problem.num_vehicle, problem.num_request)
    # vehicle_routes = decode_chromosome(chromosome, problem.num_vehicle)
    # print(vehicle_routes)
    # EC, CF, VF = cal_objectives(vehicle_routes, problem)
    # print(EC, CF, VF)
    indi_list = [create_individual(problem) for _ in range(50)]
    a = run_nsga_ii(10, problem, indi_list, 50, 50, crossover_operator, mutation_operator, 0.8, 0.1, cal_fitness)
    print("GECCO_2025")
    b = run_pfgmoea(10, problem, indi_list, 50, 50, 5, 0.01, crossover_operator, mutation_operator, 0.8, 0.1, cal_fitness)
    # print("GECCO_2025")
    # c = run_moead_plus(10, problem, indi_list, 100, 50, 5, init_weight_vectors_3d, crossover_operator,mutation_operator, cal_fitness)
    # print("GECCO_2025")
    # d = run_nsga_iii(10, problem, indi_list, 100, 50, crossover_operator, mutation_operator, 0.8, 0.1, cal_fitness)


    
    