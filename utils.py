import numpy as np
import random
from population import Individual, Population

def create_solution(graph):
    """
    Tạo ra một cá thể (chromosome) theo mã hóa LERK.
    Returns:
        individual (dict): {
            "keys": np.ndarray (leader keys + node keys),
            "objectives": list (rỗng, vì chưa tính fitness)
        }
    """
    num_nodes = graph.num_nodes
    vehicle_num = graph.vehicle_num

    # Sinh leader keys (thường lớn hơn hẳn so với node keys, ví dụ [num_nodes, 300])
    leader_keys = np.random.uniform(num_nodes, 300, vehicle_num)
    
    # Sinh node keys [0, 1)
    # Ở đây ta bỏ qua node 0 (thường là depot) nên có num_nodes - 1 keys
    node_keys = np.random.uniform(0, 1, num_nodes - 1)
    
    # Ghép leader_keys và node_keys
    keys = np.concatenate((leader_keys, node_keys))

    individual = {
        "keys": keys,
        "objectives": []  # chưa tính fitness, để trống
    }
    return individual


def create_individual(graph):
    """
    Tạo ra một cá thể (chromosome) theo mã hóa LERK.
    Returns:
        individual (Individual): {
            .chromosome: np.ndarray (leader keys + node keys),
            .objectives: list (rỗng, vì chưa tính fitness)
        }
    """
    num_nodes = graph.num_nodes
    vehicle_num = graph.vehicle_num

    # Sinh leader keys (thường lớn hơn hẳn so với node keys, ví dụ [num_nodes, 300])
    leader_keys = np.random.uniform(num_nodes, 300, vehicle_num)
    
    # Sinh node keys [0, 1)
    # Ở đây ta bỏ qua node 0 (thường là depot) nên có num_nodes - 1 keys
    node_keys = np.random.uniform(0, 1, num_nodes)
    
    # Ghép leader_keys và node_keys
    keys = np.concatenate((leader_keys, node_keys))

    individual = Individual(keys)
    return individual

def create_individual_pickup(graph):
    """
    Tạo ra một cá thể (chromosome) theo mã hóa LERK.
    Returns:
        individual (Individual): {
            .chromosome: np.ndarray (leader keys + node pick up keys),
            .objectives: list (rỗng, vì chưa tính fitness)
        }
    """
    num_nodes = graph.num_nodes
    num_pickup_nodes = len(graph.pickup_nodes)
    # print("num_pickup_nodes: ", num_pickup_nodes)
    vehicle_num = graph.vehicle_num

    # Sinh leader keys (thường lớn hơn hẳn so với node keys, ví dụ [num_nodes, 300])
    leader_keys = np.random.uniform(num_nodes, 300, vehicle_num)
    
    # Sinh node keys [0, 1)
    # Ở đây ta bỏ qua node 0 (thường là depot) nên có num_nodes - 1 keys
    node_keys = np.random.uniform(0, 1, num_pickup_nodes)
    # print(len(node_keys))
    
    # Ghép leader_keys và node_keys
    keys = np.concatenate((leader_keys, node_keys))

    individual = Individual(keys)
    return individual


def repair_time(graph, solution):
    """
    Hàm sắp xếp thứ tự các điểm (node) trong mỗi route của 'solution' 
    theo trường due_time (từ nhỏ đến lớn) trong graph.

    Args:
        graph: Đối tượng graph, trong đó graph.nodes[node_id].due_time 
               trả về due_time của node_id.
        solution: Danh sách các route, mỗi route là một list chứa 
                  các chỉ số node (VD: [leader, n1, n2, ..., nK, leader, ...]).
                  - Thông thường node >= graph.num_nodes có thể được coi là 
                    "leader" hoặc "break" node tùy quy ước.

    Returns:
        new_solution: Bản sao của solution, với mỗi route được sắp xếp 
                      lại các node < graph.num_nodes theo due_time tăng dần.
    """
    new_solution = []

    for route in solution:
        # Tách "leader" hoặc node >= graph.num_nodes (nếu có)
        # và các node "thật" trong cùng route
        
        real_nodes_sorted = sorted(
            route[1:], 
            key=lambda x: graph.nodes[x].due_time
        )

        # Tuỳ thuộc vào logic của bạn:
        # 1) Giữ nguyên leader ở đầu route và cuối route
        # 2) Chèn leader xen kẽ
        #
        # Ở đây, ví dụ đơn giản: 
        # - Đặt leader_nodes (nếu có) ở đầu route, rồi đến real_nodes_sorted
        # - Nếu có nhiều "leader" node, bạn tự quyết định cách chèn phù hợp
        new_route = route[:1] + real_nodes_sorted
        new_solution.append(new_route)

    return new_solution

def repair_pickup_delivery(graph, solution):
    """
    Đảm bảo cặp (p, d) cùng route và p đứng trước d.
    graph.requests: dict {pickup_node: delivery_node}
    """
    # Xây map: node -> (route_idx, index_in_route)
    node_position = {}
    for r_idx, route in enumerate(solution):
        for i, node in enumerate(route):
            node_position[node] = (r_idx, i)

    # Với mỗi pickup p, đảm bảo delivery d ở cùng route p
    for p, d in graph.requests.items():
        if p not in node_position or d not in node_position:
            continue  # hoặc chèn logic nếu chưa có p/d trong solution
        
        p_r, p_i = node_position[p]
        d_r, d_i = node_position[d]
        
        # Nếu khác route, move d về route p
        if p_r != d_r:
            solution[d_r].remove(d)
            solution[p_r].append(d)
            # Cập nhật lại node_position cho route cũ
            for i, nd in enumerate(solution[d_r]):
                node_position[nd] = (d_r, i)
            # Cập nhật lại node_position cho route mới
            new_index = len(solution[p_r]) - 1
            node_position[d] = (p_r, new_index)
            d_r, d_i = p_r, new_index
        
        # Đảm bảo p_i < d_i
        if p_i > d_i:
            # Move d để nằm sau p
            route = solution[p_r]
            route.remove(d)
            route.insert(p_i + 1, d)
            # Cập nhật lại index
            for i, nd in enumerate(route):
                node_position[nd] = (p_r, i)

    return solution


def crossover_operator(graph, parent1, parent2):
    """
    Thực hiện lai ghép (crossover) giữa 2 cá thể cha/mẹ (parent1, parent2)
    Trả về 2 cá thể con (child1, child2) không tính fitness.
    """
    p1_keys = parent1.chromosome
    p2_keys = parent2.chromosome
    length = len(p1_keys)
    
    # Uniform crossover
    mask = np.random.randint(0, 2, length)
    c1_keys = np.where(mask, p1_keys, p2_keys)
    c2_keys = np.where(mask, p2_keys, p1_keys)

    # Tạo 2 cá thể con
    child1 = Individual(c1_keys)
    child2 = Individual(c2_keys)
    # Nếu cần ràng buộc Pickup & Delivery, ta có thể gọi hàm repair ở đây

    # child1 = repair_pickup_delivery(graph, child1.chromosome)
    # child2 = repair_pickup_delivery(graph, child2.chromosome)

    return child1, child2

def mutation_operator(graph, individual, mutation_rate=0.1):
    """
    Đột biến (mutation) lên 1 cá thể, trả về 1 cá thể con.
    """
    keys = individual.chromosome.copy()
    
    for i in range(len(keys)):
        if random.random() < mutation_rate:
            # Gaussian noise
            keys[i] += np.random.normal(0, 0.1)
    
    # Không cho keys âm
    keys = np.clip(keys, 0, None)
    
    offspring = Individual(keys)

    # Nếu cần ràng buộc Pickup & Delivery, có thể gọi repair ở đây
    # offspring = repair_pickup_delivery(graph, offspring)

    return offspring


# def cost_list(graph, route: list):
#     distance = 0
#     ve_fair = []
#     cus_fair = []
#     time = 0
#     for i in range(2, len(route)): #@check
        
#         if route[i-1] >= graph.num_nodes and route[i] >= graph.num_nodes:
#             ve_fair.append(0)
#             distance = 0 
#             time = 0
#             continue
#         elif route[i-1] >= graph.num_nodes:
#             distance = 0
#             time = 0
#             distance += graph.dist[0][route[i]]
#             time += graph.dist[0][route[i]] / graph.vehicle_speed
#         elif route[i] >= graph.num_nodes:
#             distance += graph.dist[route[i-1]][0]
#             time += graph.dist[route[i-1]][0] / graph.vehicle_speed
#             ve_fair.append(distance)
#             distance = 0
#             time = 0
#             continue
#         else:
#             distance += graph.dist[route[i-1]][route[i]]
#             time += graph.dist[route[i-1]][route[i]] / graph.vehicle_speed
#         node = route[i]
#         customer = graph.nodes[node]
#         time = max(time, customer.ready_time)
#         time += 0 #service time, set to 0
#         if time > customer.due_time:
#             cus_fair.append(time - customer.due_time)
#         else:
#             cus_fair.append(0)
    
#     distance += graph.dist[route[-1]][0]
#     ve_fair.append(distance)

#     vehicle_fairness = variance(ve_fair)
#     total_distance = sum(ve_fair)
#     customer_fairness = variance(cus_fair)

#     return total_distance, vehicle_fairness, customer_fairness


def cost(graph, solution):
    """
    Tính các chỉ số:
      1) total_distance  - tổng quãng đường của toàn bộ solution
      2) vehicle_fairness - độ lệch bình phương trung bình về quãng đường giữa các xe
      3) customer_fairness - độ lệch bình phương trung bình về trễ hạn khách hàng
    
    Args:
        graph: đối tượng Graph (chứa self.dist, self.nodes, self.vehicle_speed, ...)
        solution: list of routes, mỗi route là list các node [0, i1, i2, ..., 0] (giả sử 0 là depot)
    
    Returns:
        (total_distance, vehicle_fairness, customer_fairness)
    """
    
    total_distance = 0.0
    ve_fair = []   # lưu quãng đường của từng vehicle (route)
    cus_fair = []  # lưu độ trễ (tardiness) của từng khách hàng
    
    for route in solution:
        route_distance = 0.0
        time = 0.0
        
        # Giả sử route = [depot, ..., depot] => tính khoảng cách từng cặp liên tiếp
        for i in range(1,len(route) - 1): #skip first leader node
            current_node = route[i]
            next_node = route[i+1]

            if current_node == next_node:
                raise Exception("Route has duplicate nodes: {}".format(route))
            
            # Cộng quãng đường giữa current_node -> next_node
            dist_ij = graph.dist[current_node][next_node]

            if dist_ij > 100000000000:
                print("graph size: ", graph.num_nodes)
                raise Exception("Invalid distance: {} -> {}".format(current_node, next_node))

            route_distance += dist_ij
            
            # Tính thời gian di chuyển
            travel_time = dist_ij / graph.vehicle_speed if graph.vehicle_speed else 0
            time += travel_time
            
            # Nếu next_node != 0 => đó là khách hàng thật (không phải depot)
            if next_node != 0:
                customer = graph.nodes[next_node]
                
                # Nếu đến sớm hơn ready_time, phải chờ => time = max(time, ready_time)
                time = max(time, customer.ready_time)
                
                # Ở đây coi service_time = 0, nếu có thì cộng thêm
                # time += customer.service_time
                
                # Nếu trễ hơn due_time => cộng vào cus_fair
                if time > customer.due_time:
                    tardiness = time - customer.due_time
                    cus_fair.append(tardiness)
                else:
                    cus_fair.append(0.0)
        
        # route_distance chính là quãng đường cho route này
        ve_fair.append(route_distance)
        total_distance += route_distance
    
    # vehicle_fairness = variance(ve_fair)
    # customer_fairness = variance(cus_fair)

    vehicle_fairness = standard_deviation(ve_fair)
    customer_fairness = standard_deviation(cus_fair)
    
    return total_distance, vehicle_fairness, customer_fairness

def cost_energy(graph, solution):
    """
    Computes:
      1) total_energy    - total energy consumption of the entire solution (litres of fuel)
      2) vehicle_fairness - standard deviation of energy consumption among vehicles (km)
      3) customer_fairness - standard deviation of customer tardiness (minutes)
    
    Args:
        graph: Graph object containing attributes like dist, nodes, 
               vehicle_speed, and energy-related parameters.
        solution: list of routes, each route is a list of nodes [0, i1, i2, ..., 0] 
                  (assuming 0 is depot)
    
    Returns:
        (total_energy, vehicle_fairness, customer_fairness)
    """

    # Extract energy-related constants from graph or define them here
    cd = graph.cd         # e.g., 0.7
    xi = graph.xi         # e.g., 1
    kappa = graph.kappa   # e.g., 44
    p = graph.p           # e.g., 1.2
    A = graph.A           # e.g., 3.192
    mk = graph.mk         # e.g., 3.2
    g = graph.g           # e.g., 9.81
    cr = graph.cr         # e.g., 0.01
    psi = graph.psi       # e.g., 737
    pi_val = graph.pi     # e.g., 0.2   (renamed to avoid conflict with math.pi)
    R = graph.R           # e.g., 165
    eta = graph.eta       # e.g., 0.36
    v_speed_km_h = 40
    v_speed_km_m = 0.6666666666666666

    total_energy = 0.0
    ve_energy = []   # list to store energy consumption per vehicle
    cus_tardiness = []  # list to store tardiness for each customer

    # Function to compute energy for a segment between two nodes
    def energy_for_leg(current_node, next_node, current_capacity):
        # Calculate distance between nodes
        d_ij = graph.dist[current_node][next_node]

        # Power consumption terms
        p_ij = 0.5 * cd * p * A * v_speed_km_h**3 + (mk + current_capacity) * g * cr * v_speed_km_h

        # Compute energy consumption L_ij using the provided formula
        # Here, we use the formula:
        # L_ij = xi/(kappa*psi) * (pi*R + p_ij/eta) * d_ij/v_speed
        L_ij = (xi / (kappa * psi)) * (pi_val * R + p_ij / eta) * (d_ij / v_speed_km_h)

        return L_ij, d_ij

    for route in solution:
        route_energy = 0.0
        current_capacity = 0.0
        time = 0.0  # to compute tardiness if needed


        L_ij, d_ij = energy_for_leg(0, route[1], current_capacity)
        route_energy += L_ij
        travel_time = d_ij / v_speed_km_m
        time += travel_time
        
        # Process each leg in the route
        for i in range(1, len(route) - 1):
            current_node = route[i]
            next_node = route[i+1]

            # Compute energy and distance for this segment
            L_ij, d_ij = energy_for_leg(current_node, next_node, current_capacity)
            route_energy += L_ij

            # Travel time calculation
            travel_time = d_ij / v_speed_km_m
            time += travel_time

            # If next_node is a customer (not depot), handle time windows and capacity updates
            if next_node != 0:
                customer = graph.nodes[next_node]
                # Wait if arriving early
                time = max(time, customer.ready_time)
                # Update capacity if needed (assuming pickup adds capacity for simplicity)
                current_capacity += customer.demand  # adjust based on pickup/delivery logic


                time += customer.service_time

                # Check for tardiness and record
                if time > customer.due_time:
                    tardiness = time - customer.due_time
                    cus_tardiness.append(tardiness)
                else:
                    cus_tardiness.append(0.0)

        L_ij, d_ij = energy_for_leg(route[-1], 0, current_capacity)
        route_energy += L_ij

        ve_energy.append(route_energy)
        total_energy += route_energy

    # Calculate fairness metrics using standard deviation
    vehicle_fairness = standard_deviation(ve_energy)
    customer_fairness = standard_deviation(cus_tardiness) if cus_tardiness else 0.0

    return total_energy, vehicle_fairness, customer_fairness


def variance(list):
    mean = sum(list) / len(list)
    variance = sum((x - mean) ** 2 for x in list) / len(list)
    return variance

def standard_deviation(list):
    return np.sqrt(variance(list))


def decode_solution(problem, keys):
    """
    Giải mã từ LERK -> danh sách các route.
    Mỗi route: [leader_key, node1, node2, ...]
    """
    vehicle_num = problem.vehicle_num
    num_nodes = problem.num_nodes
    
    # Tách leader_keys và node_keys
    leader_keys = keys[:vehicle_num]
    node_keys = keys[vehicle_num:]
    
    # Sắp xếp các node (từ 1..num_nodes-1) theo thứ tự tăng dần của node_keys
    sorted_indices = np.argsort(node_keys)
    remaining_nodes = list(sorted_indices + 1)  # shift +1 vì ta bỏ node 0 (depot)
    
    # Chuẩn bị data structure
    solution = [[] for _ in range(vehicle_num)]
    
    # Sort leader_keys để xác định thứ tự gán route
    sorted_leader_indices = np.argsort(leader_keys)
    
    # Mỗi vehicle một route, route bắt đầu bằng leader "id" 
    naive_capacity = (num_nodes - 1) // vehicle_num
    
    for i, leader_idx in enumerate(sorted_leader_indices):
        # Lấy route thứ i
        solution[i].append(int(leader_idx))  # Leader
        assigned_nodes = remaining_nodes[:naive_capacity]
        solution[i].extend(assigned_nodes)
        remaining_nodes = remaining_nodes[naive_capacity:]
    
    # Nếu còn node dư, gán tiếp vòng tròn
    for idx, nd in enumerate(remaining_nodes):
        solution[idx % vehicle_num].append(nd)
    
    # Ở đây, ta có thể gọi hàm repairPickupDelivery để đảm bảo ràng buộc (nếu cần)
    solution = repair_pickup_delivery(problem, solution)
    
    return solution


def decode_solution_pickup(problem, keys):
    """
    Giải mã từ LERK -> danh sách các route, với yêu cầu:
      - Chỉ lấy các pickup node trong problem.pickup_nodes (mỗi phần tử là một đối tượng Node).
      - Sau khi xếp pickup node, thêm delivery node tương ứng ngay sau pickup node đó.

    Mỗi route có dạng: [leader_key, p1, d1, p2, d2, ...].

    Args:
        problem: đối tượng chứa thông tin, trong đó:
           - problem.vehicle_num: số lượng xe (vehicle)
           - problem.pickup_nodes: danh sách các đối tượng Node (mỗi Node có p.nid)
           - problem.graph.nodes[node_id].did: delivery node ứng với pickup node_id
        keys: mảng random-keys. 
              - keys[:vehicle_num] là leader_keys
              - keys[vehicle_num:] là node_keys để sắp xếp pickup nodes

    Returns:
        solution: list các route, mỗi route là một list **chỉ số node** (int).
    """

    vehicle_num = problem.vehicle_num

    # Danh sách pickup node (mỗi phần tử là object Node, p.nid là chỉ số node)
    pickup_nodes = problem.pickup_nodes
    num_pickup_nodes = len(pickup_nodes)

    # 1) Tách leader_keys và node_keys
    leader_keys = keys[:vehicle_num]
    node_keys = keys[vehicle_num:]

    # 2) Ghép pickup_nodes với node_keys để sắp xếp
    #    -> [(nodeObj, key), (nodeObj, key), ...]
    pickup_key_pairs = list(zip(pickup_nodes, node_keys))

    # 3) Sắp xếp theo node_key để được thứ tự pickup
    #    -> danh sách nodeObj đã xếp
    pickup_key_pairs.sort(key=lambda x: x[1])
    sorted_pickups = [p for (p, _) in pickup_key_pairs]

    # 4) Chuẩn bị solution rỗng: mỗi phần tử solution[i] là route thứ i
    solution = [[] for _ in range(vehicle_num)]

    # 5) Sắp xếp leader_keys để xác định thứ tự gán route
    sorted_leader_indices = np.argsort(leader_keys)

    # 6) Gán pickup (và delivery) vào route theo naive_capacity
    naive_capacity = num_pickup_nodes // vehicle_num
    idx_pickup = 0

    # print("Leader indices: ", sorted_leader_indices)

    # Mỗi route: [leader_key, ...pickup + delivery...]
    for i, leader_idx in enumerate(sorted_leader_indices):
        # Thêm leader (dùng int(leader_idx) như trong logic LERK truyền thống)
        solution[i].append(int(leader_idx))
        
        # Lấy một “mẻ” pickup theo capacity
        assigned_pickups = sorted_pickups[idx_pickup : idx_pickup + naive_capacity]
        idx_pickup += naive_capacity

        # Gắn pickup & delivery tương ứng vào route
        for p in assigned_pickups:
            p_id = p.nid  # Chỉ số node của pickup
            d_id = problem.nodes[p_id].did  # Chỉ số node của delivery
            solution[i].append(p_id)
            solution[i].append(d_id)

    # 7) Nếu còn pickup dư, ta gán vòng tròn vào các route
    leftover_pickups = sorted_pickups[idx_pickup:]
    for j, p in enumerate(leftover_pickups):
        route_idx = j % vehicle_num
        p_id = p.nid
        d_id = problem.nodes[p_id].did
        solution[route_idx].append(p_id)
        solution[route_idx].append(d_id)

    # (Nếu bạn có hàm repair_time, gọi ở đây để sắp xếp thêm theo due_time hoặc logic khác)
    solution = repair_time(problem, solution)

    return solution


# def cal_fitness(problem, individual):
#     """
#     Tính toán fitness (đa mục tiêu) cho một cá thể (individual).
#     Ở đây, ta sử dụng hàm cost(route) trả về:
#         total_distance, vehicle_fairness, customer_fairness
#     Lưu các giá trị đó thành list và gán vào individual["objectives"].
    
#     Args:
#         problem: đối tượng chứa thông tin bài toán (trong đó có .cost(route)).
#         individual (dict): 
#             {
#                 "keys": np.ndarray (mãng LERK),
#                 "objectives": list rỗng hoặc đã tính
#             }

#     Returns:
#         list: [total_distance, vehicle_fairness, customer_fairness]
#     """
#     # 1) Giải mã từ random keys -> route (dạng một list duy nhất 
#     #    có chèn sentinel >= problem.graph.num_nodes để đánh dấu chia tuyến)
#     route = decode_solution(problem, individual["keys"])
    
#     # 2) Tính cost
#     total_distance, vehicle_fairness, customer_fairness = cost(problem, route)
    
#     # 3) Lưu vào individual["objectives"] (mục tiêu đa mục tiêu)
#     individual["objectives"] = [total_distance, vehicle_fairness, customer_fairness]
    
#     return individual["objectives"]


def calculate_fitness(problem, individual):
    """
    Tính toán fitness (đa mục tiêu) cho một cá thể (individual).
    Ở đây, ta sử dụng hàm cost(route) trả về:
        total_distance, vehicle_fairness, customer_fairness
    Lưu các giá trị đó thành list và gán vào individual["objectives"].
    
    Args:
        problem: đối tượng chứa thông tin bài toán (trong đó có .cost(route)).
        individual: đối tượng, có .chromosome và .objectives

    Returns:
        list: [total_distance, vehicle_fairness, customer_fairness]
    """
    # 1) Giải mã từ random keys -> route (dạng một list duy nhất 
    #    có chèn sentinel >= problem.graph.num_nodes để đánh dấu chia tuyến)
    route = decode_solution_pickup(problem, individual.chromosome)
    
    # 2) Tính cost
    total_distance, vehicle_fairness, customer_fairness = cost_energy(problem, route)
    
    # 3) Lưu vào individual["objectives"] (mục tiêu đa mục tiêu)
    individual.objectives = [total_distance, vehicle_fairness, customer_fairness]
    
    return individual.objectives