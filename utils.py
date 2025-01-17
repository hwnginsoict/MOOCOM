import numpy as np
import random
from population import Individual, Population

def create_individual_pickup_lerk(graph):
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

    # check = 0
    # for route in solution:
    #     check += len(route) - 1
    # if check != graph.num_nodes -1:
    #     print(check)
    #     raise KeyError("Error repair_time")

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


def crossover_operator_lerk(graph, parent1, parent2):
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

    return child1, child2

def mutation_operator_lerk(graph, individual, mutation_rate=0.1):
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

    return offspring

from utils_new import cost_full

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


def calculate_fitness_lerk(problem, individual):
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
    total_distance, vehicle_fairness, customer_fairness, max_time = cost_full(problem, route)
    
    # 3) Lưu vào individual["objectives"] (mục tiêu đa mục tiêu)
    individual.objectives = [total_distance, vehicle_fairness, customer_fairness, max_time]
    
    return individual.objectives