import numpy as np
import random

def create_solution(problem):
    """
    Tạo ra một cá thể (chromosome) theo mã hóa LERK.
    Returns:
        individual (dict): {
            "keys": np.ndarray (leader keys + node keys),
            "objectives": list (rỗng, vì chưa tính fitness)
        }
    """
    num_nodes = problem.num_nodes
    num_vehicles = problem.num_vehicles

    # Sinh leader keys (thường lớn hơn hẳn so với node keys, ví dụ [num_nodes, 300])
    leader_keys = np.random.uniform(num_nodes, 300, num_vehicles)
    
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



def crossover_operator(problem, parent1, parent2):
    """
    Thực hiện lai ghép (crossover) giữa 2 cá thể cha/mẹ (parent1, parent2)
    Trả về 2 cá thể con (child1, child2) không tính fitness.
    """
    p1_keys = parent1["keys"]
    p2_keys = parent2["keys"]
    length = len(p1_keys)
    
    # Uniform crossover
    mask = np.random.randint(0, 2, length)
    c1_keys = np.where(mask, p1_keys, p2_keys)
    c2_keys = np.where(mask, p2_keys, p1_keys)

    # Tạo 2 cá thể con
    child1 = {
        "keys": c1_keys,
        "objectives": []
    }
    child2 = {
        "keys": c2_keys,
        "objectives": []
    }

    # Nếu cần ràng buộc Pickup & Delivery, ta có thể gọi hàm repair ở đây
    child1 = repair_pickup_delivery(problem, child1)
    child2 = repair_pickup_delivery(problem, child2)

    return child1, child2


def mutation_operator(problem, individual, mutation_rate=0.1):
    """
    Đột biến (mutation) lên 1 cá thể, trả về 1 cá thể con.
    """
    keys = individual["keys"].copy()
    
    for i in range(len(keys)):
        if random.random() < mutation_rate:
            # Gaussian noise
            keys[i] += np.random.normal(0, 0.1)
    
    # Không cho keys âm
    keys = np.clip(keys, 0, None)
    
    offspring = {
        "keys": keys,
        "objectives": []
    }

    # Nếu cần ràng buộc Pickup & Delivery, có thể gọi repair ở đây
    # offspring = repair_pickup_delivery(problem, offspring)

    return offspring


def cal_fitness(problem, individual):
    """
    Tính hàm mục tiêu (fitness) cho individual.
    Ở đây ví dụ tính tổng quãng đường, trả về list (đa mục tiêu).
    """
    # 1) Decode solution từ random keys
    solution = decode_solution(problem, individual["keys"])
    
    # 2) Tính quãng đường
    dist_matrix = problem.graph.dist  # hoặc problem.graph.dist
    total_distance = 0.0
    for route in solution:
        # route[0] là leader key (trong decode_solution)
        # route[1:] là các node
        nodes = route[1:]
        for i in range(1, len(nodes)):
            prev_node = nodes[i-1]
            curr_node = nodes[i]
            total_distance += dist_matrix[prev_node][curr_node]

    # Giả sử ta chỉ có 1 mục tiêu là tổng quãng đường
    # Thông thường bài toán Minimization => ta trả về total_distance
    # Nếu GA framework muốn maximize fitness,
    # có thể trả về 1/total_distance hoặc -total_distance tuỳ cách quy ước
    objectives = [total_distance]  

    # 3) Lưu kết quả vào individual
    individual["objectives"] = objectives
    return objectives


def decode_solution(problem, keys):
    """
    Giải mã từ LERK -> danh sách các route.
    Mỗi route: [leader_key, node1, node2, ...]
    """
    num_vehicles = problem.num_vehicles
    num_nodes = problem.num_nodes
    
    # Tách leader_keys và node_keys
    leader_keys = keys[:num_vehicles]
    node_keys = keys[num_vehicles:]
    
    # Sắp xếp các node (từ 1..num_nodes-1) theo thứ tự tăng dần của node_keys
    sorted_indices = np.argsort(node_keys)
    remaining_nodes = list(sorted_indices + 1)  # shift +1 vì ta bỏ node 0 (depot)
    
    # Chuẩn bị data structure
    solution = [[] for _ in range(num_vehicles)]
    
    # Sort leader_keys để xác định thứ tự gán route
    sorted_leader_indices = np.argsort(leader_keys)
    
    # Mỗi vehicle một route, route bắt đầu bằng leader "id" 
    naive_capacity = (num_nodes - 1) // num_vehicles
    
    for i, leader_idx in enumerate(sorted_leader_indices):
        # Lấy route thứ i
        solution[i].append(int(leader_idx))  # Leader
        assigned_nodes = remaining_nodes[:naive_capacity]
        solution[i].extend(assigned_nodes)
        remaining_nodes = remaining_nodes[naive_capacity:]
    
    # Nếu còn node dư, gán tiếp vòng tròn
    for idx, nd in enumerate(remaining_nodes):
        solution[idx % num_vehicles].append(nd)
    
    # Ở đây, ta có thể gọi hàm repairPickupDelivery để đảm bảo ràng buộc (nếu cần)
    # solution = repair_pickup_delivery(problem, solution)
    
    return solution


def repair_pickup_delivery(problem, solution):
    """
    Đảm bảo cặp (p, d) cùng route và p đứng trước d.
    problem.requests: dict {pickup_node: delivery_node}
    """
    # Xây map: node -> (route_idx, index_in_route)
    node_position = {}
    for r_idx, route in enumerate(solution):
        for i, node in enumerate(route):
            node_position[node] = (r_idx, i)

    # Với mỗi pickup p, đảm bảo delivery d ở cùng route p
    for p, d in problem.requests.items():
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
