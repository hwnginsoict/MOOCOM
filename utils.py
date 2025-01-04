import numpy as np
import random

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
    child1 = repair_pickup_delivery(graph, child1)
    child2 = repair_pickup_delivery(graph, child2)

    return child1, child2


def mutation_operator(graph, individual, mutation_rate=0.1):
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
    offspring = repair_pickup_delivery(graph, offspring)

    return offspring


def cost_list(graph, route: list):
    distance = 0
    ve_fair = []
    cus_fair = []
    time = 0
    for i in range(1, len(route)):
        
        if route[i-1] >= graph.num_nodes and route[i] >= graph.num_nodes:
            ve_fair.append(0)
            distance = 0 
            time = 0
            continue
        elif route[i-1] >= graph.num_nodes:
            distance = 0
            time = 0
            distance += graph.dist[0][route[i]]
            time += graph.dist[0][route[i]] / graph.vehicle_speed
        elif route[i] >= graph.num_nodes:
            distance += graph.dist[route[i-1]][0]
            time += graph.dist[route[i-1]][0] / graph.vehicle_speed
            ve_fair.append(distance)
            distance = 0
            time = 0
            continue
        else:
            distance += graph.dist[route[i-1]][route[i]]
            time += graph.dist[route[i-1]][route[i]] / graph.vehicle_speed
        node = route[i]
        customer = graph.nodes[node]
        time = max(time, customer.ready_time)
        time += 0 #service time, set to 0
        if time > customer.due_time:
            cus_fair.append(time - customer.due_time)
        else:
            cus_fair.append(0)
    
    distance += graph.dist[route[-1]][0]
    ve_fair.append(distance)

    vehicle_fairness = variance(ve_fair)
    total_distance = sum(ve_fair)
    customer_fairness = variance(cus_fair)

    # print(len(ve_fair))
    # print(len(cus_fair))
    return total_distance, vehicle_fairness, customer_fairness


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
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i+1]
            
            # Cộng quãng đường giữa current_node -> next_node
            dist_ij = graph.dist[current_node][next_node]
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
    
    vehicle_fairness = variance(ve_fair)
    customer_fairness = variance(cus_fair)
    
    return total_distance, vehicle_fairness, customer_fairness


def variance(list):
    mean = sum(list) / len(list)
    variance = sum((x - mean) ** 2 for x in list) / len(list)
    return variance


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


def cal_fitness(problem, individual):
    """
    Tính toán fitness (đa mục tiêu) cho một cá thể (individual).
    Ở đây, ta sử dụng hàm cost(route) trả về:
        total_distance, vehicle_fairness, customer_fairness
    Lưu các giá trị đó thành list và gán vào individual["objectives"].
    
    Args:
        problem: đối tượng chứa thông tin bài toán (trong đó có .cost(route)).
        individual (dict): 
            {
                "keys": np.ndarray (mãng LERK),
                "objectives": list rỗng hoặc đã tính
            }

    Returns:
        list: [total_distance, vehicle_fairness, customer_fairness]
    """
    # 1) Giải mã từ random keys -> route (dạng một list duy nhất 
    #    có chèn sentinel >= problem.graph.num_nodes để đánh dấu chia tuyến)
    route = decode_solution(problem, individual["keys"])
    
    # 2) Tính cost
    total_distance, vehicle_fairness, customer_fairness = cost(problem, route)
    
    # 3) Lưu vào individual["objectives"] (mục tiêu đa mục tiêu)
    individual["objectives"] = [total_distance, vehicle_fairness, customer_fairness]
    
    return individual["objectives"]