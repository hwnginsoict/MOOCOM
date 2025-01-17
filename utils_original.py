class Gene:
    def __init__(self, gene_type, gene_id, value):
        """
        gene_type: str, "vehicle" hoặc "request"
        gene_id: int, id của xe hoặc request
        value: float, giá trị real-coded
        """
        self.type = gene_type
        self.id = gene_id
        self.value = value

    def __repr__(self):
        return f"Gene({self.type}, id={self.id}, value={self.value:.3f})"


import numpy as np
import random

from population import Individual

def create_individual_lerk(problem):
    """
    Tạo 1 cá thể LERK dưới dạng 1 list Gene (length = vehicle_num + request_num).
    Ràng buộc:
      - Gene đầu (index 0) chắc chắn là 1 xe.
      - Gene cuối (index -1) chắc chắn là 1 request.
    """

    vehicle_ids = list(range(problem.vehicle_num))  # giả sử xe đánh số từ 0..(vehicle_num-1)
    request_ids = list(problem.request_ids)         # tuỳ bạn tổ chức request_id thế nào

    # Số phần tử
    n_vehicles = len(vehicle_ids)
    n_requests = len(request_ids)
    N = n_vehicles + n_requests

    genes = []

    # 1) Chọn ngẫu nhiên 1 xe cho vị trí đầu
    #    (hoặc bạn muốn fix 1 xe cụ thể, ví dụ xe "3", thì thay vào)
    first_vehicle = random.choice(vehicle_ids)
    vehicle_ids.remove(first_vehicle)  # lấy ra rồi thì loại khỏi list
    first_value = np.random.uniform(0, 1)  # hoặc bất kỳ range nào
    genes.append(Gene("vehicle", first_vehicle, first_value))

    # 2) Chọn ngẫu nhiên 1 request cho vị trí cuối
    last_request = random.choice(request_ids)
    request_ids.remove(last_request)
    last_value = np.random.uniform(0, 1)
    # (Tạm thời ta append sau cùng, xíu nữa sẽ đặt vào vị trí -1)
    last_gene = Gene("request", last_request, last_value)

    # 3) Trộn (shuffle) các xe còn lại + request còn lại để “nhét” vào giữa
    middle_genes = []
    for vid in vehicle_ids:
        val = np.random.uniform(0, 1)
        middle_genes.append(Gene("vehicle", vid, val))

    for rid in request_ids:
        val = np.random.uniform(0, 1)
        middle_genes.append(Gene("request", rid, val))

    random.shuffle(middle_genes)

    # 4) Ghép: first_gene + middle_genes + last_gene
    #    Sao cho độ dài cuối cùng đúng N = vehicle_num + request_num
    genes = genes + middle_genes + [last_gene]

    # Kiểm tra đủ độ dài
    assert len(genes) == N, "Chiều dài chromosome không khớp"

    # Cuối cùng, gói vào Individual. 
    # Ta lưu chromosome dạng list[Gene], thay vì np.ndarray như trước.
    individual = Individual(genes)
    return individual


def decode_solution(problem, chromosome):
    """
    chromosome: list[Gene], với gene.type in {"vehicle", "request"}.
    Ta sẽ sort theo gene.value để suy ra thứ tự thăm.
    Mỗi khi gặp gene.type="vehicle" => mở route mới.
    gene.type="request" => gán vào route hiện tại.
    """

    # 1) Sắp xếp gene theo value
    sorted_genes = sorted(chromosome, key=lambda g: g.value)

    solution = []
    current_route = None

    for g in sorted_genes:
        if g.type == "vehicle":
            # Mở route mới
            # Nếu current_route đang có sẵn => push vào solution trước
            if current_route is not None and len(current_route) > 0:
                solution.append(current_route)
            current_route = [g.id]  # bắt đầu route bởi xe
        else:
            # g.type == "request"
            if current_route is None:
                # Trường hợp hy hữu: gene đầu mà lại là request
                # (trên lý thuyết đã ép gene đầu là vehicle, nên ít xảy ra)
                current_route = []
            current_route.append(g.id)

    # Sau vòng for, còn current_route cuối
    if current_route is not None and len(current_route) > 0:
        solution.append(current_route)

    return solution


from utils_new import cost_full

def calculate_fitness_lerk(problem, individual):
    """
    Tính fitness cho 1 cá thể LERK (được mã hoá thành list[Gene]).
    """
    # 1) Giải mã
    route = decode_solution(problem, individual.chromosome)

    # 2) Tính cost
    #  Giả sử bạn đã có hàm cost_full(problem, route) -> (dist, fairness1, fairness2, max_time)
    total_distance, vehicle_fairness, customer_fairness, max_time = cost_full(problem, route)

    # 3) Gán vào đối tượng
    individual.objectives = [total_distance, vehicle_fairness, customer_fairness, max_time]
    return individual.objectives


def crossover_operator_lerk(problem, parent1, parent2):
    """
    Uniform crossover trên list[Gene].
    Ràng buộc: gene[0] luôn "vehicle", gene[-1] luôn "request".
    """
    c1_chrom = []
    c2_chrom = []
    length = len(parent1.chromosome)

    for i in range(length):
        g1 = parent1.chromosome[i]
        g2 = parent2.chromosome[i]

        if i == 0:
            # Buộc con phải "vehicle"
            # Gỉa sử ta cứ lấy g1 cho child1, g2 cho child2 
            # (hoặc hoán đổi nếu g2 cũng là vehicle)
            c1_chrom.append(g1 if g1.type == "vehicle" else g2)
            c2_chrom.append(g2 if g2.type == "vehicle" else g1)
        elif i == length - 1:
            # Buộc con phải "request"
            c1_chrom.append(g1 if g1.type == "request" else g2)
            c2_chrom.append(g2 if g2.type == "request" else g1)
        else:
            # vị trí giữa => uniform crossover
            if random.random() < 0.5:
                c1_chrom.append(g1)
                c2_chrom.append(g2)
            else:
                c1_chrom.append(g2)
                c2_chrom.append(g1)

    child1 = Individual(c1_chrom)
    child2 = Individual(c2_chrom)
    return child1, child2

def mutation_operator_lerk(problem, individual, mutation_rate=0.1):
    """
    Đột biến: thay đổi gene.value ngẫu nhiên.
    Vẫn phải giữ gene[0].type="vehicle" và gene[-1].type="request".
    """
    new_chrom = []
    for i, g in enumerate(individual.chromosome):
        # copy gene
        gene_type = g.type
        gene_id = g.id
        gene_val = g.value

        # bỏ qua gene đầu & cuối (nếu muốn cố định type)
        if i not in [0, len(individual.chromosome)-1]:
            if random.random() < mutation_rate:
                # Thay đổi value (ví dụ Gaussian noise)
                gene_val += np.random.normal(0, 0.1)
                # hoặc random.uniform(0,1)...

        # Tạo gene mới
        new_g = Gene(gene_type, gene_id, gene_val)
        new_chrom.append(new_g)

    # Tạo cá thể mới
    offspring = Individual(new_chrom)
    return offspring
