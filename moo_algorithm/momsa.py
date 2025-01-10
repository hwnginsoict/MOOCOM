import numpy as np
import multiprocessing
import sys
import os

# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from moo_algorithm.metric import cal_hv_front
from population import Population, Individual
from utils import mutation_operator, calculate_fitness, create_individual_pickup
from graph.graph import Graph

# ------------------------------------------------------------------
# A typical spiral-flight update used in Moth Flame Optimization or Moth Swarm
# can be something like:
#   M_{new} = Flame + Dist * exp(b * t) * cos(2*pi*t)
# where Dist = |Moth - Flame|, t is a random in [-1,1], and b is a constant.
# You can customize or refine this for your MOMSA variant.
# ------------------------------------------------------------------
def spiral_flight_update(moth_position, flame_position, b=1.0):
    """
    Spiral flight update that moves 'moth_position' around 'flame_position'.
    b: constant to shape the logarithmic spiral
    Returns new position for the moth (array-like).
    """
    # Convert to numpy arrays
    M = np.array(moth_position, dtype=float)
    F = np.array(flame_position, dtype=float)

    # Distance between moth and flame
    dist = np.linalg.norm(M - F)

    # l in [-1, 1]
    l = (2.0 * np.random.rand()) - 1.0

    # Spiral formula (simplified version)
    new_pos = F + dist * np.exp(b * l) * np.cos(2.0 * np.pi * l)

    # If your solution is multidimensional (chromosome), you might do:
    # new_pos_dim = F + (M - F) * np.exp(b * l) * np.cos(2.0 * np.pi * l)
    # For a single number, the above is fine. Adapt for D-dimensional if needed.
    if isinstance(moth_position, list):
        return new_pos.tolist()
    return new_pos


class MOMSAPopulation(Population):
    """
    Inherits from your Population class.
    Instead of weight vectors + Tchebycheff, we use
    a multi-objective Moth Swarm style approach.
    """
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.external_pop = []  # Store non-dominated solutions (Pareto archive)

    def update_external(self, indivs: list):
        """
        Incorporate new individuals into the external population (archive).
        Remove dominated solutions.
        """
        for indi in indivs:
            old_len = len(self.external_pop)
            # Remove any solutions dominated by indi
            self.external_pop = [
                other for other in self.external_pop
                if not indi.dominates(other)
            ]
            # If we actually removed some, indi is definitely non-dominated
            if len(self.external_pop) < old_len:
                self.external_pop.append(indi)
                continue
            # Otherwise, check if indi is dominated by any existing member
            for other in self.external_pop:
                if other.dominates(indi):
                    break
            else:
                self.external_pop.append(indi)

    def filter_external(self):
        """
        Optionally remove duplicates or reduce archive size.
        A simple example that removes exact-duplicate objective vectors:
        """
        unique_objs = set()
        new_ext = []
        for indi in self.external_pop:
            obj_tuple = tuple(indi.objectives)
            if obj_tuple not in unique_objs:
                new_ext.append(indi)
                unique_objs.add(obj_tuple)
        self.external_pop = new_ext

    def reproduction(self, problem,
                     mutation_operator,
                     mutation_rate=0.1,
                     b=1.0):
        """
        Generate new offspring using the spiral flight mechanism.
        Usually, we pick a 'flame' from the best solutions (e.g., from external_pop).
        """
        offspring = []

        # If no external solutions, fallback to population itself
        if len(self.external_pop) == 0:
            flames = self.indivs
        else:
            flames = self.external_pop

        # Sort or shuffle flames if desired:
        # Example: sort by hypervolume contribution or some metric
        # or just pick random flames from external_pop
        np.random.shuffle(flames)

        # Number of flames used can decrease over time in MFO,
        # or you can just pick random from external archive:
        # For simplicity, pick random flames for each moth
        for i in range(self.pop_size):
            moth = self.indivs[i]
            flame = np.random.choice(flames)
            new_chrom = spiral_flight_update(moth.chromosome, flame.chromosome, b=b)

            # We can also do a random "mutation" or small random tweak
            if np.random.rand() < mutation_rate:
                # Convert new_chrom to an Individual-like structure
                temp_indi = Individual(chromosome=new_chrom)
                temp_indi = mutation_operator(problem, temp_indi)
                new_chrom = temp_indi.chromosome

            # Create an offspring Individual
            off = Individual(chromosome=new_chrom)
            offspring.append(off)

        return offspring

    def momsa_selection(self, offspring):
        merged = self.indivs + offspring

        nondom = []
        for ind in merged:
            if not any(o.dominates(ind) for o in merged if o != ind):
                nondom.append(ind)

        # If nondom is already bigger than pop_size, down-select:
        if len(nondom) > self.pop_size:
            # e.g. random or crowding-based selection
            np.random.shuffle(nondom)
            self.indivs = nondom[:self.pop_size]
        # If nondom is *fewer* than pop_size, fill up with some dominated solutions
        elif len(nondom) < self.pop_size:
            # pick the dominated solutions
            dominated = [ind for ind in merged if ind not in nondom]
            # shuffle to avoid bias
            np.random.shuffle(dominated)
            needed = self.pop_size - len(nondom)
            self.indivs = nondom + dominated[:needed]
        else:
            # nondom == pop_size
            self.indivs = nondom



#
# ------------------------------------------------------------------
# Main run function for MOMSA
# ------------------------------------------------------------------
#

def run_momsa(processing_number,
              problem,
              indi_list,
              pop_size,
              max_gen,
              mutation_rate,
              calculate_fitness_func=calculate_fitness,
              b=1.0):
    """
    Example run function that parallels your MOEA/D structure but uses
    the MOMSAPopulation class. Evaluates in parallel, does spiral updates,
    and stores non-dominated solutions in an external archive.
    """
    np.random.seed(0)

    # 1) Create population & pre-generate chromosomes
    momsa_pop = MOMSAPopulation(pop_size)
    momsa_pop.pre_indi_gen(indi_list)

    # 2) Evaluate the initial population in parallel
    pool = multiprocessing.Pool(processing_number)
    arg = [(problem, indi) for indi in momsa_pop.indivs]
    result = pool.starmap(calculate_fitness_func, arg)
    for individual, fitness in zip(momsa_pop.indivs, result):
        individual.objectives = fitness

    # 3) Update external (archive) with initial population
    momsa_pop.update_external(momsa_pop.indivs)
    momsa_pop.filter_external()

    # 4) (Optional) print initial HV
    print("Generation 0 HV:", cal_hv_front(momsa_pop.external_pop, np.array([1, 1, 1])))

    # 5) Main loop
    for gen in range(max_gen):
        # 5a) Generate new offspring using spiral flight
        offspring = momsa_pop.reproduction(problem,
                                           mutation_operator=mutation_operator,
                                           mutation_rate=mutation_rate,
                                           b=b)

        # 5b) Evaluate offspring in parallel
        arg = [(problem, off) for off in offspring]
        result = pool.starmap(calculate_fitness_func, arg)
        for off, fit in zip(offspring, result):
            off.objectives = fit

        # 5c) Update external with offspring
        momsa_pop.update_external(offspring)
        momsa_pop.filter_external()

        # 5d) Perform selection to get next generation
        momsa_pop.momsa_selection(offspring)

        # 5e) Print or log progress
        hv = cal_hv_front(momsa_pop.external_pop, np.array([1, 1, 1]))
        print(f"Generation {gen+1} HV: {hv}")

    pool.close()
    return cal_hv_front(momsa_pop.external_pop, np.array([1, 1, 1]))

if __name__ == "__main__":
    filepath = "./data/dpdptw/200/LC1_2_1.csv"
    graph = Graph(filepath)

    # Create a big pool of random individuals (solutions)
    indi_list = [create_individual_pickup(graph) for _ in range(1000)]

    # Run MOMSA
    final_hv = run_momsa(
        processing_number=4,
        problem=graph,
        indi_list=indi_list,
        pop_size=1000,        # e.g. 1000 moths
        max_gen=100,          # number of iterations
        mutation_rate=0.1,   # probability to apply mutation on each spiral update
        calculate_fitness_func=calculate_fitness,
        b=1.0                # spiral constant
    )

    print("Final HV:", final_hv)

