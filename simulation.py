from graph.graph import Graph
#

def run(timestep = 100, filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'):
    current_route = []
    graph = Graph(filepath)
    time = 0
    while True:
        time += timestep
        if time > graph.nodes[0].due_time:
            break


if __name__ == "__main__":
    run()



