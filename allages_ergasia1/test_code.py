import time
import networkx as nx
import matplotlib.pyplot as plt

def hungarian_algorithm(cost_matrix):
    n = len(cost_matrix)
    G = nx.Graph()

    # 'left' are the job nodes (0..n-1)
    left = range(n)
    # 'right' are the worker nodes (n..2n-1)
    right = range(n, 2*n)

    print("Left partition (Jobs):", list(left))
    print("Right partition (Workers):", list(right))

    # Build the bipartite graph
    for i in left:
        for j in right:
            G.add_edge(i, j, weight=cost_matrix[i][j - n])
    
    # -- (Optional) Draw the bipartite graph to visualize it --

    # Build a position dict to place the two partitions side by side
    pos = {}
    # Put left nodes (jobs) at x=0, spaced by their index
    for index, node in enumerate(left):
        pos[node] = (0, index)
    # Put right nodes (workers) at x=1, spaced by their index
    for index, node in enumerate(right):
        pos[node] = (1, index - n)

    # Draw the graph with node labels
    plt.figure()
    nx.draw(G, pos, with_labels=True)
    
    # Draw edge labels showing weights (the cost)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show the graph in a pop-up window (or you can savefig instead)
    plt.title("Bipartite Graph Representation of Cost Matrix")
    plt.show()
    # -------------------------------------------------------------

    # Now run the Hungarian (minimum weight full matching) algorithm
    start = time.time()
    matching = nx.algorithms.bipartite.minimum_weight_full_matching(
        G, top_nodes=left, weight='weight'
    )
    end = time.time()

    # Compute the total cost from the matching
    total_cost = 0
    for i in left:
        j = matching[i]
        cost = cost_matrix[i][j - n]
        total_cost += cost
    print("Hungarian total cost:", total_cost)

    return total_cost, end - start

# Example of how you might call this:
if __name__ == "__main__":
    # A sample 4x4 cost matrix (row=Job, col=Worker)
    cost_matrix = [
        [52, 89, 40, 77],
        [96, 92, 76, 33],
        [31, 71,  6, 20],
        [93, 70, 63, 95]
    ]
    hungarian_algorithm(cost_matrix)
