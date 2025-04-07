import time
import networkx as nx
import matplotlib.pyplot as plt
from erotima1 import read_file, assignment_problem_solver

def hungarian_algorithm(cost_matrix):
    n = len(cost_matrix)
    G = nx.Graph()
    #left = job
    left = range(n)
    #right = worker
    right = range(n, 2*n)
    print(left,right)

    for i in left:
        for j in right:
            print(i, j, cost_matrix[i][j - n])
            G.add_edge(i, j, weight=cost_matrix[i][j - n])
    start = time.time()
    matching = nx.algorithms.bipartite.minimum_weight_full_matching(G, top_nodes=left, weight='weight')
    end = time.time()

    total_cost = 0
    for i in left:
        j = matching[i]
        cost = cost_matrix[i][j - n]
        total_cost += cost
    return total_cost, end - start


def run_comparison(files):
    sizes, costs_lp, times_lp, costs_nx, times_nx = [], [], [], [], []

    for file in files:
        print(f"Processing {file}")
        matrix = read_file(file)
        n = len(matrix)
        sizes.append(n)

        # ortools
        cost_lp, _, time_lp = assignment_problem_solver(matrix)
        costs_lp.append(cost_lp)
        times_lp.append(time_lp)

        # hungarian
        cost_nx, time_nx = hungarian_algorithm(matrix)
        costs_nx.append(cost_nx)
        times_nx.append(time_nx)

        print(f"For {file}: ortools time = {time_lp:.2f} sec, hungarian algo time = {time_nx:.2f} sec")
    return sizes, costs_lp, times_lp, costs_nx, times_nx


def plot_comparisons(sizes, costs_lp, times_lp, costs_nx, times_nx):
    # Cost
    plt.figure()
    plt.plot(sizes, costs_lp, label='OR-Tools', marker='o')
    plt.plot(sizes, costs_nx, label='Hungarian Algorithm', marker='x')
    plt.xlabel('Μέγεθος Προβλήματος (n)')
    plt.ylabel('Κόστος Λύσης')
    plt.title('Σύγκριση Κόστους')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_img/cost_comparison.png")

    # Time
    plt.figure()
    plt.plot(sizes, times_lp, label='OR-Tools', marker='o')
    plt.plot(sizes, times_nx, label='Hungarian Algorithm', marker='x')
    plt.xlabel('Μέγεθος Προβλήματος (n)')
    plt.ylabel('Χρόνος (sec)')
    plt.title('Σύγκριση Χρόνου Επίλυσης')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_img/time_comparison.png")

    plt.show()


def main():
    files = ["dataset/assign5.txt"]
    # files = [
    #     "dataset/assign100.txt", "dataset/assign200.txt", "dataset/assign300.txt",
    #     "dataset/assign400.txt", "dataset/assign500.txt", "dataset/assign600.txt",
    #     "dataset/assign700.txt", "dataset/assign800.txt"
    # ]

    sizes, costs_lp, times_lp, costs_nx, times_nx = run_comparison(files)
    plot_comparisons(sizes, costs_lp, times_lp, costs_nx, times_nx)


if __name__ == "__main__":
    main()
