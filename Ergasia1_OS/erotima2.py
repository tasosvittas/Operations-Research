import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from erotima1 import read_file, assignment_problem_solver

def run_comparison(files):
    sizes, costs_lp, times_lp, costs_hungarian, times_hungarian = [], [], [], [], []

    for file in files:
        print(f"Processing {file}")
        jobs_matrix = read_file(file)
        size = jobs_matrix.shape[0]
        sizes.append(size)

        # Μέθοδος 1: Μαθηματική Μοντελοποίηση
        cost_lp, _, time_lp = assignment_problem_solver(jobs_matrix)
        costs_lp.append(cost_lp)
        times_lp.append(time_lp)

        # Μέθοδος 2: Ουγγρικός Αλγόριθμος
        start = time.time()
        row_ind, col_ind = linear_sum_assignment(jobs_matrix)
        end = time.time()
        cost_hungarian = jobs_matrix[row_ind, col_ind].sum()
        time_hungarian = end - start
        costs_hungarian.append(cost_hungarian)
        times_hungarian.append(time_hungarian)

    return sizes, costs_lp, times_lp, costs_hungarian, times_hungarian


def plot_comparisons(sizes, costs_lp, times_lp, costs_hungarian, times_hungarian):
    # Κόστος σύγκριση
    plt.figure()
    plt.plot(sizes, costs_lp, label='LP Solver', marker='o')
    plt.plot(sizes, costs_hungarian, label='Hungarian Algorithm', marker='x')
    plt.xlabel('Size (n)')
    plt.ylabel('Total Cost')
    plt.title('Σύγκριση Κόστους Λύσης')
    plt.legend()
    plt.grid(True)
    plt.savefig("cost_comparison.png")

    # Χρόνος σύγκριση
    plt.figure()
    plt.plot(sizes, times_lp, label='LP Solver', marker='o')
    plt.plot(sizes, times_hungarian, label='Hungarian Algorithm', marker='x')
    plt.xlabel('Size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Σύγκριση Χρόνου Επίλυσης')
    plt.legend()
    plt.grid(True)
    plt.savefig("time_comparison.png")

    plt.show()


def main():
    files = [
        "dataset/assign100.txt", "dataset/assign200.txt", "dataset/assign300.txt",
        "dataset/assign400.txt", "dataset/assign500.txt", "dataset/assign600.txt",
        "dataset/assign700.txt", "dataset/assign800.txt"
    ]
    sizes, costs_lp, times_lp, costs_hungarian, times_hungarian = run_comparison(files)
    plot_comparisons(sizes, costs_lp, times_lp, costs_hungarian, times_hungarian)

if __name__ == "__main__":
    main()
