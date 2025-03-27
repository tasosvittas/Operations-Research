import time
import numpy as np
from ortools.linear_solver import pywraplp
from erotima1 import read_file, write_solution

def assignment_with_group_constraint(jobs_matrix):
    n = len(jobs_matrix)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, None, 0

    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

    # Περιορισμοί ανάθεσης
    for i in range(n):
        solver.Add(solver.Sum([x[i, j] for j in range(n)]) == 1)
    for j in range(n):
        solver.Add(solver.Sum([x[i, j] for i in range(n)]) <= 1)

    # ΝΕΟΙ Περιορισμοί ομάδων κάθε 5 εργασιών/εργαζομένων
    for start in range(0, n, 5):
        group_jobs = list(range(start, min(start + 5, n)))
        group_workers = list(range(start, min(start + 5, n)))
        group_assignments = []
        for i in group_jobs:
            for j in group_workers:
                group_assignments.append(x[i, j])
        solver.Add(solver.Sum(group_assignments) >= 2)

    # Στόχος: ελαχιστοποίηση κόστους
    objective_terms = [jobs_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)]
    solver.Minimize(solver.Sum(objective_terms))

    # Επίλυση
    start = time.time()
    status = solver.Solve()
    end = time.time()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_cost = solver.Objective().Value()
        assignments = [(i, j, jobs_matrix[i][j]) for i in range(n) for j in range(n) if x[i, j].solution_value() > 0.5]
        return total_cost, assignments, end - start
    else:
        return None, None, end - start


def main():
    files = [
        "dataset/assign100.txt", "dataset/assign200.txt", "dataset/assign300.txt",
        "dataset/assign400.txt", "dataset/assign500.txt", "dataset/assign600.txt",
        "dataset/assign700.txt", "dataset/assign800.txt"
    ]

    for file in files:
        jobs_matrix = read_file(file)
        total_cost, assignments, solve_time = assignment_with_group_constraint(jobs_matrix)
        solution_file = file.replace(".txt", "_group_solution.txt")
        write_solution(solution_file, total_cost, assignments)
        print(f"[Group Constraint] Solved {file}: Total Cost = {total_cost}, Time = {solve_time:.2f} sec")

if __name__ == "__main__":
    main()
