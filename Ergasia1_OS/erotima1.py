import time
import numpy as np
from ortools.linear_solver import pywraplp

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    n = int(lines[0].strip())
    matrix = []
    all_costs = []
    for line in lines[1:]:
        numbers = line.split()          # Χωρίζει τη γραμμή σε ξεχωριστούς αριθμούς (ως strings)
        for num in numbers:
            all_costs.append(int(num))  # Μετατρέπει κάθε αριθμό σε int και τον προσθέτει στη λίστα
    for i in range(n):
        matrix.append(all_costs[i * n:(i + 1) * n])
    return np.array(matrix)

def assignment_problem_solver(jobs_matrix):
    workers = len(jobs_matrix)
    jobs = len(jobs_matrix[0])
    
    solver =  pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return
    
    x = {}
    for i in range(workers):
        for j in range(jobs):
            x[i, j] = solver.IntVar(0, 1, "")

    for i in range(workers):
        solver.Add(solver.Sum([x[i, j] for j in range(jobs)]) <= 1)
    for j in range(jobs):
        solver.Add(solver.Sum([x[i, j] for i in range(workers)]) == 1)


    objective_terms = []
    for i in range(workers):
        for j in range(jobs):
            objective_terms.append(jobs_matrix[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    start_time = time.time()
    status = solver.Solve()
    end_time = time.time()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_cost = solver.Objective().Value()
        assignments = []
        for i in range(workers):
            for j in range(jobs):
                if x[i,j].solution_value() > 0.5:
                    assignments.append((i, j, jobs_matrix[i][j]))
        return total_cost, assignments, end_time-start_time
    else:
        return None, None, end_time - start_time


def write_solution(file_path, total_cost, assignments):
    with open(file_path, 'w') as file:
        file.write(f"{total_cost}\n")
        for assignment in assignments:
            file.write(f"{assignment[0]},{assignment[1]},{assignment[2]}\n")

def main():
    files = [
        "dataset/assign5.txt",
        "dataset/assign100.txt", "dataset/assign200.txt", "dataset/assign300.txt",
        "dataset/assign400.txt", "dataset/assign500.txt", "dataset/assign600.txt",
        "dataset/assign700.txt", "dataset/assign800.txt"
    ]

    for file in files:
        jobs_matrix = read_file(file)
        total_cost, assignments, solve_time = assignment_problem_solver(jobs_matrix)
        solution_file = file.replace(".txt", "_solution.txt")
        write_solution(solution_file, total_cost, assignments)
        print(f"Solved {file}: Total Cost = {total_cost}, Time = {solve_time:.2f} seconds")

if __name__ == "__main__":
    main()