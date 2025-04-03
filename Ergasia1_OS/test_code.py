#!/usr/bin/env python3
"""
erotima1.py

Solves the assignment problem using Google OR-Tools.
For each input file (assign100.txt, assign200.txt, …, assign800.txt),
it reads the cost matrix, solves the model, and writes the solution to an output file.
The output file has the solution cost on the first line and, for each task,
a line with: task number, worker number, assignment cost.
"""

import time
from ortools.linear_solver import pywraplp

def read_cost_matrix(filename):
    with open(filename, 'r') as f:
        # Split all tokens (the file first token is n, then n*n cost values)
        tokens = f.read().split()
    n = int(tokens[0])
    if len(tokens[1:]) != n * n:
        raise ValueError(f"Expected {n*n} cost values, but got {len(tokens[1:])} in file {filename}.")
    costs = list(map(int, tokens[1:]))
    cost_matrix = [costs[i*n:(i+1)*n] for i in range(n)]
    return cost_matrix

def solve_assignment(cost_matrix):
    n = len(cost_matrix)
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        raise Exception("Solver not created. Check OR-Tools installation.")
        
    # Create binary variables: x[i][j] == 1 if task i is assigned to worker j.
    x = [[solver.BoolVar(f'x[{i},{j}]') for j in range(n)] for i in range(n)]
    
    # Each task is assigned to exactly one worker.
    for i in range(n):
        solver.Add(sum(x[i][j] for j in range(n)) == 1)
    
    # Each worker is assigned to exactly one task.
    for j in range(n):
        solver.Add(sum(x[i][j] for i in range(n)) == 1)
    
    # Objective: minimize total cost.
    objective = solver.Objective()
    for i in range(n):
        for j in range(n):
            objective.SetCoefficient(x[i][j], cost_matrix[i][j])
    objective.SetMinimization()
    
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        raise Exception("No feasible solution found.")
    
    solution_cost = solver.Objective().Value()
    assignments = []
    for i in range(n):
        for j in range(n):
            if x[i][j].solution_value() > 0.5:
                assignments.append((i, j, cost_matrix[i][j]))
    return solution_cost, assignments

def main():
    # List of instance files to solve.
    files = ["dataset/assign5.txt","dataset/assign100.txt", "dataset/assign200.txt", "dataset/assign300.txt", "dataset/assign400.txt", 
             "dataset/assign500.txt", "dataset/assign600.txt", "dataset/assign700.txt", "dataset/assign800.txt"]
    
    for file in files:
        print(f"Solving instance from file: {file}")
        cost_matrix = read_cost_matrix(file)
        start_time = time.time()
        sol_cost, assignments = solve_assignment(cost_matrix)
        elapsed = time.time() - start_time
        print(f"Solution cost: {sol_cost}, Time: {elapsed:.4f} seconds")
        out_file = file.replace(".txt", "_sol.txt")
        with open(out_file, 'w') as fout:
            fout.write(f"{sol_cost}\n")
            for task, worker, cost in assignments:
                fout.write(f"{task},{worker},{cost}\n")

if __name__ == "__main__":
    main()
