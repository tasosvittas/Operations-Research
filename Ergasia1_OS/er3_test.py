import os
import time
import numpy as np
from ortools.linear_solver import pywraplp
from erotima1 import read_file, write_solution

def assignment_groups_solution(jobs_matrix):
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
        solver.Add(solver.Sum([x[i, j] for j in range(n)]) <= 1)
    for j in range(n):
        solver.Add(solver.Sum([x[i, j] for i in range(n)]) == 1)

    for start in range(0, n, 5):
        jobs = range(start, min(start + 5, n))
        workers = range(start, min(start + 5, n))
        assignments = []
        for i in jobs:
            for j in workers:
                assignments.append(x[i,j])
        solver.Add(solver.Sum(assignments) >=2)


def main():
    files = [
        "dataset/assign4.txt",
        "dataset/assign100.txt", "dataset/assign200.txt", "dataset/assign300.txt",
        "dataset/assign400.txt", "dataset/assign500.txt", "dataset/assign600.txt",
        "dataset/assign700.txt", "dataset/assign800.txt"
    ]

    for file in files:
        jobs_matrix = read_file(file)
