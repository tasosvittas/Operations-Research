from ortools.linear_solver import pywraplp
import time

def read_cost_matrix(filename):
    with open(filename, 'r') as f:
        data = list(map(int, f.read().split()))
    n = data[0]
    costs = data[1:]
    return [costs[i*n:(i+1)*n] for i in range(n)]

def solve_assignment(cost_matrix):
    n = len(cost_matrix)
    solver = pywraplp.Solver.CreateSolver('CBC')
    x = [[solver.BoolVar('x[%d][%d]' % (i, j)) for j in range(n)] for i in range(n)]
    
    for i in range(n):
        solver.Add(solver.Sum(x[i][j] for j in range(n)) == 1)
    for j in range(n):
        solver.Add(solver.Sum(x[i][j] for i in range(n)) == 1)

    solver.Minimize(solver.Sum(cost_matrix[i][j] * x[i][j] for i in range(n) for j in range(n)))
    solver.Solve()

    total_cost = solver.Objective().Value()
    assignments = [(i, j, cost_matrix[i][j]) for i in range(n) for j in range(n) if x[i][j].solution_value() > 0.5]
    return total_cost, assignments

def main():
    files = [
        "dataset/assign4.txt", "dataset/assign100.txt", "dataset/assign200.txt",
        "dataset/assign300.txt", "dataset/assign400.txt", "dataset/assign500.txt",
        "dataset/assign600.txt", "dataset/assign700.txt", "dataset/assign800.txt"
    ]

    for file in files:
        print(f"Solving: {file}")
        matrix = read_cost_matrix(file)
        start = time.time()
        cost, assignments = solve_assignment(matrix)
        print(f"Cost: {int(cost)}, Time: {time.time() - start:.2f}s")

        with open(file.replace(".txt", "_sol.txt"), "w") as f:
            f.write(f"{int(cost)}\n")
            for task, worker, c in assignments:
                f.write(f"{task},{worker},{c}\n")

if __name__ == "__main__":
    main()
