from ortools.linear_solver import pywraplp
import time

def read_cost_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_tasks = int(lines[0].strip())
        costs = []
        for line in lines[1:]:
            costs.extend(map(int, line.strip().split()))
        cost_matrix = [costs[i*num_tasks:(i+1)*num_tasks] for i in range(num_tasks)]
        return cost_matrix

def solve_assignment_problem(cost_matrix):
    num_workers = len(cost_matrix)
    num_tasks = len(cost_matrix[0])

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, None

    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(cost_matrix[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    start_time = time.time()
    status = solver.Solve()
    end_time = time.time()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_cost = solver.Objective().Value()
        assignments = []
        for i in range(num_workers):
            for j in range(num_tasks):
                if x[i, j].solution_value() > 0.5:
                    assignments.append((i, j, cost_matrix[i][j]))
        return total_cost, assignments, end_time - start_time
    else:
        return None, None, end_time - start_time

def write_solution(file_path, total_cost, assignments):
    with open(file_path, 'w') as file:
        file.write(f"{total_cost}\n")
        for assignment in assignments:
            file.write(f"{assignment[0]},{assignment[1]},{assignment[2]}\n")

def main():
    problem_files = [
        "dataset/assign5.txt",
        "dataset/assign100.txt", "dataset/assign200.txt", "dataset/assign300.txt",
        "dataset/assign400.txt", "dataset/assign500.txt", "dataset/assign600.txt",
        "dataset/assign700.txt", "dataset/assign800.txt"
    ]

    for problem_file in problem_files:
        cost_matrix = read_cost_matrix(problem_file)
        total_cost, assignments, solve_time = solve_assignment_problem(cost_matrix)
        if total_cost is not None:
            solution_file = problem_file.replace(".txt", "_solution.txt")
            write_solution(solution_file, total_cost, assignments)
            print(f"Solved {problem_file}: Total Cost = {total_cost}, Time = {solve_time:.2f} seconds")
        else:
            print(f"No solution found for {problem_file}")

if __name__ == "__main__":
    main()