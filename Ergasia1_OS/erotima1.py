import time
import numpy as np
from ortools.linear_solver import pywraplp

def read_assignment_file(filename):
    """ Διαβάζει τα δεδομένα από ένα αρχείο ανάθεσης και επιστρέφει τον πίνακα κόστους """
    with open(filename, 'r') as file:
        lines = file.readlines()

    n = int(lines[0].strip())  # Πρώτη γραμμή: πλήθος εργασιών/εργαζομένων
    cost_matrix = []


    all_costs = [int(num) for line in lines[1:] for num in line.split()]
    
    for i in range(n):
        cost_matrix.append(all_costs[i * n:(i + 1) * n])  # Δημιουργία μήτρας κόστους
    print(np.array(cost_matrix))
    return np.array(cost_matrix)

def solve_assignment_problem(cost_matrix):
    """ Χρησιμοποιεί Google OR-Tools για την εύρεση της βέλτιστης ανάθεσης """
    n = cost_matrix.shape[0]
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Solver not found!")

    # Μεταβλητές απόφασης: x[i][j] = 1 αν η εργασία i ανατεθεί στον εργαζόμενο j, αλλιώς 0
    x = []
    for i in range(n):
        x.append([solver.BoolVar(f'x[{i},{j}]') for j in range(n)])

    # Περιορισμοί: Κάθε εργασία ανατίθεται σε έναν μόνο εργαζόμενο
    for i in range(n):
        solver.Add(sum(x[i][j] for j in range(n)) == 1)

    # Περιορισμοί: Κάθε εργαζόμενος αναλαμβάνει μία μόνο εργασία
    for j in range(n):
        solver.Add(sum(x[i][j] for i in range(n)) == 1)

    # Αντικειμενική συνάρτηση: Ελαχιστοποίηση κόστους ανάθεσης
    solver.Minimize(solver.Sum(cost_matrix[i, j] * x[i][j] for i in range(n) for j in range(n)))

    # Επίλυση
    start_time = time.time()
    status = solver.Solve()
    end_time = time.time()

    if status == pywraplp.Solver.OPTIMAL:
        total_cost = solver.Objective().Value()
        assignment = [(i, j, cost_matrix[i, j]) for i in range(n) for j in range(n) if x[i][j].solution_value() == 1]
        return total_cost, assignment, end_time - start_time
    else:
        raise Exception("No optimal solution found!")

def save_solution(filename, total_cost, assignment, elapsed_time):
    """ Αποθηκεύει τη λύση σε αρχείο """
    output_file = filename.replace(".txt", "_solution.txt")
    with open(output_file, 'w') as file:
        file.write(f"{int(total_cost)}\n")
        for task, worker, cost in assignment:
            file.write(f"{task},{worker},{cost}\n")
        file.write(f"\nSolution Time: {elapsed_time:.4f} seconds\n")

def main():
    """ Κύρια συνάρτηση που διαβάζει τα αρχεία, επιλύει τα προβλήματα και αποθηκεύει τις λύσεις """
    instances = [100, 200, 300, 400, 500, 600, 700, 800]  # Διαστάσεις προβλημάτων
    base_url = "dataset/assign"  # Βάση ονομάτων αρχείων
    
    for size in instances:
        filename = f"{base_url}{size}.txt"
        print(f"Solving {filename}...")
        
        try:
            cost_matrix = read_assignment_file(filename)
            total_cost, assignment, elapsed_time = solve_assignment_problem(cost_matrix)
            save_solution(filename, total_cost, assignment, elapsed_time)
            print(f"✅ {filename} solved! Cost: {int(total_cost)}, Time: {elapsed_time:.4f} sec")
        except Exception as e:
            print(f"❌ Error solving {filename}: {e}")

if __name__ == "__main__":
    main()
