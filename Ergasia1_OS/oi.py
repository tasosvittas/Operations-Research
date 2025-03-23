from ortools.linear_solver import pywraplp
import time

def solve_assignment_problem(input_file, output_file):
    """
    Διαβάζει τα δεδομένα από το αρχείο input_file,
    επιλύει το πρόβλημα ανάθεσης με τη βιβλιοθήκη OR-Tools
    και γράφει τη λύση στο αρχείο output_file.
    """

    # =============== 1. Φόρτωση Δεδομένων ===============
    with open(input_file, 'r') as f:
        # Πρώτη γραμμή: αριθμός εργασιών και εργαζομένων (n)
        n = int(f.readline().strip())

        # Διαβάζουμε τις υπόλοιπες τιμές σε μία λίστα (tokens) ώσπου να έχουμε n*n στοιχεία
        tokens = []
        for line in f:
            tokens.extend(line.split())

        # Mετατρέπουμε τα tokens σε ακέραιους και τα περνάμε σε έναν πίνακα n x n (costs)
        costs = []
        index = 0
        for i in range(n):
            row_costs = list(map(int, tokens[index:index + n]))
            costs.append(row_costs)
            index += n

    # =============== 2. Δημιουργία Solver ===============
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Πρόβλημα κατά τη δημιουργία του solver.")
        return

    # Μετράμε χρόνο από εδώ
    start_time = time.time()

    # =============== 3. Ορισμός Μεταβλητών x[i, j] = 0 ή 1 ===============
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = solver.IntVar(0, 1, f"x[{i},{j}]")

    # =============== 4. Περιορισμοί ===============
    # 4.1. Κάθε εργαζόμενος i μπορεί να αναλάβει το πολύ μία εργασία.
    for i in range(n):
        solver.Add(solver.Sum(x[i, j] for j in range(n)) <= 1)

    # 4.2. Κάθε εργασία j πρέπει να ανατεθεί ακριβώς σε έναν εργαζόμενο.
    for j in range(n):
        solver.Add(solver.Sum(x[i, j] for i in range(n)) == 1)

    # =============== 5. Συνάρτηση Στόχου ===============
    # Ελαχιστοποίηση του συνολικού κόστους
    objective_terms = []
    for i in range(n):
        for j in range(n):
            objective_terms.append(costs[i][j] * x[i, j])

    solver.Minimize(solver.Sum(objective_terms))

    # =============== 6. Επίλυση ===============
    status = solver.Solve()
    end_time = time.time()

    # =============== 7. Εμφάνιση & Καταγραφή Αποτελεσμάτων ===============
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_cost = solver.Objective().Value()
        solve_time = end_time - start_time

        print(f"Αρχείο: {input_file}")
        print(f"Συνολικό κόστος = {total_cost}")
        print(f"Χρόνος επίλυσης = {solve_time:.4f} sec")

        # Γράφουμε τα αποτελέσματα σε αρχείο εξόδου
        with open(output_file, 'w') as f_out:
            # Πρώτη γραμμή: συνολικό κόστος
            f_out.write(f"{int(total_cost)}\n")

            # Στις υπόλοιπες γραμμές: εργασία, εργαζόμενος, κόστος ανάθεσης
            for j in range(n):
                for i in range(n):
                    if x[i, j].solution_value() > 0.5:
                        cost_ij = costs[i][j]
                        # Γράφουμε: εργασία j, εργαζόμενος i, κόστος cost_ij
                        f_out.write(f"{j},{i},{cost_ij}\n")
                        break  # Αφού βρέθηκε ο μοναδικός i για το j
    else:
        print("Δεν βρέθηκε εφικτή λύση ή παρουσιαστηκε πρόβλημα στον solver.")


def main():
    # Παράδειγμα κλήσης για τα αρχεία assign100, assign200, κ.ο.κ.
    # Προσαρμόστε τις διαδρομές (paths) ανάλογα με το πού έχετε αποθηκεύσει τα αρχεία.
    
    # Λίστα με τα μεγέθη που θέλουμε να λύσουμε
    sizes = [100, 200, 300, 400, 500, 600, 700, 800]
    
    for sz in sizes:
        input_file = f"dataset/assign{sz}.txt"      # π.χ. "assign100.txt"
        output_file = f"solution{sz}.txt"   # π.χ. "solution100.txt"
        solve_assignment_problem(input_file, output_file)


if __name__ == "__main__":
    main()
