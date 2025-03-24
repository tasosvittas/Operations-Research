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
        print(all_costs)
    for i in range(n):
        matrix.append(all_costs[i * n:(i + 1) * n])
    return np.array(matrix)
    

def main():
    filename = "dataset/assign100.txt"
    matrix = read_file(filename)


if __name__ == "__main__":
    main()