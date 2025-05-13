from ortools.sat.python import cp_model
from read_dataset import load_data
import pandas as pd

def cpsat_solver():
    demand_nodes, truck_assignments, problem_data = load_data()
    model = cp_model.CpModel()

    #parametroi
    burrito_price = problem_data['burrito_price'][0]
    ingredient_cost = problem_data['ingredient_cost'][0]
    truck_cost = problem_data['truck_cost'][0]
    print(burrito_price, ingredient_cost, truck_cost)


if __name__ == '__main__':
    cpsat_solver()