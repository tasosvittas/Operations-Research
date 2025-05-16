from ortools.sat.python import cp_model
from read_dataset import load_data
import pandas as pd

def cpsat_solver():
    demand_nodes, truck_assignments, problem_data = load_data()
    model = cp_model.CpModel()

    #parametroi
    price = problem_data['burrito_price'][0]
    cost = problem_data['ingredient_cost'][0]
    truck_cost = problem_data['truck_cost'][0]
    profit_per_unit = price - cost

    demands = demand_nodes['index'].unique()
    trucks = truck_assignments['truck_node_index'].unique()
    
    feasible = trucks[trucks['scaled_demand'] > 0]

    truck_active = {}
    for truck in trucks:
        truck_active[truck] = model.NewBoolVar(f"truck_{truck}")

    assignments = {
        (row['demand_node_index'], row['truck_node_index']): 
        model.NewBoolVar(f'assign_{row["demand_node_index"]}_to_{row["truck_node_index"]}')
        for _, row in feasible.iterrows()
    }

    print(assignments)


if __name__ == '__main__':
    cpsat_solver()