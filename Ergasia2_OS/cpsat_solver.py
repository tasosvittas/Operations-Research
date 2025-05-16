from ortools.sat.python import cp_model
import pandas as pd

def load_data():
    # demand_nodes = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day5_demand_node_data.csv')
    # truck_assignments = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day5_demand_truck_data.csv')
    # problem_data = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day5_problem_data.csv')
    demand_nodes = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_demand_node_data.csv')
    truck_assignments = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_demand_truck_data.csv')
    problem_data = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_problem_data.csv')
    return demand_nodes, truck_assignments, problem_data

def solve_with_cpsat():
    demand_nodes, truck_assignments, problem_data = load_data()
    
    model = cp_model.CpModel()
    
    # Parameters
    burrito_price = problem_data['burrito_price'][0]
    ingredient_cost = problem_data['ingredient_cost'][0]
    truck_cost = problem_data['truck_cost'][0]
    max_truck_capacity = 200  # Added capacity constraint
    
    # Prepare data
    all_demands = demand_nodes['index'].unique()
    all_trucks = truck_assignments['truck_node_index'].unique()
    
    # Create assignment dictionary with available units
    feasible_assignments = truck_assignments[truck_assignments['scaled_demand'] > 0]
    
    # Decision variables
    truck_active = {truck: model.NewBoolVar(f'truck_{truck}') for truck in all_trucks}
    print(truck_active)
    assign_vars = {
        (row['demand_node_index'], row['truck_node_index']): model.NewBoolVar(f'assign_{row["demand_node_index"]}_{row["truck_node_index"]}')
        for _, row in feasible_assignments.iterrows()
    }
    print("s",assign_vars)

    # Constraints
    # 1. Each demand served by at most one truck
    for demand in all_demands:
        relevant_assignments = [var for (d, t), var in assign_vars.items() if d == demand]
        if relevant_assignments:
            model.Add(sum(relevant_assignments) <= 1)
    
    # 2. Truck must be active if serving any demand
    for (d, t), var in assign_vars.items():
        model.Add(var <= truck_active[t])
    
    # 3. Truck capacity constraint
    for truck in all_trucks:
        truck_demand = sum(
            row['scaled_demand'] * assign_vars[(row['demand_node_index'], truck)]
            for _, row in feasible_assignments[feasible_assignments['truck_node_index'] == truck].iterrows()
        )
        model.Add(truck_demand <= max_truck_capacity)
    
    # Calculate components for objective
    total_units = sum(
        row['scaled_demand'] * var
        for (d, t), var in assign_vars.items()
        for _, row in feasible_assignments[(feasible_assignments['demand_node_index'] == d) & 
                                         (feasible_assignments['truck_node_index'] == t)].iterrows()
    )
    
    total_truck_costs = sum(truck_active[t] * truck_cost for t in all_trucks)
    
    # Objective: Maximize profit = (price - cost)*units - truck_costs
    model.Maximize((burrito_price - ingredient_cost) * total_units - total_truck_costs)
    
    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Output results
    if status == cp_model.OPTIMAL:
        print(f'Optimal profit: {solver.ObjectiveValue()}')
        print('\nActive trucks:')
        active_trucks = [t for t in all_trucks if solver.Value(truck_active[t])]
        print(active_trucks)
        
        print('\nAssignments:')
        total_units = 0
        for truck in active_trucks:
            print(f'\n{truck}:')
            truck_units = 0
            for (d, t), var in assign_vars.items():
                if t == truck and solver.Value(var):
                    units = feasible_assignments[
                        (feasible_assignments['demand_node_index'] == d) & 
                        (feasible_assignments['truck_node_index'] == t)
                    ]['scaled_demand'].values[0]
                    print(f'  {d} -> {units} units')
                    truck_units += units
            print(f'  Total: {truck_units} units')
            total_units += truck_units
        
        print(f'\nTotal units sold: {total_units}')
        print(f'Revenue: €{total_units * burrito_price}')
        print(f'Ingredient costs: €{total_units * ingredient_cost}')
        print(f'Truck costs: €{len(active_trucks) * truck_cost}')
    else:
        print('No optimal solution found.')

if __name__ == '__main__':
    solve_with_cpsat()