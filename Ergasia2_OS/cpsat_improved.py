from ortools.sat.python import cp_model
import pandas as pd

def load_data(day=1):
    """Load input data files for the specified day"""
    base_path = 'burrito_dataset/scenDE4BD4_round1_day{}_'
    files = {
        'demand': base_path + 'demand_node_data.csv',
        'trucks': base_path + 'demand_truck_data.csv',
        'problem': base_path + 'problem_data.csv'
    }
    return (
        pd.read_csv(files['demand'].format(day)),
        pd.read_csv(files['trucks'].format(day)),
        pd.read_csv(files['problem'].format(day))
    )

def solve_burrito_assignment():
    # Load and prepare data
    demands, trucks, problem = load_data(day=1)
    MAX_CAPACITY = 200
    
    # Extract business parameters
    price = problem['burrito_price'][0]
    cost = problem['ingredient_cost'][0]
    truck_cost = problem['truck_cost'][0]
    profit_per_unit = price - cost
    
    # Filter to only feasible truck-demand pairs
    feasible = trucks[trucks['scaled_demand'] > 0]
    
    # Initialize model
    model = cp_model.CpModel()
    
    # Decision Variables
    is_truck_used = {
        truck: model.NewBoolVar(f'use_truck_{truck}') 
        for truck in trucks['truck_node_index'].unique()
    }
    print("blaaaaa", is_truck_used)
    
    assignments = {
        (row['demand_node_index'], row['truck_node_index']): 
        model.NewBoolVar(f'assign_{row["demand_node_index"]}_to_{row["truck_node_index"]}')
        for _, row in feasible.iterrows()
    }
    print("ssss",assignments)
    # Constraints
    # 1. Each demand served by at most one truck
    for demand in demands['index'].unique():
        possible_trucks = [var for (d, t), var in assignments.items() if d == demand]
        if possible_trucks:
            model.Add(sum(possible_trucks) <= 1)
    
    # 2. Link assignments to truck usage
    for (d, t), var in assignments.items():
        model.Add(var <= is_truck_used[t])
    
    # 3. Respect truck capacity
    for truck in is_truck_used:
        assigned_demand = sum(
            row['scaled_demand'] * assignments[(row['demand_node_index'], truck)]
            for _, row in feasible[feasible['truck_node_index'] == truck].iterrows()
        )
        model.Add(assigned_demand <= MAX_CAPACITY)
    
    # Calculate objective components
    total_units_expr = sum(
        row['scaled_demand'] * var
        for (d, t), var in assignments.items()
        for _, row in feasible[(feasible['demand_node_index'] == d) & 
                              (feasible['truck_node_index'] == t)].iterrows()
    )
    
    total_profit = profit_per_unit * total_units_expr - sum(
        is_truck_used[t] * truck_cost for t in is_truck_used
    )
    
    # Solve
    model.Maximize(total_profit)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Output results
    if status == cp_model.OPTIMAL:
        # Calculate actual values from the solution
        total_units = sum(
            row['scaled_demand'] * solver.Value(var)
            for (d, t), var in assignments.items()
            for _, row in feasible[(feasible['demand_node_index'] == d) & 
                                  (feasible['truck_node_index'] == t)].iterrows()
            if solver.Value(var)
        )
        
        print_results(solver, is_truck_used, assignments, feasible, 
                     total_units, price, cost, truck_cost)
    else:
        print("No optimal solution found.")

def print_results(solver, trucks, assignments, feasible, units, price, cost, truck_cost):
    """Display the solution in a readable format"""
    print(f"Optimal Profit: €{solver.ObjectiveValue():.2f}\n")
    
    active_trucks = [t for t in trucks if solver.Value(trucks[t])]
    print(f"Active Trucks ({len(active_trucks)}): {sorted(active_trucks)}\n")
    
    print("Assignments:")
    for truck in active_trucks:
        print(f"\nTruck {truck}:")
        truck_units = 0
        for (d, t), var in assignments.items():
            if t == truck and solver.Value(var):
                amount = feasible[
                    (feasible['demand_node_index'] == d) & 
                    (feasible['truck_node_index'] == t)
                ]['scaled_demand'].values[0]
                print(f"  - Demand node {d}: {amount} burritos")
                truck_units += amount
        print(f"  TOTAL: {truck_units} burritos")
    
    print(f"\nSummary:")
    print(f"- Total burritos sold: {units}")
    print(f"- Revenue: €{units * price:.2f}")
    print(f"- Ingredient cost: €{units * cost:.2f}")
    print(f"- Truck costs: €{len(active_trucks) * truck_cost:.2f}")

if __name__ == '__main__':
    solve_burrito_assignment()