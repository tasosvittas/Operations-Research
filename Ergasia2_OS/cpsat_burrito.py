from ortools.sat.python import cp_model
from read_dataset import load_data
import pandas as pd

def cpsat_solver(day, total_profit):
    demand_nodes, truck_assignments, problem_data = load_data(day)
    model = cp_model.CpModel()

    #parametroi
    price = problem_data['burrito_price'][0]
    cost = problem_data['ingredient_cost'][0]
    truck_cost = problem_data['truck_cost'][0]

    demands = demand_nodes['index'].unique()
    trucks = truck_assignments['truck_node_index'].unique()
    
    feasible_assignments = truck_assignments[truck_assignments['scaled_demand'] > 0]

    #boolean gia na kserw an h kantina einai active h oxi 
    truck_active = {}
    for truck in trucks:
        truck_active[truck] = model.NewBoolVar(f"truck_{truck}")
    
    #boolean gia na kserw an h syndesi demand kai kantina einai ok
    assignments = {} 
    for _, row in feasible_assignments.iterrows():
        demand = row['demand_node_index']
        truck = row['truck_node_index']
        var = model.NewBoolVar(f'assign_{demand}_{truck}')
        assignments[(demand, truck)] = var

    #kathe demand kalyptetai mono apo mia kantina
    for demand in demands:
        relevant_assignments = []
        for (d, t), var in assignments.items():
            if d == demand:
                relevant_assignments.append(var)

        if relevant_assignments:
            model.Add(sum(relevant_assignments) <= 1)


    #elegxos gia kantina active or not
    for (d, t), var in assignments.items():
        model.Add(var <= truck_active[t])

    total_units = sum(
        row['scaled_demand'] * var
        for (d, t), var in assignments.items()
        for _, row in feasible_assignments[(feasible_assignments['demand_node_index'] == d) & 
                                         (feasible_assignments['truck_node_index'] == t)].iterrows()
    )

    total_truck_costs = sum(truck_active[t] * truck_cost for t in trucks)
    
    # Maximize profit = (price - cost)*units - truck_costs
    model.Maximize((price - cost) * total_units - total_truck_costs)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print(f'\nDay {day}')
        profit = solver.ObjectiveValue()
        print(f'Profit: €{profit:.2f}')

        active_trucks = [t for t in trucks if solver.Value(truck_active[t])]
        print('Active trucks:', active_trucks)

        total_units = 0
        for (d, t), var in assignments.items():
            if solver.Value(var):
                units = feasible_assignments[
                    (feasible_assignments['demand_node_index'] == d) & 
                    (feasible_assignments['truck_node_index'] == t)
                ]['scaled_demand'].values[0]
                total_units += units

        print(f'Total units sold: {total_units}')
        print(f'Revenue: €{total_units * price:.2f}')
        print(f'Ingredient costs: €{total_units * cost:.2f}')
        print(f'Truck costs: €{len(active_trucks) * truck_cost:.2f}')

        return profit
    else:
        print(f'\nDay {day}')
        print('No optimal solution found.')
        return 0  

if __name__ == '__main__':
    total_profit = 0
    for day in range(1, 6): 
        daily_profit = cpsat_solver(day, total_profit)
        total_profit += daily_profit

    print("\n==============================")
    print(f"Total Score (5 days): €{total_profit:.2f}")