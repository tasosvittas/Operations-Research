from gurobipy import Model, GRB, quicksum
from read_dataset import load_data
import pandas as pd
import time

def solve_with_gurobi(day):
    demand_nodes, truck_assignments, problem_data = load_data(day)

    #parametroi
    burrito_price = problem_data['burrito_price'][0]
    ingredient_cost = problem_data['ingredient_cost'][0]
    truck_cost = problem_data['truck_cost'][0]

    feasible_assignments = truck_assignments[truck_assignments['scaled_demand'] > 0]
    all_demands = demand_nodes['index'].unique()
    all_trucks = truck_assignments['truck_node_index'].unique()

    model = Model("burrito_gurobi")
    model.setParam("OutputFlag", 0) 

    assign = {}
    for _, row in feasible_assignments.iterrows():
        d = row['demand_node_index']
        t = row['truck_node_index']
        assign[(d, t)] = model.addVar(vtype=GRB.BINARY, name=f"assign_{d}_{t}")

    truck_active = {
        t: model.addVar(vtype=GRB.BINARY, name=f"truck_{t}")
        for t in all_trucks
    }

    model.update()

    # kathe demand = me mia kantina mono
    for d in all_demands:
        relevant = [assign[(d, t)] for (dd, t) in assign if dd == d]
        if relevant:
            model.addConstr(quicksum(relevant) <= 1)

    #elegxos gia kantina active or not
    for (d, t), var in assign.items():
        model.addConstr(var <= truck_active[t])

    #synoliko plithos burritos pou tha diathethoun
    total_units = quicksum(assign[(d, t)] * feasible_assignments[
        (feasible_assignments['demand_node_index'] == d) &
        (feasible_assignments['truck_node_index'] == t)
    ]['scaled_demand'].values[0] for (d, t) in assign)

    total_truck_costs = quicksum(truck_active[t] * truck_cost for t in all_trucks)
    
    # profit = (price - cost)*units - truck_costs
    profit = (burrito_price - ingredient_cost) * total_units - total_truck_costs
    model.setObjective(profit, GRB.MAXIMIZE)

    start_time = time.time()
    model.optimize()
    end_time = time.time()
    elapsed = end_time - start_time

    if model.status == GRB.OPTIMAL:
        print(f"\nDay {day}")
        print(f"Profit: €{model.objVal:.2f}")
        print(f"Total Time (Gurobi): {elapsed:.4f} seconds")

        active_trucks = [t for t in all_trucks if truck_active[t].x > 0.5]
        print("Active trucks:", active_trucks)

        total_units = 0
        for (d, t), var in assign.items():
            if var.x > 0.5:
                units = feasible_assignments[
                    (feasible_assignments['demand_node_index'] == d) &
                    (feasible_assignments['truck_node_index'] == t)
                ]['scaled_demand'].values[0]
                total_units += units

        revenue = total_units * burrito_price
        ingredient_total = total_units * ingredient_cost
        truck_total = len(active_trucks) * truck_cost

        print(f"Total units sold: {total_units}")
        print(f"Revenue: €{revenue:.2f}")
        print(f"Ingredient costs: €{ingredient_total:.2f}")
        print(f"Truck costs: €{truck_total:.2f}")
        return model.objVal
    else:
        print(f"\nDay {day}")
        print("No optimal solution found.")
        return 0


if __name__ == "__main__":
    total_profit = 0
    total_start = time.time()  

    for day in range(1, 6):
        total_profit += solve_with_gurobi(day)

    total_end = time.time()
    total_elapsed = total_end - total_start

    print("\n==============================")
    print(f"Total Score (5 days): €{total_profit:.2f}")
    print(f"Total Time (Gurobi): {total_elapsed:.4f} seconds")
