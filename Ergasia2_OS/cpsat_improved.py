from ortools.sat.python import cp_model
import csv

def load_data(demand_node_file, demand_truck_file, problem_data_file):
    """Load input data from three CSV files (excluding truck node data)."""
    # Load demand nodes
    demand_nodes = {}
    with open(demand_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            demand_nodes[row['index']] = {
                'name': row['name'],
                'demand': int(row['demand'])
            }

    # Load demand-truck assignments
    demand_truck_data = []
    truck_nodes = set()
    with open(demand_truck_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            demand_truck_data.append({
                'demand_node': row['demand_node_index'],
                'truck_node': row['truck_node_index'],
                'distance': float(row['distance']),
                'scaled_demand': int(row['scaled_demand'])
            })
            truck_nodes.add(row['truck_node_index'])

    # Load problem parameters
    with open(problem_data_file, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        burrito_price = int(row['burrito_price'])
        ingredient_cost = int(row['ingredient_cost'])
        truck_cost = int(row['truck_cost'])

    return {
        'demand_nodes': demand_nodes,
        'truck_nodes': list(truck_nodes),
        'demand_truck_data': demand_truck_data,
        'burrito_price': burrito_price,
        'ingredient_cost': ingredient_cost,
        'truck_cost': truck_cost
    }

def build_and_solve_model(data):
    """Build and solve the optimization model."""
    model = cp_model.CpModel()

    # Variables
    truck_vars = {truck: model.NewBoolVar(f'truck_{truck}') for truck in data['truck_nodes']}

    assignment_vars = {}
    for dt in data['demand_truck_data']:
        key = (dt['demand_node'], dt['truck_node'])
        assignment_vars[key] = model.NewBoolVar(f'assignment_{key[0]}_{key[1]}')

    # Constraint: Demand can only be assigned to deployed trucks
    for (demand_node, truck_node), var in assignment_vars.items():
        model.Add(var <= truck_vars[truck_node])

    # Constraint: Satisfy each customer's demand through multiple trucks (split delivery allowed)
    for demand_node in data['demand_nodes']:
        demand = data['demand_nodes'][demand_node]['demand']
        terms = [
            (assignment_vars[(demand_node, dt['truck_node'])], dt['scaled_demand'])
            for dt in data['demand_truck_data']
            if dt['demand_node'] == demand_node
        ]
        model.Add(sum(var * sd for var, sd in terms) >= demand)

    # Objective: Maximize profit
    revenue = sum(
        assignment_vars[(dt['demand_node'], dt['truck_node'])] *
        dt['scaled_demand'] * data['burrito_price']
        for dt in data['demand_truck_data']
    )
    truck_costs = sum(truck_vars[t] * data['truck_cost'] for t in data['truck_nodes'])
    ingredient_costs = sum(
        assignment_vars[(dt['demand_node'], dt['truck_node'])] *
        dt['scaled_demand'] * data['ingredient_cost']
        for dt in data['demand_truck_data']
    )
    model.Maximize(revenue - truck_costs - ingredient_costs)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        deployed_trucks = [t for t in data['truck_nodes'] if solver.Value(truck_vars[t])]
        assignments = [
            {
                'demand_node': d,
                'truck_node': t,
                'scaled_demand': next(dt['scaled_demand'] for dt in data['demand_truck_data']
                                      if dt['demand_node'] == d and dt['truck_node'] == t)
            }
            for (d, t), var in assignment_vars.items() if solver.Value(var)
        ]
        return {
            'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
            'profit': solver.ObjectiveValue(),
            'deployed_trucks': deployed_trucks,
            'assignments': assignments
        }
    else:
        return {
            'status': 'INFEASIBLE',
            'profit': None,
            'deployed_trucks': [],
            'assignments': []
        }


def print_solution(solution, data):
    if solution['status'] in ['INFEASIBLE', 'UNKNOWN']:
        print("No solution found.")
        return

    print("\n=== Solution ===")
    print(f"Total Profit: {solution['profit']}")
    print("\nDeployed Trucks:")
    for truck in solution['deployed_trucks']:
        print(f"- {truck}")

    print("\nAssignments:")
    for a in solution['assignments']:
        demand = data['demand_nodes'][a['demand_node']]
        print(f"{demand['name']} -> {a['truck_node']} "
              f"(serves {a['scaled_demand']} of {demand['demand']})")

def solve_burrito_problem(demand_node_file, demand_truck_file, problem_data_file):
    try:
        data = load_data(demand_node_file, demand_truck_file, problem_data_file)

        # === Check feasibility before modeling ===
        print("\n[Feasibility Check]")
        for dn in data['demand_nodes']:
            demand = data['demand_nodes'][dn]['demand']
            available = sum(
                dt['scaled_demand'] 
                for dt in data['demand_truck_data'] 
                if dt['demand_node'] == dn
            )
            if available < demand:
                print(f"[!] Demand node {dn} ({data['demand_nodes'][dn]['name']}) "
                      f"has demand {demand} but only {available} available → INFEASIBLE")
            else:
                print(f"[✓] Demand node {dn} is OK (demand: {demand}, available: {available})")

        # Build and solve model
        solution = build_and_solve_model(data)

        # Print results
        print_solution(solution, data)

        return solution
    except Exception as e:
        print(f"Error: {e}")
        return None


# Example usage
if __name__ == "__main__":
    solve_burrito_problem(
        demand_node_file='burrito_dataset/scenDE4BD4_round1_day1_demand_node_data.csv',
        demand_truck_file='burrito_dataset/scenDE4BD4_round1_day1_demand_truck_data.csv',
        problem_data_file='burrito_dataset/scenDE4BD4_round1_day1_problem_data.csv'
    )
