from ortools.sat.python import cp_model
import pandas as pd
import time

class BurritoOptimizer:
    def __init__(self):
        self.demand_nodes = None
        self.truck_assignments = None
        self.problem_data = None
        
    def load_data(self):
        """Load and validate input data"""
        self.demand_nodes = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_demand_node_data.csv')
        self.truck_assignments = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_demand_truck_data.csv')
        self.problem_data = pd.read_csv('burrito_dataset/cscenDE4BD4_round1_day1_problem_data.csv')
        
        # Data validation
        assert not self.truck_assignments['scaled_demand'].isnull().any()
        assert (self.truck_assignments['scaled_demand'] >= 0).all()
        assert len(self.demand_nodes) > 0
        
    def solve(self):
        """Build and solve the optimization model"""
        model = cp_model.CpModel()
        
        # Parameters
        params = {
            'burrito_price': self.problem_data['burrito_price'][0],
            'ingredient_cost': self.problem_data['ingredient_cost'][0],
            'truck_cost': self.problem_data['truck_cost'][0],
            'max_truck_capacity': 200,
            'min_demand_coverage': 0.8  # Cover at least 80% of each demand
        }
        
        # Pre-process data
        feasible_assignments = self.truck_assignments[
            self.truck_assignments['scaled_demand'] > 0
        ].copy()
        
        all_demands = self.demand_nodes['index'].unique()
        all_trucks = feasible_assignments['truck_node_index'].unique()
        
        # Create variables
        truck_active = {t: model.NewBoolVar(f'truck_{t}') for t in all_trucks}
        assignments = {
            (row['demand_node_index'], row['truck_node_index']): model.NewBoolVar(f'assign_{row["demand_node_index"]}_{row["truck_node_index"]}')
            for _, row in feasible_assignments.iterrows()
        }
        
        # Add constraints
        self._add_constraints(model, feasible_assignments, all_demands, all_trucks, 
                            truck_active, assignments, params)
        
        # Set objective
        self._set_objective(model, feasible_assignments, truck_active, 
                          assignments, params)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300.0
        start_time = time.time()
        status = solver.Solve(model)
        solve_time = time.time() - start_time
        
        # Process and return results
        return self._process_results(solver, status, feasible_assignments, 
                                  all_trucks, truck_active, assignments, 
                                  params, solve_time)
    
    def _add_constraints(self, model, assignments_df, demands, trucks, 
                        truck_active, assignments, params):
        """Add all constraints to the model"""
        # 1. Demand coverage constraints
        for demand in demands:
            relevant = [a for (d,t), a in assignments.items() if d == demand]
            if relevant:
                # Each demand served by at most one truck
                model.Add(sum(relevant) <= 1)
                
                # Optional: Minimum coverage constraint
                original_demand = self.demand_nodes.loc[
                    self.demand_nodes['index'] == demand, 'demand'].values[0]
                min_coverage = int(params['min_demand_coverage'] * original_demand)
                
                covered_demand = sum(
                    assignments_df.loc[
                        (assignments_df['demand_node_index'] == demand) & 
                        (assignments_df['truck_node_index'] == truck),
                        'scaled_demand'].values[0] * assignments.get((demand, truck), 0)
                    for truck in trucks
                )
                model.Add(covered_demand >= min_coverage)
        
        # 2. Truck activation constraints
        for (d, t), var in assignments.items():
            model.Add(var <= truck_active[t])
        
        # 3. Truck capacity constraints
        for truck in trucks:
            truck_demand = sum(
                row['scaled_demand'] * assignments[(row['demand_node_index'], truck)]
                for _, row in assignments_df[assignments_df['truck_node_index'] == truck].iterrows()
            )
            model.Add(truck_demand <= params['max_truck_capacity'])
        
        # 4. Symmetry breaking (improves performance)
        for i, truck in enumerate(trucks[:-1]):
            model.Add(truck_active[truck] >= truck_active[trucks[i+1]])
    
    def _set_objective(self, model, assignments_df, truck_active, 
                      assignments, params):
        """Define the optimization objective"""
        total_units = sum(
            row['scaled_demand'] * assignments[(row['demand_node_index'], row['truck_node_index'])]
            for _, row in assignments_df.iterrows()
        )
        
        total_truck_costs = sum(
            truck_active[t] * params['truck_cost'] 
            for t in truck_active
        )
        
        # Create profit variable for better reporting
        max_possible = sum(self.demand_nodes['demand']) * params['burrito_price']
        profit = model.NewIntVar(-max_possible, max_possible, 'profit')
        model.Add(
            profit == (params['burrito_price'] - params['ingredient_cost']) * total_units - total_truck_costs
        )
        model.Maximize(profit)
    
    def _process_results(self, solver, status, assignments_df, trucks, 
                       truck_active, assignments, params, solve_time):
        """Extract and format results"""
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return {'status': 'No solution found'}
        
        results = {
            'status': 'Optimal' if status == cp_model.OPTIMAL else 'Feasible',
            'solve_time': solve_time,
            'active_trucks': [],
            'assignments': {},
            'metrics': {}
        }
        
        # Get active trucks
        results['active_trucks'] = [
            t for t in trucks if solver.Value(truck_active[t])
        ]
        
        # Process assignments
        total_units = 0
        for truck in results['active_trucks']:
            truck_assigns = {}
            truck_units = 0
            
            for (d, t), var in assignments.items():
                if t == truck and solver.Value(var):
                    units = assignments_df[
                        (assignments_df['demand_node_index'] == d) & 
                        (assignments_df['truck_node_index'] == t)
                    ]['scaled_demand'].values[0]
                    truck_assigns[d] = units
                    truck_units += units
            
            results['assignments'][truck] = {
                'demands_served': truck_assigns,
                'total_units': truck_units,
                'utilization': truck_units / params['max_truck_capacity']
            }
            total_units += truck_units
        
        # Calculate metrics
        results['metrics'] = {
            'total_units': total_units,
            'revenue': total_units * params['burrito_price'],
            'ingredient_costs': total_units * params['ingredient_cost'],
            'truck_costs': len(results['active_trucks']) * params['truck_cost'],
            'profit': solver.ObjectiveValue(),
            'solve_time': solve_time
        }
        
        return results

if __name__ == '__main__':
    optimizer = BurritoOptimizer()
    optimizer.load_data()
    results = optimizer.solve()
    
    print("\n=== Optimization Results ===")
    print(f"Status: {results['status']}")
    print(f"Solve Time: {results['solve_time']:.2f} seconds")
    print(f"\nProfit: €{results['metrics']['profit']}")
    print(f"Revenue: €{results['metrics']['revenue']}")
    print(f"Ingredient Costs: €{results['metrics']['ingredient_costs']}")
    print(f"Truck Costs: €{results['metrics']['truck_costs']}")
    
    print("\nActive Trucks:")
    for truck in results['active_trucks']:
        print(f"\n{truck}:")
        print(f"Total Units: {results['assignments'][truck]['total_units']}")
        print(f"Utilization: {results['assignments'][truck]['utilization']:.1%}")
        print("Demands Served:")
        for demand, units in results['assignments'][truck]['demands_served'].items():
            print(f"  {demand}: {units} units")