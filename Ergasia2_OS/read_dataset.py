import pandas as pd

def load_data(day):
    demand_nodes = pd.read_csv(f'burrito_dataset/scenDE4BD4_round1_day{day}_demand_node_data.csv')
    truck_assignments = pd.read_csv(f'burrito_dataset/scenDE4BD4_round1_day{day}_demand_truck_data.csv')
    problem_data = pd.read_csv(f'burrito_dataset/scenDE4BD4_round1_day{day}_problem_data.csv')
    return demand_nodes, truck_assignments, problem_data
