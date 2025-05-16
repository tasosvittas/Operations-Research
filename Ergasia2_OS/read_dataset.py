import pandas as pd

def load_data():
    # start_date = 1
    # end_date = 5
    # for i in range(start_date, end_date):
    #     demand_nodes = pd.read_csv(f'burrito_dataset/scenDE4BD4_round1_day{i}_demand_node_data.csv')
    #     truck_assignments = pd.read_csv(f'burrito_dataset/scenDE4BD4_round1_day{i}_demand_truck_data.csv')
    #     problem_data = pd.read_csv(f'burrito_dataset/scenDE4BD4_round1_day{i}_problem_data.csv')

    demand_nodes = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_demand_node_data.csv')
    truck_assignments = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_demand_truck_data.csv')
    problem_data = pd.read_csv('burrito_dataset/scenDE4BD4_round1_day1_problem_data.csv')

        # print(demand_nodes,truck_assignments,problem_data)
    return demand_nodes, truck_assignments, problem_data


# if __name__ == '__main__':
#     load_data()