import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


opti_path = os.path.join(
    os.getcwd(), "_graphs/validation/opts/cuts_ER_20spin_p15_100graphs.pkl")
trained_path = os.path.join(
    os.getcwd(), "ER_20spin/eco/data/results_ER_20spin_p15_100graphs.pkl")

benchmark_path = os.path.join(
    os.getcwd(), "_graphs/benchmarks/gset_800spin_graphs.pkl")
benchmark_ising_path = os.path.join(
    os.getcwd(), "_graphs/benchmarks/opts/cuts_ising_125spin.pkl")

results_path = os.path.join(
    os.getcwd(), "_graphs/benchmarks/gset_800spin_graphs.pkl")


def show_graphs_info(path):
    # Open the file for reading in binary mode
    with open(path, 'rb') as file:
        # Call the load method to deserialize the object from the file
        graphs = pickle.load(file)
        print(f"The path contains {len(graphs)} graphs.")
        first_graph = graphs[0]
        print(f"The first graph is composed of {len(first_graph)} vertices.")
        print(first_graph)


def show_graph_results(path):
    print(f"The current path is {path}")

    # Open the file for reading in binary mode
    with open(path, 'rb') as file:
        # Call the load method to deserialize the object from the file
        solutions = pickle.load(file)
        print(f"The path contains {len(solutions)} graph solutions")
        print(f"The graph solutions are the following: {solutions}")
        return solutions
        # first_graph = graphs[0]
        # print(f"The first graph is composed of {len(first_graph)} vertices.")


def results_to_array(path_list):
    if path_list is not list:
        path_list = [path_list]
    y = []
    for idx, path in enumerate(path_list):
        with open(path, 'rb') as file:
            # Call the load method to deserialize the object from the file
            solutions = pickle.load(file)
            solutions = np.array(solutions)
            if idx == 0:
                df = pd.DataFrame(solutions, columns=['x', f'y0'])
                y.append('y0')
            else:
                new_df = pd.DataFrame(solutions, columns=['x', f'y{idx}'])
                df = pd.merge(df, new_df, on='x')
                y.append(f'y{idx}')
    return df, y


print("================= Plot results in a single graph")
# df, y = results_to_array(results_path)
# df.plot(x='x', y=y)
# plt.xlabel('Timestep')
# plt.ylabel('Maxcut')
# plt.title('Comparison of the best cuts')
# plt.show()

print("================= Graphs info")
# show_graphs_info(results_test_path)
# show_graphs_info(trained_path)
# show_graphs_info(benchmark_ising_path)
# show_graphs_info(results_path)
show_graphs_info(results_path)
print("================= Optimal result")
# results_test = show_graph_results(results_test_path)
# results_opti = show_graph_results(results_opti_path)
# one = results_test["cut"]
# two = results_opti["cut"]
# one.plot(color='blue', label='Self trained')
# # two.plot(color='red', label='Optimized')
# plt.title('Max-3-cut')
# plt.legend()

# # Show the plot
# plt.show()
