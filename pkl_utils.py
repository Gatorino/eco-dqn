import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Create an ArgumentParser object
# parser = argparse.ArgumentParser(description='Parser')
# parser.add_argument('graph_vertices', type=int,
#                     help='Number of vertices of the graph', default=800)

# # Parse the arguments
# args = parser.parse_args()
# print(args.graph_vertices)
# graph_vertices = args.graph_vertices

# dir = os.path.join(
#     os.getcwd(), "_graphs/validation/opts")

# if graph_vertices == 800:
#     path1 = os.path.join(dir, "cuts_gset_800spin.pkl")
#     path2 = os.path.join(dir, "sols_gset_800spin.pkl")


# path = os.path.join(
#     os.getcwd(), "_graphs/benchmarks/gset_800spin_graphs.pkl")

opti_path = os.path.join(
    os.getcwd(), "_graphs/validation/opts/cuts_ER_20spin_p15_100graphs.pkl")
trained_path = os.path.join(
    os.getcwd(), "ER_20spin/eco/data/results_ER_20spin_p15_100graphs.pkl")

benchmark_path = os.path.join(
    os.getcwd(), "_graphs/benchmarks/gset_800spin_graphs.pkl")
benchmark_ising_path = os.path.join(
    os.getcwd(), "_graphs/benchmarks/opts/cuts_ising_125spin.pkl")

results_path = os.path.join(
    os.getcwd(), "kcut/eco/2sets/network/test_scores.pkl")
results_test_path = os.path.join(
    os.getcwd(), "kcut/eco/2sets/test/results_BA_20spin_m4_100graphs.pkl")
results_opti_path = os.path.join(
    os.getcwd(), "temp/kcut/eco/2sets/test/results_BA_20spin_m4_100graphs.pkl")


def show_graphs_info(path):
    # if "opts" in path:
    #     print("The path contains 'opts'. This folder contains solutions and not graphs.")
    #     return

    # Open the file for reading in binary mode
    with open(path, 'rb') as file:
        # Call the load method to deserialize the object from the file
        graphs = pickle.load(file)
        print(f"The path contains {len(graphs)} graphs.")
        print(type(graphs))
        # print(graphs.head())
        # print(graphs.columns)
        first_graph = graphs[0]
        print(f"The first graph is composed of {len(first_graph)} vertices.")
        print(first_graph)


def show_graph_results(path):
    print(f"The current path is {path}")
    # if "opts" not in path:
    #     print("The path contains does not contain 'opts'. This folder should not contain solutions.")
    #     return

    # Open the file for reading in binary mode
    with open(path, 'rb') as file:
        # Call the load method to deserialize the object from the file
        solutions = pickle.load(file)
        print(f"The path contains {len(solutions)} graph solutions")
        print(f"The graph solutions are the following: {solutions}")
        return solutions
        # first_graph = graphs[0]
        # print(f"The first graph is composed of {len(first_graph)} vertices.")

    # with open(path, 'rb') as file:
    #     # Call the load method to deserialize the object from the file
    #     solutions = pickle.load(file)
    #     print(f"The path contains {len(solutions)} graph solutions")
    #     # print(f"The graph solutions are the following: {solutions}")
    #     first_graph = solutions[0]
    #     print(f"The first graph is composed of {len(first_graph)} vertices.")
    #     print(first_graph)


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
print("================= Optimal result")
results_test = show_graph_results(results_test_path)
results_opti = show_graph_results(results_opti_path)
one = results_test["cut"]
two = results_opti["cut"]
one.plot(color='blue', label='Self trained')
two.plot(color='red', label='Optimized')
plt.title('Max-2-cut')
plt.legend()

# Show the plot
plt.show()
# show_graph_results(results_path)
# show_graph_results(benchmark_ising_path)
# show_graph_results(opti_path)
# show_graph_results(path)
