import os

import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np

import src.envs.core as ising_env
from vns import vns
from experiments.utils import test_network, load_graph_set, mk_dir
from src.envs.utils import (SingleGraphGenerator, RandomBarabasiAlbertGraphGenerator,
                            RewardSignal, ExtraAction, EdgeType,
                            OptimisationTarget, SpinBasis,
                            DEFAULT_OBSERVABLES)
from src.networks.mpnn import MPNN

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass


def run(save_loc=f"vns/results",
        # graph_save_loc=f"_graphs/validation/BA_{n_spins}spin_m4_100graphs.pkl",
        step_factor=None,
        n_attemps=50):
    """
    Runs vns on the selected graphs 

    """

    # n_spins_train = 2000
    n_sets = 5
    graph_spins = 200

    graph_save_loc = f"_graphs/validation/BA_{graph_spins}spin_m4_100graphs.pkl"

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    if step_factor is None:
        step_factor = 2

    env_args = {'n_sets': n_sets,
                'observables': DEFAULT_OBSERVABLES,
                'reward_signal': RewardSignal.BLS,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': OptimisationTarget.CUT,
                'spin_basis': SpinBasis.BINARY,
                'norm_rewards': True,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': None,
                'basin_reward': None,
                'reversible_spins': True}

    ####################################################
    # LOAD VALIDATION GRAPHS
    ####################################################

    graphs_test = load_graph_set(graph_save_loc)  # 100 graphs
    first_graph = graphs_test[0]

    # ####################################################
    # # TEST VNS ON VALIDATION GRAPHS
    # ####################################################

    n_spins = first_graph.shape[0]
    state = np.zeros(n_spins)
    state[:] = np.array(np.random.randint(n_sets, size=n_spins,))
    adjacency_matrix = first_graph
    print("n_spins", n_spins)
    print("matrix", adjacency_matrix)

    results = vns.vns(state, adjacency_matrix, n_sets=5, k_max=5)
    print(results)
    print(vns.compute_cut(results, adjacency_matrix))


if __name__ == "__main__":
    run()
