import os

import matplotlib.pyplot as plt
import torch
import argparse

import src.envs.core as ising_env
from experiments.utils import test_network, load_graph_set, mk_dir
from src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            DEFAULT_OBSERVABLES)
from src.networks.mpnn import MPNN

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

# Change number of sets and vertices of the graph.
parser = argparse.ArgumentParser(description='Arguments of the environment.')
parser.add_argument('-k', '--n_sets', type=int,
                    required=True, help='Number of sets.')
parser.add_argument('-s', '--n_spins', type=int,
                    required=True, help='Number of vertices of the training graphs.')
parser.add_argument('-l', '--save_loc', type=str,
                    required=True, help='Directory for the results.')
parser.add_argument('-g', '--graph_spins', type=int,
                    required=True, help='Number of spins of the validation graphs.')
args = parser.parse_args()
n_sets = args.n_sets
n_spins = args.n_spins
save_loc = args.save_loc
graph_spins = args.graph_spins

# kcut/eco/{n_sets}sets/test/


def run(save_loc=f"kcut/eco/{n_sets}sets/test/",
        # network_save_loc="experiments/pretrained_agent/networks/eco/network_best_BA_20spin.pth",
        # network_save_loc=f"kcut/eco/{n_sets}sets/network/network_best.pth",
        # graph_save_loc=f"_graphs/validation/BA_{n_spins}spin_m4_100graphs.pkl",
        batched=True,
        max_batch_size=None,
        step_factor=None,
        n_attemps=50):

    network_save_loc = f"kcut/eco/{n_sets}sets/network/network_best.pth"
    graph_save_loc = f"_graphs/validation/BA_{graph_spins}spin_m4_100graphs.pkl"

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # FOLDER LOCATIONS
    ####################################################

    print("save location :", save_loc)
    print("network params :", network_save_loc)
    mk_dir(save_loc)

    ####################################################
    # NETWORK SETUP
    ####################################################

    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }

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

    graphs_test = load_graph_set(graph_save_loc)

    ####################################################
    # SETUP NETWORK TO TEST
    ####################################################

    test_env = ising_env.make("SpinSystem",
                              SingleGraphGenerator(graphs_test[0]),
                              graphs_test[0].shape[0] * step_factor,
                              **env_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    print("Set torch default device to {}.".format(device))

    network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                         **network_args).to(device)

    network.load_state_dict(torch.load(network_save_loc, map_location=device))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    print("Sucessfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))

    ####################################################
    # TEST NETWORK ON VALIDATION GRAPHS
    ####################################################

    results, results_raw, history = test_network(network, env_args, graphs_test, device, step_factor,
                                                 return_raw=True, return_history=True, n_attempts=n_attemps,
                                                 batched=batched, max_batch_size=max_batch_size)

    results_fname = "results_" + \
        os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + ".pkl"
    results_raw_fname = "results_" + \
        os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_raw.pkl"
    history_fname = "results_" + \
        os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_history.pkl"

    for res, fname, label in zip([results, results_raw, history],
                                 [results_fname, results_raw_fname, history_fname],
                                 ["results", "results_raw", "history"]):
        save_path = os.path.join(save_loc, fname)
        res.to_pickle(save_path)
        print("{} saved to {}".format(label, save_path))


if __name__ == "__main__":
    run()
