import os
import pickle
import networkx as nx
import time
import numpy as np
import scipy as sp
import pandas as pd
import torch

from collections import namedtuple
from copy import deepcopy

#from vns import vns
import src.envs.core as ising_env
from src.envs.utils import (SingleGraphGenerator, SpinBasis)
from src.agents.solver import Network, Greedy

####################################################
# TESTING ON GRAPHS
####################################################


def test_network(network, env_args, graphs_test, device=None, step_factor=1, batched=True,
                 n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if batched:
        return __test_network_batched(network, env_args, graphs_test, device, step_factor,
                                      n_attempts, return_raw, return_history, max_batch_size)
    else:
        if max_batch_size is not None:
            print(
                "Warning: max_batch_size argument will be ignored for when batched=False.")
        return __test_network_sequential(network, env_args, graphs_test, step_factor,
                                         n_attempts, return_raw, return_history)


def __test_network_batched(network, env_args, graphs_test, device=None, step_factor=1,
                           n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)

    # HELPER FUNCTION FOR NETWORK TESTING

    acting_in_reversible_spin_env = env_args['reversible_spins']

    if env_args['reversible_spins']:
        # If MDP is reversible, both actions are allowed.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = (0, 1)
        elif env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = (1, -1)
    else:
        # If MDP is irreversible, only return the state of spins that haven't been flipped.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = 0
        if env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = 1

    def state_to_one_hot(state, n_spins):
        new_values = np.zeros((env_args['n_sets'], n_spins))
        former_values = state[0, :].astype(int)
        for i in range(n_spins):
            new_values[former_values[i], i] = 1
        state = np.vstack((new_values, state[1:, :]))

        return state

    def predict(states):

        qs = network(states)

        if acting_in_reversible_spin_env:
            if qs.dim() == 1:
                actions = [qs.argmax().item()]
            else:
                actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
            return actions
        else:
            print("Triggering allowed_action_state")
            if qs.dim() == 1:
                x = (states.squeeze()[:, 0] == allowed_action_state).nonzero()
                actions = [x[qs[x].argmax().item()].item()]
            else:
                disallowed_actions_mask = (
                    states[:, :, 0] != allowed_action_state)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -1000)
                actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()
            return actions

    # NETWORK TESTING

    results = []
    results_raw = []
    if return_history:
        history = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1

    for j, test_graph in enumerate(graphs_test):
        print(f"Starting graph {j}")
        i_comp = 0
        i_batch = 0
        t_total = 0

        n_spins = test_graph.shape[0]
        n_steps = int(n_spins * step_factor)

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)

        # print("Running greedy solver with +1 initialisation of spins...", end="...")
        # # Calculate the greedy cut with all spins initialised to +1
        # greedy_env = deepcopy(test_env)
        # greedy_env.reset(spins=np.array([1] * test_graph.shape[0]))

        # greedy_agent = Greedy(greedy_env)
        # greedy_agent.solve()

        # greedy_single_cut = greedy_env.get_best_cut()
        # greedy_single_spins = greedy_env.best_spins

        # print("done.")

        if return_history:
            actions_history = []
            rewards_history = []
            scores_history = []
            immanencys_history = []
            greedy_availables_history = []
            distance_best_scores_history = []
            distance_best_states_history = []

        best_cuts = []
        init_spins = []
        best_spins = []

        #vns_cuts = []
        #vns_spins = []
        #greedy_cuts = []
        #greedy_spins = []

        while i_comp < n_attempts:
            # print("New attempt")

            if max_batch_size is None:
                batch_size = n_attempts
            else:
                batch_size = min(n_attempts - i_comp, max_batch_size)

            i_comp_batch = 0

            if return_history:
                actions_history_batch = [[None]*batch_size]
                rewards_history_batch = [[None] * batch_size]
                scores_history_batch = []
                immanencys_history_batch = [[None]*batch_size]
                greedy_availables_history_batch = [[None]*batch_size]
                distance_best_scores_history_batch = [[None]*batch_size]
                distance_best_states_history_batch = [[None]*batch_size]

            test_envs = [None] * batch_size
            best_cuts_batch = [-1e3] * batch_size
            init_spins_batch = [[] for _ in range(batch_size)]
            best_spins_batch = [[] for _ in range(batch_size)]

            #greedy_envs = [None] * batch_size
            #greedy_cuts_batch = []
            #greedy_spins_batch = []

            #vns_envs = [None] * batch_size
            #vns_cuts_batch = []
            #vns_spins_batch = []

            obs_batch = [None] * batch_size
            
            #immanencys = [[] for _ in range(batch_size)]
            #greedy_availables = [[] for _ in range(batch_size)]
            #distance_from_best_scores = [[] for _ in range(batch_size)]
            #distance_from_best_states = [[] for _ in range(batch_size)]
            
            #print("Preparing batch of {} environments for graph {}.".format(
             #   batch_size, j), end="...")
            np.random.seed(1)
            init_cuts = [None] * batch_size
            for i in range(batch_size):
                env = deepcopy(test_env)
                # obs_batch[i] = state_to_one_hot(env.reset(), n_spins)
                print("Test seed batch size init", np.random.randint(100))
                spins = np.array(np.random.randint(env.n_sets, size=n_spins)) 
                #print(f"Testing seed {i}", np.random.randint(100))
                obs_batch[i] = env.reset(spins)
                test_envs[i] = env
                #greedy_envs[i] = deepcopy(env)
                #vns_envs[i] = deepcopy(env)
                init_spins_batch[i] = env.best_spins
                #init_cuts[i] = env.get_best_cut()
            if return_history:
                scores_history_batch.append(
                    [env.calculate_score() for env in test_envs])
                
            #for i in range(batch_size):
            #    print(f"Initial cut n°{i}: {init_cuts[i]}")
            #print("done.")

            # Calculate the max cut acting w.r.t. the network
            t_start = time.time()

            # pool = mp.Pool(processes=16)

            k = 0
            total_predict_time = 0
            total_env_step_time = 0
            total_history_time = 0
            total_obs_conversion_time = 0
            #print("Starting to run envs") 
            while i_comp_batch < batch_size:
                t1 = time.time()
                # Note: Do not convert list of np.arrays to FloatTensor, it is very slow!
                # see: https://github.com/pytorch/pytorch/issues/13918
                # Hence, here we convert a list of np arrays to a np array.
                obs_conversion_time = time.time()
                obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
                total_obs_conversion_time += time.time()-obs_conversion_time
                predict_time = time.time()
                actions = predict(obs_batch)
                total_predict_time += time.time()-predict_time
                # print("Predict time", time.time()-predict_time)
                obs_batch = []

                if return_history:
                    scores = []
                    rewards = []
                    immanencys = []
                    greedy_availables = []
                    distance_best_scores = []
                    distance_best_states = []


                i = 0
                if isinstance(actions, int):
                    actions = np.array([actions])
                for env, action in zip(test_envs, actions):

                    if env is not None:
                        env_step_time = time.time()
                        #obs, rew, done, info = env.step(action)
                        obs, rew, done, info, immanency, greedy_available, distance_best_score, distance_best_state = env.step(action)
                        #print("greedy", greedy_available)
                        #if done:
                        #    print("done one env")
                        #immanencys[i].append(immanency)
                        #greedy_availables[i].append(greedy_available)
                        #distance_from_best_scores[i].append(distance_from_best_score)
                        #distance_from_best_states[i].append(distance_from_best_state)
                        total_env_step_time += time.time()-env_step_time

                        if return_history:
                            history_time = time.time()
                            scores.append(env.calculate_score())
                            rewards.append(rew)
                            immanencys.append(immanency)
                            greedy_availables.append(greedy_available)
                            distance_best_scores.append(distance_best_score)
                            distance_best_states.append(distance_best_states)
                            total_history_time += time.time()-history_time

                        if not done:
                            obs_batch.append(obs)
                        else:
                            #print("One env done") 
                            best_cuts_batch[i] = env.get_best_cut()
                            best_spins_batch[i] = env.best_spins
                            i_comp_batch += 1
                            i_comp += 1
                            test_envs[i] = None
                    i += 1
                    k += 1

                if return_history:
                    actions_history_batch.append(actions)
                    scores_history_batch.append(scores)
                    rewards_history_batch.append(rewards)
                    immanencys_history_batch.append(immanencys)
                    greedy_availables_history_batch.append(greedy_availables)
                    distance_best_scores_history_batch.append(distance_best_scores)
                    distance_best_states_history_batch.append(distance_best_states)
            

            # print("\t",
            #       "Par. steps :", k,
            #       "Env steps : {}/{}".format(k/batch_size, n_steps),
            #       'Time: {0:.3g}s'.format(time.time()-t1))

            #dic = {'immanency': immanencys, 'greedy available': greedy_availables, 'distance_score': distance_from_best_scores, 'distance_state': distance_from_best_states}
            #df = pd.DataFrame(dic)
            #df.to_csv('global_features.csv')
            mean_env_step_time = total_env_step_time / (n_steps*n_attempts)
            mean_predict_time = total_predict_time / n_steps
            #print("Mean env step", mean_env_step_time)
            #print("Total env step", total_env_step_time)
            #print("Mean predict time", mean_predict_time)
            #print("Total predict time", total_predict_time)
            #print("Total history time", total_history_time)
            #print("Total conversion time", total_obs_conversion_time)
            t_total += (time.time() - t_start)
            i_batch += 1
            print("Finished agent testing batch {}.".format(i_batch))
            print("Network Time", t_total)

            #if env_args["reversible_spins"]:

            #    print("Running vns with {} random initialisations of spins for batch {}...".format(
            #        batch_size, i_batch), end="...")

            #    start_vns = time.time()

            #    adjacency_matrix = test_graph

            #    for env in vns_envs:
                    # print("VNS solving")
                    # current_time = time.time()-t_start
                    # print("Time", current_time)
            #        state = env.state[0, :env.n_spins]
            #        solution = vns.vns(
            #            state, adjacency_matrix, n_sets=env.n_sets, k_max=1)
            #        cut = vns.compute_cut(solution, adjacency_matrix)
            #        vns_cuts_batch.append(cut)
            #        vns_spins_batch.append(solution)

            #    print("done.")

                #print("Running greedy solver with {} random initialisations of spins for batch {}...".format(
                 #   batch_size, i_batch), end="...")

                # current_time = time.time()-t_start
                # print("Time", current_time)
                # start_greedy = time.time()

                #for env in greedy_envs:
                    # print("Greedy solving")
                    # current_time = time.time()-t_start
                    # print("Time", current_time)
                 #   Greedy(env).solve()
                  #  cut = env.get_best_cut()
                   # greedy_cuts_batch.append(cut)
                    #greedy_spins_batch.append(env.best_spins)

                #for i in range(batch_size):
                #    print(f"Greedy sol n°{i}: {greedy_cuts_batch[i]}")
                #print("done.")
            current_time = time.time()-t_start
            # print("Time", current_time)
            #print("VNS Solver time", time.time()-start_vns)

            if return_history:
                actions_history += actions_history_batch
                rewards_history += rewards_history_batch
                scores_history += scores_history_batch
                immanencys_history += immanencys_history_batch
                greedy_availables_history += greedy_availables_history_batch
                distance_best_scores_history += distance_best_scores_history_batch
                distance_best_states_history += distance_best_states_history_batch

            best_cuts += best_cuts_batch
            init_spins += init_spins_batch
            best_spins += best_spins_batch

        #if env_args["reversible_spins"]:
            #vns_cuts += vns_cuts_batch
            #vns_spins += vns_spins_batch
            #vns_mean_cut = np.mean(vns_cuts)
            #     greedy_cuts += greedy_cuts_batch
            #     greedy_spins += greedy_spins_batch

            # print("\tGraph {}, par. steps: {}, comp: {}/{}".format(j, k, i_comp, batch_size),
            #       end="\r" if n_spins<100 else "")

        #print("Finished big while")
        current_time = time.time()-t_start
        #print("Time", current_time)
        i_best = np.argmax(best_cuts)
        best_cut = best_cuts[i_best]
        sol = best_spins[i_best]
        std = np.std(best_cuts)
        mean_cut = np.mean(best_cuts)
        mean_distance = 0
        for solution in best_spins:
            mean_distance += np.count_nonzero(sol!=solution)
        mean_distance /= len(best_spins)-1
        #if env_args["reversible_spins"]:
        #    idx_best_vns = np.argmax(vns_cuts)
        #    vns_cut = vns_cuts[idx_best_vns]
        #    vns_spins = vns_spins[idx_best_vns]
        #     idx_best_greedy = np.argmax(greedy_cuts)
        #     greedy_random_cut = greedy_cuts[idx_best_greedy]
        #     greedy_random_spins = greedy_spins[idx_best_greedy]
        #     greedy_random_mean_cut = np.mean(greedy_cuts)
        # else:
        #     greedy_random_cut = greedy_single_cut
        #     greedy_random_spins = greedy_single_spins
        #     greedy_random_mean_cut = greedy_single_cut

        # print('Graph {}, best(mean) cut: {}({}), greedy cut (rand init / +1 init) : {} / {}.  ({} attempts in {}s)\t\t\t'.format(
        #     j, best_cut, mean_cut, greedy_random_cut, greedy_single_cut, n_attempts, np.round(t_total, 2)))

        # results.append([best_cut, sol,
        #                 mean_cut,
        #                 greedy_single_cut, greedy_single_spins,
        #                 greedy_random_cut, greedy_random_spins,
        #                 greedy_random_mean_cut,
        #                 t_total/(n_attempts)])
        print("Best cut", best_cut)
        print("mean cut", mean_cut)
        print("std", std)
        print("mean distance", mean_distance)
        # print("Greedy single cut", greedy_single_cut)
        #print("VNS best cut", vns_cut)
        #print("VNS mean cut", vns_mean_cut)

        results.append(
            [best_cut, mean_cut, sol, std, mean_distance])

        # print("Greedy Random cut", greedy_random_cut)
        # print("Greedy random mean cut", greedy_random_mean_cut)

        #results_raw.append([init_spins,
        #                    best_cuts, best_spins,
        #                    vns_cuts, vns_spins])

        if return_history:
            history.append([np.array(actions_history).T.tolist(),
                            np.array(scores_history).T.tolist(),
                            np.array(rewards_history).T.tolist(),
                            np.array(immanencys_history).T.tolist(),
                            np.array(greedy_availables_history).T.tolist(),
                            np.array(distance_best_states_history).T.tolist(),
                            np.array(distance_best_scores_history).T.tolist()])
    results = pd.DataFrame(data=results, columns=["best_cut", "mean_cut", "sol","std", "mean_distance"])

    # results = pd.DataFrame(data=results, columns=["cut", "sol",
    #                                               "mean cut",
    #                                               "greedy (+1 init) cut", "greedy (+1 init) sol",
    #                                               "greedy (rand init) cut", "greedy (rand init) sol",
    #                                               "greedy (rand init) mean cut",
    #                                               "time"])

    #results_raw = pd.DataFrame(data=results_raw, columns=["init spins",
    #                                                      "cuts", "sols",
    #                                                      "vns cuts", "vns sols"])

    if return_history:
        history = pd.DataFrame(data=history, columns=[
                               "actions", "scores", "rewards", "immanencys", "greedy_availables", "distance_best_scores", "distance_best_states"])

    if return_raw == False and return_history == False:
        return results
    else:
        ret = [results]
        if return_raw:
            ret.append(results_raw)
        if return_history:
            ret.append(history)
        return ret


def __test_network_sequential(network, env_args, graphs_test, step_factor=1,
                              n_attempts=50, return_raw=False, return_history=False):

    if return_raw or return_history:
        raise NotImplementedError(
            "I've not got to this yet!  Used the batched test script (it's faster anyway).")

    results = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1

    for i, test_graph in enumerate(graphs_test):

        n_steps = int(test_graph.shape[0] * step_factor)

        best_cut = -1e3
        best_spins = []

        greedy_random_cut = -1e3
        greedy_random_spins = []

        greedy_single_cut = -1e3
        greedy_single_spins = []

        times = []

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)
        net_agent = Network(network, test_env,
                            record_cut=False, record_rewards=False, record_qs=False)

        greedy_env = deepcopy(test_env)
        greedy_env.reset(spins=np.array([1] * test_graph.shape[0]))
        greedy_agent = Greedy(greedy_env)

        greedy_agent.solve()

        greedy_single_cut = greedy_env.get_best_cut()
        greedy_single_spins = greedy_env.best_spins

        for k in range(n_attempts):

            net_agent.reset(clear_history=True)
            greedy_env = deepcopy(test_env)
            greedy_agent = Greedy(greedy_env)

            tstart = time.time()
            net_agent.solve()
            times.append(time.time() - tstart)

            cut = test_env.get_best_cut()
            if cut > best_cut:
                best_cut = cut
                best_spins = test_env.best_spins

            greedy_agent.solve()

            greedy_cut = greedy_env.get_best_cut()
            if greedy_cut > greedy_random_cut:
                greedy_random_cut = greedy_cut
                greedy_random_spins = greedy_env.best_spins

            # print('\nGraph {}, attempt : {}/{}, best cut : {}, greedy cut (rand init / +1 init) : {} / {}\t\t\t'.format(
            #     i + 1, k, n_attemps, best_cut, greedy_random_cut, greedy_single_cut),
            #     end="\r")
            print('\nGraph {}, attempt : {}/{}, best cut : {}, greedy cut (rand init / +1 init) : {} / {}\t\t\t'.format(
                i + 1, k, n_attempts, best_cut, greedy_random_cut, greedy_single_cut),
                end=".")

        results.append([best_cut, best_spins,
                        greedy_single_cut, greedy_single_spins,
                        greedy_random_cut, greedy_random_spins,
                        np.mean(times)])

    return pd.DataFrame(data=results, columns=["cut", "sol",
                                               "greedy (+1 init) cut", "greedy (+1 init) sol",
                                               "greedy (rand init) cut", "greedy (rand init) sol",
                                               "time"])

####################################################
# LOADING GRAPHS
####################################################


Graph = namedtuple('Graph', 'name n_vertices n_edges matrix bk_val bk_sol')


def load_graph(graph_dir, graph_name):

    inst_loc = os.path.join(graph_dir, 'instances', graph_name+'.mc')
    val_loc = os.path.join(graph_dir, 'bkvl', graph_name+'.bkvl')
    sol_loc = os.path.join(graph_dir, 'bksol', graph_name+'.bksol')

    vertices, edges, matrix = 0, 0, None
    bk_val, bk_sol = None, None

    with open(inst_loc) as f:
        for line in f:
            arr = list(map(int, line.strip().split(' ')))
            if len(arr) == 2:  # contains the number of vertices and edges
                n_vertices, n_edges = arr
                matrix = np.zeros((n_vertices, n_vertices))
            else:
                assert type(
                    matrix) == np.ndarray, 'First line in file should define graph dimensions.'
                i, j, w = arr[0]-1, arr[1]-1, arr[2]
                matrix[[i, j], [j, i]] = w

    with open(val_loc) as f:
        bk_val = float(f.readline())

    with open(sol_loc) as f:
        bk_sol_str = f.readline().strip()
        bk_sol = np.array([int(x) for x in list(bk_sol_str)] +
                          [np.random.choice([0, 1])])  # final spin is 'no-action'

    return Graph(graph_name, n_vertices, n_edges, matrix, bk_val, bk_sol)


def load_graph_set(graph_save_loc):
    graphs_test = pickle.load(open(graph_save_loc, 'rb'))

    def graph_to_array(g):
        if type(g) == nx.Graph:
            g = nx.to_numpy_array(g)
        elif type(g) == sp.sparse.csr_matrix:
            g = g.toarray()
        return g

    graphs_test = [graph_to_array(g) for g in graphs_test]
    print('{} target graphs loaded from {}'.format(
        len(graphs_test), graph_save_loc))
    return graphs_test

####################################################
# FILE UTILS
####################################################


def mk_dir(export_dir, quite=False):
    if not os.path.exists(export_dir):
        try:
            os.makedirs(export_dir)
            print('created dir: ', export_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
        except Exception:
            pass
    else:
        print('dir already exists: ', export_dir)
