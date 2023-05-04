from abc import ABC, abstractmethod
from collections import namedtuple
from operator import matmul
import time

import numpy as np
import torch.multiprocessing as mp
from numba import jit, float64, int64

from src.envs.utils import (EdgeType,
                            RewardSignal,
                            ExtraAction,
                            OptimisationTarget,
                            Observable,
                            SpinBasis,
                            DEFAULT_OBSERVABLES,
                            GraphGenerator,
                            RandomGraphGenerator,
                            HistoryBuffer)

# A container for get_result function below. Works just like tuple, but prettier.
ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "is_done", "info"))


class SpinSystemFactory(object):
    '''
    Factory class for returning new SpinSystem.
    '''

    @staticmethod
    def get(graph_generator=None,
            max_steps=20,
            n_sets=5,
            observables=DEFAULT_OBSERVABLES,
            reward_signal=RewardSignal.DENSE,
            extra_action=ExtraAction.PASS,
            optimisation_target=OptimisationTarget.ENERGY,
            spin_basis=SpinBasis.SIGNED,
            norm_rewards=False,
            memory_length=None,  # None means an infinite memory.
            horizon_length=None,  # None means an infinite horizon.
            # None means no punishment for re-visiting states.
            stag_punishment=None,
            # None means no reward for reaching a local minima.
            basin_reward=None,
            # Whether the spins can be flipped more than once (i.e. True-->Georgian MDP).
            reversible_spins=True,
            init_snap=None,
            seed=None):

        if graph_generator.biased:
            return SpinSystemBiased(graph_generator, max_steps, n_sets,
                                    observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                    norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                    reversible_spins,
                                    init_snap, seed)
        else:
            return SpinSystemUnbiased(graph_generator, max_steps, n_sets,
                                      observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                      norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                      reversible_spins,
                                      init_snap, seed)


class SpinSystemBase(ABC):
    '''
    SpinSystemBase implements the functionality of a SpinSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''

    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class action_space():
        def __init__(self, n_actions):
            self.n = n_actions
            self.actions = np.arange(self.n)

        def sample(self, n=1):
            return np.random.choice(self.actions, n)

    class observation_space():
        def __init__(self, n_spins, n_observables):
            self.shape = [n_spins, n_observables]

    def __init__(self,
                 graph_generator=None,
                 max_steps=20,
                 n_sets=5,
                 observables=DEFAULT_OBSERVABLES,
                 reward_signal=RewardSignal.DENSE,
                 extra_action=ExtraAction.PASS,
                 optimisation_target=OptimisationTarget.ENERGY,
                 spin_basis=SpinBasis.MULTIPLE,
                 norm_rewards=False,
                 memory_length=None,  # None means an infinite memory.
                 horizon_length=None,  # None means an infinite horizon.
                 stag_punishment=None,
                 basin_reward=None,
                 reversible_spins=False,
                 init_snap=None,
                 seed=None):
        '''
        Init method.

        Args:
            graph_generator: A GraphGenerator (or subclass thereof) object.
            max_steps: Maximum number of steps before termination.
            reward_signal: RewardSignal enum determining how and when rewards are returned.
            extra_action: ExtraAction enum determining if and what additional action is allowed,
                          beyond simply flipping spins.
            init_snap: Optional snapshot to load spin system into pre-configured state for MCTS.
            seed: Optional random seed.
        '''

        if seed != None:
            np.random.seed(seed)

        self.n_sets = n_sets
        print(f"Spin system created with {self.n_sets}")

        # Ensure first observable is the spin state.
        # This allows us to access the spins as self.state[0,:self.n_spins.]
        assert observables[0] == Observable.SPIN_STATE, "First observable must be Observation.SPIN_STATE."

        self.observables = list(enumerate(observables))
        self.extra_action = extra_action

        if graph_generator != None:
            assert isinstance(
                graph_generator, GraphGenerator), "graph_generator must be a GraphGenerator implementation."
            self.gg = graph_generator
        else:
            # provide a default graph generator if one is not passed
            self.gg = RandomGraphGenerator(n_spins=20,
                                           edge_type=EdgeType.DISCRETE,
                                           biased=False,
                                           extra_action=(extra_action != extra_action.NONE))

        self.n_spins = self.gg.n_spins  # Total number of spins in episode
        self.max_steps = max_steps  # Number of actions before reset

        self.reward_signal = reward_signal
        self.norm_rewards = norm_rewards

        self.n_actions = self.n_spins
        if extra_action != ExtraAction.NONE:
            self.n_actions += 1

        self.action_space = self.action_space(self.n_actions)
        self.observation_space = self.observation_space(
            self.n_spins, len(self.observables))

        self.current_step = 0

        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()
            self.bias = None

        self.optimisation_target = optimisation_target
        self.spin_basis = spin_basis

        self.memory_length = memory_length
        self.horizon_length = horizon_length if horizon_length is not None else self.max_steps
        self.stag_punishment = stag_punishment
        self.basin_reward = basin_reward
        self.reversible_spins = reversible_spins

        self.reset()

        self.score = self.calculate_score()
        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_spins = self.state[0, :]

        if init_snap != None:
            self.load_snapshot(init_snap)

    def reset(self, spins=None):
        """
        Explanation here
        """
        self.current_step = 0
        if self.gg.biased:
            # self.matrix, self.bias = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix, self.bias = self.gg.get()
        else:
            # self.matrix = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix = self.gg.get()
        self._reset_graph_observables()

        spinsOne = np.array([1] * self.n_spins)
        local_rewards_available = np.array(self.get_immediate_rewards_available(
            spinsOne))[:, 0]
        local_rewards_available = local_rewards_available[np.nonzero(
            local_rewards_available)]
        if local_rewards_available.size == 0:
            # We've generated an empty graph, this is pointless, try again.
            self.reset()
        else:
            self.max_local_reward_available = np.max(local_rewards_available)

        self.state = self._reset_state(spins)
        self.score = self.calculate_score()

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_obs_score = self.score
        self.best_spins = self.state[0, :self.n_spins].copy()
        self.best_obs_spins = self.state[0, :self.n_spins].copy()

        if self.memory_length is not None:
            self.score_memory = np.array(
                [self.best_score] * self.memory_length)
            self.spins_memory = np.array(
                [self.best_spins] * self.memory_length)
            self.idx_memory = 1

        self._reset_graph_observables()

        if self.stag_punishment is not None or self.basin_reward is not None:
            self.history_buffer = HistoryBuffer()

        return self.get_observation()

    def _reset_graph_observables(self):
        # Reset observed adjacency matrix
        if self.extra_action != self.extra_action.NONE:
            # Pad adjacency matrix for disconnected extra-action spins of value 0.
            self.matrix_obs = np.zeros(
                (self.matrix.shape[0] + 1, self.matrix.shape[0] + 1))
            self.matrix_obs[:-1, :-1] = self.matrix
        else:
            self.matrix_obs = self.matrix

        # Reset observed bias vector,
        if self.gg.biased:
            if self.extra_action != self.extra_action.NONE:
                # Pad bias for disconnected extra-action spins of value 0.
                self.bias_obs = np.concatenate((self.bias, [0]))
            else:
                self.bias_obs = self.bias

    def _reset_state(self, spins=None):
        state = np.zeros((self.observation_space.shape[1], self.n_actions))

        if spins is None:
            if self.reversible_spins:
                # For reversible spins, initialise randomly to {+1,-1}.
                # state[0, :self.n_spins] = 2 * \
                #     np.random.randint(2, size=self.n_spins) - 1

                # For reversible spins with n_sets, initialize to integers between 0 and n_sets-1 included.
                state[0, :self.n_spins] = np.array(
                    np.random.randint(self.n_sets, size=self.n_spins))
            else:
                # For irreversible spins, initialise all to +1 (i.e. allowed to be flipped).
                state[0, :self.n_spins] = 1
        else:
            state[0, :] = self._format_spins_to_signed(spins)

        state = state.astype('float')
        immediate_rewards_available = np.array(
            self.get_immediate_rewards_available(spins=state[0, :self.n_spins]))
        target_sets = immediate_rewards_available[:, 1]
        immediate_rewards_available = immediate_rewards_available[:, 0]

        # If any observables other than "immediate energy available" require setting to values other than
        # 0 at this stage, we should use a 'for k,v in enumerate(self.observables)' loop.
        for idx, obs in self.observables:
            if obs == Observable.IMMEDIATE_REWARD_AVAILABLE:
                state[idx, :self.n_spins] = immediate_rewards_available / \
                    self.max_local_reward_available
            elif obs == Observable.LOCAL_DIVERSITY:
                state[idx, :self.n_spins] = self.get_local_diversity(
                    self.matrix, state[0, :], target_sets)
            elif obs == Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                state[idx, :self.n_spins] = 1 - \
                    np.sum(immediate_rewards_available <= 0) / self.n_spins
        return state

    def _get_spins(self, basis=SpinBasis.SIGNED):
        spins = self.state[0, :self.n_spins]

        if basis == SpinBasis.SIGNED:
            pass
        elif basis == SpinSystemBiased:
            print("To be CHANGED, _get_spins, SpinBasis.MULTIPLE")
            # convert {1,-1} --> {0,1}
            spins[0, :] = (1 - spins[0, :]) / 2
        else:
            raise NotImplementedError("Unrecognised SpinBasis")

        return spins

    def calculate_best_energy(self):
        if self.n_spins <= 10:
            # Generally, for small systems the time taken to start multiple processes is not worth it.
            res = self.calculate_best_brute()

        else:
            # Start up processing pool
            n_cpu = int(mp.cpu_count()) / 2

            pool = mp.Pool(mp.cpu_count())

            # Split up state trials across the number of cpus
            iMax = 2 ** (self.n_spins)
            args = np.round(np.linspace(
                0, np.ceil(iMax / n_cpu) * n_cpu, n_cpu + 1))
            arg_pairs = [list(args) for args in zip(args, args[1:])]

            # Try all the states.
            #             res = pool.starmap(self._calc_over_range, arg_pairs)
            try:
                res = pool.starmap(self._calc_over_range, arg_pairs)
                # Return the best solution,
                idx_best = np.argmin([e for e, s in res])
                res = res[idx_best]
            except Exception as e:
                # Falling back to single-thread implementation.
                # res = self.calculate_best_brute()
                res = self._calc_over_range(0, 2 ** (self.n_spins))
            finally:
                # No matter what happens, make sure we tidy up after outselves.
                pool.close()

            if self.spin_basis == SpinBasis.BINARY:
                # convert {1,-1} --> {0,1}
                best_score, best_spins = res
                best_spins = (1 - best_spins) / 2
                res = best_score, best_spins

            if self.optimisation_target == OptimisationTarget.CUT:
                best_energy, best_spins = res
                best_cut = self.calculate_cut(best_spins)
                print("best_cut", best_cut)
                res = best_cut, best_spins
            elif self.optimisation_target == OptimisationTarget.ENERGY:
                pass
            else:
                raise NotImplementedError()

        return res

    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def step(self, action):
        done = False

        rew = 0  # Default reward to zero.
        randomised_spins = False
        self.current_step += 1

        if self.current_step > self.max_steps:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError

        new_state = np.copy(self.state)

        ############################################################
        # 1. Performs the action and calculates the score change. #
        ############################################################
        if action == self.n_spins:
            if self.extra_action == ExtraAction.PASS:
                delta_score = 0
            if self.extra_action == ExtraAction.RANDOMISE:
                # Randomise the spin configuration.
                randomised_spins = True
                random_actions = np.random.choice([1, -1], self.n_spins)
                new_state[0, :] = self.state[0, :] * random_actions
                new_score = self.calculate_score(new_state[0, :])
                delta_score = new_score - self.score
                self.score = new_score
        else:
            # Perform the action and calculate the score change.
            # new_state[0,action] = -self.state[0,action]
            if self.gg.biased:
                delta_score, target_set = self._calculate_score_change(
                    new_state[0, :self.n_spins], self.matrix, self.bias, action)
            else:
                delta_score, target_set = self._calculate_score_change(
                    new_state[0, :self.n_spins], self.matrix, action)
            self.score += delta_score
            new_state[0, action] = target_set

        #############################################################################################
        # 2. Calculate reward for action and update anymemory buffers.                              #
        #   a) Calculate reward (always w.r.t best observable score).                              #
        #   b) If new global best has been found: update best ever score and spin parameters.      #
        #   c) If the memory buffer is finite (i.e. self.memory_length is not None):                #
        #          - Add score/spins to their respective buffers.                                  #
        #          - Update best observable score and spins w.r.t. the new buffers.                #
        #      else (if the memory is infinite):                                                    #
        #          - If new best has been found: update best observable score and spin parameters. #                                                                        #
        #############################################################################################

        self.state = new_state
        immeditate_rewards_available = np.array(
            self.get_immediate_rewards_available())
        target_sets = immeditate_rewards_available[:, 1]
        immeditate_rewards_available = immeditate_rewards_available[:, 0]

        if self.score > self.best_obs_score:
            if self.reward_signal == RewardSignal.BLS:
                rew = self.score - self.best_obs_score
            elif self.reward_signal == RewardSignal.CUSTOM_BLS:
                rew = self.score - self.best_obs_score
                rew = rew / (rew + 0.1)

        if self.reward_signal == RewardSignal.DENSE:
            rew = delta_score
        elif self.reward_signal == RewardSignal.SINGLE and done:
            rew = self.score - self.init_score

        if self.norm_rewards:
            rew /= self.n_spins

        if self.stag_punishment is not None or self.basin_reward is not None:
            visiting_new_state = self.history_buffer.update(action)

        if self.stag_punishment is not None:
            if not visiting_new_state:
                rew -= self.stag_punishment

        if self.basin_reward is not None:
            if np.all(immeditate_rewards_available <= 0):
                # All immediate score changes are +ive <--> we are in a local minima.
                if visiting_new_state:
                    # #####TEMP####
                    # if self.reward_signal != RewardSignal.BLS or (self.score > self.best_obs_score):
                    # ####TEMP####
                    rew += self.basin_reward

        if self.score > self.best_score:
            self.best_score = self.score
            self.best_spins = self.state[0, :self.n_spins].copy()

        if self.memory_length is not None:
            # For case of finite memory length.
            self.score_memory[self.idx_memory] = self.score
            self.spins_memory[self.idx_memory] = self.state[0, :self.n_spins]
            self.idx_memory = (self.idx_memory + 1) % self.memory_length
            self.best_obs_score = self.score_memory.max()
            self.best_obs_spins = self.spins_memory[self.score_memory.argmax()].copy(
            )
        else:
            self.best_obs_score = self.best_score
            self.best_obs_spins = self.best_spins.copy()
        #############################################################################################
        # 3. Updates the state of the system (except self.state[0,:] as this is always the spin     #
        #    configuration and has already been done.                                               #
        #   a) Update self.state local features to reflect the chosen action.                       #                                                                  #
        #   b) Update global features in self.state (always w.r.t. best observable score/spins)     #
        #############################################################################################

        for idx, observable in self.observables:

            ### Local observables ###
            if observable == Observable.IMMEDIATE_REWARD_AVAILABLE:
                self.state[idx, :self.n_spins] = immeditate_rewards_available / \
                    self.max_local_reward_available
            elif observable == Observable.LOCAL_DIVERSITY:
                self.state[idx, :self.n_spins] = self.get_local_diversity(
                    self.matrix, self.state[0, :], target_sets)

            elif observable == Observable.TIME_SINCE_FLIP:
                self.state[idx, :] += (1. / self.max_steps)
                if randomised_spins:
                    self.state[idx, :] = self.state[idx, :] * \
                        (random_actions > 0)
                else:
                    self.state[idx, action] = 0

            ### Global observables ###
            elif observable == Observable.EPISODE_TIME:
                self.state[idx, :] += (1. / self.max_steps)

            elif observable == Observable.TERMINATION_IMMANENCY:
                # Update 'Immanency of episode termination'
                self.state[idx, :] = max(
                    0, ((self.current_step - self.max_steps) / self.horizon_length) + 1)

            elif observable == Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                self.state[idx, :] = 1 - \
                    np.sum(
                        immeditate_rewards_available <= 0) / self.n_spins

            elif observable == Observable.DISTANCE_FROM_BEST_SCORE:
                self.state[idx, :] = np.abs(
                    self.score - self.best_obs_score) / self.max_local_reward_available

            elif observable == Observable.DISTANCE_FROM_BEST_STATE:
                self.state[idx, :self.n_spins] = np.count_nonzero(
                    self.best_obs_spins[:self.n_spins] - self.state[0, :self.n_spins])

        #############################################################################################
        # 4. Check termination criteria.                                                            #
        #############################################################################################
        if self.current_step == self.max_steps:
            # Maximum number of steps taken --> done.
            # print("Done : maximum number of steps taken")
            done = True

        if not self.reversible_spins:
            if len((self.state[0, :self.n_spins] > 0).nonzero()[0]) == 0:
                # If no more spins to flip --> done.
                # print("Done : no more spins to flip")
                done = True

        return (self.get_observation(), rew, done, None)

    def get_observation(self):
        state = self.state.copy()
        if self.spin_basis == SpinBasis.BINARY:
            # convert {1,-1} --> {0,1}
            state[0, :] = (1-state[0, :])/2

        if self.gg.biased:
            return np.vstack((state, self.matrix_obs, self.bias_obs))
        else:
            return np.vstack((state, self.matrix_obs))

    def get_immediate_rewards_available(self, spins=None):
        if spins is None:
            spins = self._get_spins()

        if self.optimisation_target == OptimisationTarget.ENERGY:
            immediate_reward_function = lambda *args: -1 * \
                self._get_immeditate_energies_avaialable_jit(*args)
        elif self.optimisation_target == OptimisationTarget.CUT:
            # immediate_reward_function = self._get_immeditate_cuts_avaialable_jit
            immediate_reward_function = self._get_immeditate_cuts_avaialable_jit
        else:
            raise NotImplementedError(
                "Optimisation target {} not recognised.".format(self.optimisation_ta))

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        if self.gg.biased:
            bias = self.bias.astype('float64')
            return immediate_reward_function(spins, matrix, bias)
        else:
            return immediate_reward_function(spins, matrix, self.n_sets)

    def get_allowed_action_states(self):
        if self.reversible_spins:
            # If MDP is reversible, both actions are allowed.
            if self.spin_basis == SpinBasis.BINARY:
                return (0, 1)
            elif self.spin_basis == SpinBasis.SIGNED:
                return (1, -1)
        else:
            # If MDP is irreversible, only return the state of spins that haven't been flipped.
            if self.spin_basis == SpinBasis.BINARY:
                return 0
            if self.spin_basis == SpinBasis.SIGNED:
                return 1

    def calculate_score(self, spins=None):
        if self.optimisation_target == OptimisationTarget.CUT:
            score = self.calculate_cut(spins)
        elif self.optimisation_target == OptimisationTarget.ENERGY:
            score = -1.*self.calculate_energy(spins)
        else:
            raise NotImplementedError
        return score

    def _calculate_score_change(self, new_spins, matrix, action):
        if self.optimisation_target == OptimisationTarget.CUT:
            # delta_score = self._calculate_cut_change(new_spins, matrix, action)
            delta_score = self._calculate_cut_change(
                new_spins, matrix, action, self.n_sets)
        elif self.optimisation_target == OptimisationTarget.ENERGY:
            delta_score = -1. * \
                self._calculate_energy_change(new_spins, matrix, action)
        else:
            raise NotImplementedError
        return delta_score

    def _format_spins_to_signed(self, spins):
        if self.spin_basis == SpinBasis.BINARY:
            if not np.isin(spins, [0, 1]).all():
                raise Exception(
                    "SpinSystem is configured for binary spins ([0,1]).")
            # Convert to signed spins for calculation.
            spins = 2 * spins - 1
        elif self.spin_basis == SpinBasis.SIGNED:
            if not np.isin(spins, [-1, 1]).all():
                raise Exception(
                    "SpinSystem is configured for signed spins ([-1,1]).")
        return spins

    @abstractmethod
    def calculate_energy(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def calculate_cut(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def get_best_cut(self):
        raise NotImplementedError

    @abstractmethod
    def _calc_over_range(self, i0, iMax):
        raise NotImplementedError

    @abstractmethod
    def _calculate_energy_change(self, new_spins, matrix, action):
        raise NotImplementedError

    @abstractmethod
    def _calculate_cut_change(self, new_spins, matrix, action):
        raise NotImplementedError

##########
# Classes for implementing the calculation methods with/without biases.
##########


class SpinSystemUnbiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        else:
            spins = self._format_spins_to_signed(spins)

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        return self._calculate_energy_jit(spins, matrix)

    # def calculate_cut(self, spins=None):
        # if spins is None:
        #     spins = self._get_spins()
        # else:
        #     spins = self._format_spins_to_signed(spins)

    #     return (1/4) * np.sum(np.multiply(self.matrix, 1 - np.outer(spins, spins)))

    def calculate_cut(self, spins=None):
        # adj_matrix, set_membership):
        """
        Computes the cut value of a graph given its adjacency matrix and a list of node set memberships.

        Args:
            adj_matrix (np.ndarray): The adjacency matrix of the input graph. Uses the matrix from the class.

        Returns:
            float: The cut value of the input graph.
        """
        # Compute the cut value by summing over the entries of the adjacency matrix
        # where one node is in a set and the other one is in another set
        if spins is None:
            spins = self._get_spins()
        else:
            raise NotImplementedError()
            spins = self._format_spins_to_signed(spins)

        cut_value = 0
        for i in range(self.matrix.shape[0]):
            for j in range(i+1, self.matrix.shape[1]):
                if spins[i] != spins[j]:
                    cut_value += self.matrix[i][j]
        return cut_value

    def get_best_cut(self):
        if self.optimisation_target == OptimisationTarget.CUT:
            return self.best_score
        else:
            raise NotImplementedError(
                "Can't return best cut when optimisation target is set to energy.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix)

    @staticmethod
    @jit(float64(float64[:], float64[:, :], int64), nopython=True)
    def _calculate_energy_change(new_spins, matrix, action):
        return -2 * new_spins[action] * matmul(new_spins.T, matrix[:, action])

    # @staticmethod
    # @jit(float64(float64[:], float64[:, :], int64), nopython=True)
    # def _calculate_cut_change(new_spins, matrix, action):
    #     return -1 * new_spins[action] * matmul(new_spins.T, matrix[:, action])

    @staticmethod
    @jit('Tuple((float64, int64))(float64[:], float64[:, :], int64, int64)', nopython=True)
    # @jit(nopython=True)
    def _calculate_cut_change(set_membership, adj_matrix, action, n_sets):
        """
        Given a vertex (action), computes all the possible changes and returns the best set and the associated reward.

        Args:
            set_membership (np.ndarray): A list or array indicating the current set membership of each node.
            adj_matrix (np.ndarray): The adjacency matrix of the input graph.
            action (int): The index of the vertex being changed.
            n_sets (int): The number of sets the graph is partitionned in.

        Returns:
            float, int: The change of cut value and the best target set.
        """

        best_immediate_reward = np.NINF
        # Will be updated in the loop. The set that gets the best reward for the current action.
        target_set = 0
        weights = adj_matrix[action]

        # Iterate through the potential new sets
        for set_label in range(n_sets):
            if set_label == set_membership[action]:
                continue
            immediate_reward = 0
            # Iterate through the edges of the action
            for vertex_idx, edge_weight in enumerate(weights):
                if edge_weight == 0:
                    continue
                if set_membership[vertex_idx] == set_label:
                    immediate_reward -= edge_weight
                elif set_membership[vertex_idx] == set_membership[action]:
                    immediate_reward += edge_weight
            # Update best reward
            if immediate_reward > best_immediate_reward:
                best_immediate_reward = immediate_reward
                target_set = set_label

        return best_immediate_reward, target_set

    # @staticmethod
    # @jit('Tuple((float64, int64))(float64[:], float64[:, :], int64, int64)', nopython=True)
    # # @jit(nopython=True)
    # def _calculate_cut_change(set_membership, adj_matrix, action, n_sets):
    #     """
    #     Given a vertex (action), computes the possible changes between the current set and a newly random set.
    #      Returns the reward correponding to a switch to this random set.

    #     Args:
    #         set_membership (np.ndarray): A list or array indicating the current set membership of each node.
    #         adj_matrix (np.ndarray): The adjacency matrix of the input graph.
    #         action (int): The index of the vertex being changed.
    #         n_sets (int): The number of sets the graph is partitionned in.

    #     Returns:
    #         float, int: The change of cut value and the best target set.
    #     """

    #     immediate_reward = 0
    #     weights = adj_matrix[action]
    #     current_set = set_membership[action]
    #     retry = True
    #     while retry:
    #         target_set = np.random.choice(n_sets)
    #         if target_set != current_set:
    #             retry = False
    #     for vertex_idx, edge_weight in enumerate(weights):
    #         # Don't change cut value if there is no edge.
    #         if edge_weight == 0:
    #             continue
    #         # Decrease reward if there is an edge to the target set.
    #         if set_membership[vertex_idx] == target_set:
    #             immediate_reward -= edge_weight
    #         # Increase reward if there is an edge to the original set.
    #         elif set_membership[vertex_idx] == current_set:
    #             immediate_reward += edge_weight

    #     return immediate_reward, target_set

    @staticmethod
    @jit(float64(float64[:], float64[:, :]), nopython=True)
    def _calculate_energy_jit(spins, matrix):
        return - matmul(spins.T, matmul(matrix, spins)) / 2

    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
            current_energy = - matmul(spins.T, matmul(matrix, spins)) / 2
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins

    @staticmethod
    @jit(float64[:](float64[:], float64[:, :]), nopython=True)
    def _get_immeditate_energies_avaialable_jit(spins, matrix):
        return 2 * spins * matmul(matrix, spins)

    # @staticmethod
    # @jit(float64[:](float64[:], float64[:, :]), nopython=True)
    # def _get_immeditate_cuts_avaialable_jit(spins, matrix):
    #     return spins * matmul(matrix, spins)

    @staticmethod
    # @jit(float64[:](float64[:], float64[:, :], float64), nopython=True)
    def _get_immeditate_cuts_avaialable_jit(set_membership, adj_matrix, n_sets):
        """
        Computes all possible cut changes and returns the cut change with the associated set.

        Args:
            set_membership (np.ndarray): A list or array indicating the current set membership of each node.
            adj_matrix (np.ndarray): The adjacency matrix of the input graph.
            n_sets (int): The number of sets the graph is partitionned in.

        Returns:
            list(tuple(float, int)): The positive change of cut value and corresponding best set. 
                                    Positive means increased cut value. Negative means decreased cut value.
        """
        # immediate_cuts_available = []
        # for action in range(len(set_membership)):
        #     best_immediate_reward, target_set = SpinSystemUnbiased._calculate_cut_change(
        #         set_membership, adj_matrix, action, n_sets)
        #     # print(rew, target)
        #     immediate_cuts_available.append(
        #         (best_immediate_reward, target_set))

        return [SpinSystemUnbiased._calculate_cut_change(
            set_membership, adj_matrix, action, n_sets) for action in range(len(set_membership))]

    @staticmethod
    @jit('float64[:](float64[:, :], float64[:], float64[:])', nopython=True)
    def get_local_diversity(adj_matrix, current_sets, target_sets):
        """
        The difference between the number of nodes in the target set and the current set

        Args:
            adj_matrix (np.ndarray): The adjacency matrix of the input graph.
            current_sets (np.ndarray): An array of the current set of each node.
            target_sets (np.ndarray): An array of the target set of each node. The target set provides the best change.

        Returns:
            np.ndarray: The local diversity of each node
        """
        n_spins = len(adj_matrix)
        local_diversity = np.zeros(n_spins)
        for node in range(n_spins):
            for idx, weight in enumerate(adj_matrix[node]):
                if weight != 0 and target_sets[idx] == current_sets[node]:
                    local_diversity[node] += 1
                elif weight != 0 and target_sets[idx] == target_sets[node]:
                    local_diversity[node] -= 1

        return local_diversity


class SpinSystemBiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if type(spins) == type(None):
            spins = self._get_spins()

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')

        return self._calculate_energy_jit(spins, matrix, bias)

    def calculate_cut(self, spins=None):
        raise NotImplementedError(
            "MaxCut not defined/implemented for biased SpinSystems.")

    def get_best_cut(self):
        raise NotImplementedError(
            "MaxCut not defined/implemented for biased SpinSystems.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix, bias)

    @staticmethod
    @jit(nopython=True)
    def _calculate_energy_change(new_spins, matrix, bias, action):
        return 2 * new_spins[action] * (matmul(new_spins.T, matrix[:, action]) + bias[action])

    @staticmethod
    @jit(nopython=True)
    def _calculate_cut_change(new_spins, matrix, bias, action):
        raise NotImplementedError(
            "MaxCut not defined/implemented for biased SpinSystems.")

    @staticmethod
    @jit(nopython=True)
    def _calculate_energy_jit(spins, matrix, bias):
        return matmul(spins.T, matmul(matrix, spins))/2 + matmul(spins.T, bias)

    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix, bias):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
            current_energy = - \
                (matmul(spins.T, matmul(matrix, spins))/2 + matmul(spins.T, bias))
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_energies_avaialable_jit(spins, matrix, bias):
        return - (2 * spins * (matmul(matrix, spins) + bias))

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_cuts_avaialable_jit(spins, matrix, bias):
        raise NotImplementedError(
            "MaxCut not defined/implemented for biased SpinSystems.")
