# Adaptive Spacing algorithm

import matplotlib.pyplot as plt
from event_sequence_system import Environment
from network_simulation_utility import SimulateNetwork
from event_sequence_utility import SetUpEnv, GetNetworkPartialSequence, TrainAgentOnEventSequences
from event_sequence_prediction_model.SpacingFeasibilityNew import FeasCheck
import numpy as np
import time

np.random.seed(0)

grid_dim = 3
source_nodes = [1]
sink_nodes = [10, 12]
rr_list = [1, 2]
drops = [[0,1,1,0]]
branch_config = np.asarray([0]*28)

# var_strng = [1] * 28
var_strng = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1,]
spacing = np.array(([[0,	-0.16933,	-0.410262,	-0.8505267]]))
ADA_DEFAULT_PARAMS = {
    "branch_config": np.asarray(var_strng),
    "spacing": spacing,
}

ADA_DROP_EXIT_DICT = {0:0, 1:1}

class RunResults:
    def __init__(self, tot_time, fit_hist, best_fit, best_fit_spacing):
        self.tot_time = tot_time
        self.fit_hist = fit_hist
        self.best_fit = best_fit
        self.best_fit_spacing = best_fit_spacing
    @staticmethod
    def AverageResults(result_list):
        avg_fit_hist = np.mean([result.fit_hist for result in result_list], axis = 0)
        avg_time = np.mean([result.tot_time for result in result_list])
        best_fit_idx = np.argmax([result.best_fit for result in result_list])
        best_fit = result_list[best_fit_idx].best_fit
        best_fit_spacing = result_list[best_fit_idx].best_fit_spacing
        return RunResults(avg_time, avg_fit_hist, best_fit, best_fit_spacing)
    def Describe(self):
        print("Total Time Elapsed: {:.4f}s".format(self.tot_time))
        print("Best Fit: {:.4f}".format(self.best_fit))
        print("Best Fit Spacing: {}".format(self.best_fit_spacing.round(4)))


def AdaptiveSpacingOptimization(drops, params = ADA_DEFAULT_PARAMS, seed = 0, DROP_EXIT_DICT = ADA_DROP_EXIT_DICT):
    env, agent = SetUpEnv(drops, DROP_EXIT_DICT, params, seed = seed)
    droplet_count = sum(len(entry_sequence) for entry_sequence in drops)
    fit_hist = []
    best_fit = 0
    best_fit_spacing = None
    PARTIAL_SEQUENCE_LENGTH = 6
    NUM_PARTIAL_SEQUENCES = 50
    NUM_ATTEMPTS_PER_PARTIAL_SEQ = 20
    startTime = time.time()
    for _ in range(NUM_PARTIAL_SEQUENCES):
        # Create Partial Sequence
        intermediateEventSequence = GetNetworkPartialSequence(agent, env, PARTIAL_SEQUENCE_LENGTH)

        results = FeasCheck(env.network, drops).GetEquations(intermediateEventSequence)
        _, res = FeasCheck(env.network, drops).check_feasibility(intermediateEventSequence, True)

        A, b = results['A'], results['b']
        A = np.array(A).astype(float)
        b = np.array(b).astype(float).reshape((len(b), 1))
        entry_indices = results['variables']

        num_entered = A.shape[1]
        if num_entered == 1:
            raise RuntimeError("At least two droplets should enter the network. Please increase partial sequence length")

        # Calculate spacing from solution
        def expandSolutionIfNotFull(droplet_count, x_sol, A, b, num_entered):
            if num_entered < droplet_count:
                # Need to expand A,
                entry_times = list(x_sol)
                std_droplets = np.std(x_sol)
                while len(entry_times) < droplet_count:
                    n_s, n_d = A.shape
                    entry_times.append(entry_times[-1] + std_droplets)
                    new_A = np.zeros((n_s + 1, n_d + 1))
                    new_b = np.zeros((new_A.shape[0], 1))
                    # New droplet should enter only after the previous droplet. Hence, t_i - t_(i+1) < 0
                    new_A[-1, -2] = 1
                    new_A[-1, -1] = -1
                    new_A[:-1, :-1] = A
                    new_b[:-1, :] = b
                    A = new_A
                    b = new_b
                actual_entry_times = np.asarray(entry_times).reshape((4,1))
            else:
                actual_entry_times = x_sol.reshape((4,1))
            return actual_entry_times, A, b

        def getSpacingFromTime(env: Environment, actual_entry_times):
            """
            Function to extract spacing from feasibility solution
            """
            b = env.network.width_channel
            h = env.network.height_channel
            Q = env.network.Q_in
            beta = env.network.beta
            return (-beta * Q / (b*h) * actual_entry_times).T

        def addZeroToFront(spacing):
            new_spacing = np.zeros((spacing.shape[0] + 1, spacing.shape[1]))
            new_spacing[1:,:] = spacing
            return new_spacing

        actual_entry_times, A, b = expandSolutionIfNotFull(droplet_count, res.x[:num_entered],  A, b, num_entered)
        spacing_values = getSpacingFromTime(env, actual_entry_times)


        x0 = actual_entry_times[1:]
        A_mod = A[1:,1:]
        b_mod = b[1:]
        is_feasible = lambda x: ((A_mod.dot(x) - b_mod < 1e-9).all())

        # reduce A for distance from inequality line calculations
        reduced_A = []
        reduced_b = []
        norm_A = []
        for idx in range(A_mod.shape[0]):
            norm_A_idx = np.linalg.norm(A_mod[idx,:])
            if norm_A_idx > 1e-6:
                reduced_A.append(A_mod[idx,:])
                norm_A.append(norm_A_idx)
                reduced_b.append(b_mod[idx])
        reduced_A = np.asarray(reduced_A)
        reduced_b = np.asarray(reduced_b)
        norm_A = np.asarray(norm_A).reshape((reduced_A.shape[0]), 1)
        sqrt_2 = np.sqrt(2)
        obj_fun = lambda x: (((reduced_A.dot(x) - reduced_b) / norm_A) / sqrt_2)

        is_feasible(x0)
        x_current = x0

        def GetStateActionRewardHistory(raw_eq):
            states = []
            actions = []
            rewards = []
            for raw_state in raw_eq:
                states.append(env.pack_state(raw_state))
            for idx in range(1, len(raw_eq)):
                drop = -1
                for drop in raw_eq[idx]:
                    if raw_eq[idx][drop] != raw_eq[idx - 1][drop]:
                        actions.append(drop)
                        break
                exit = raw_eq[idx][drop] - env.event_sequence.ex0 + 1
                if exit > 0:
                    if DROP_EXIT_DICT[drops[0][drop]] == exit:
                        rewards.append(env.single_drop_reward)
                    else:
                        rewards.append(-env.single_drop_reward)
                else:
                    rewards.append(0)
            return states, actions, rewards

        stepSize = np.std(x_current) * 0.1
        incRatio = 1.01
        decRatio = 0.95

        for _ in range(NUM_ATTEMPTS_PER_PARTIAL_SEQ):
            new_spacing = addZeroToFront(x_current).T
            # Run simulation with spacing
            fit, e_q, t_s, raw_eq, s_n = SimulateNetwork(drops, spacing=new_spacing, params=params, ret_sequence= True, ret_raw_seq= True, ret_network= True, gen_images = False)
            fit_hist.append(fit)
            if fit > best_fit:
                best_fit = fit
                best_fit_spacing = new_spacing
            states, actions, rewards = GetStateActionRewardHistory(raw_eq)
            TrainAgentOnEventSequences(agent, [states], [actions], [rewards])

            # Update x to random nearby value
            dir = np.random.randn(*x_current.shape)
            dir /= np.linalg.norm(dir)
            while True:
                new_x = x_current + dir * stepSize
                if not is_feasible(new_x):
                    stepSize *= decRatio
                    distances = obj_fun(x_current)
                    arg_min_dist = np.argmax(distances)
                    min_norm_dir = (reduced_A[arg_min_dist] / norm_A[arg_min_dist]).reshape(x_current.shape)
                    x_current -= min_norm_dir * stepSize
                else:
                    stepSize *= incRatio
                    break
            x_current = new_x
    print("AdaSpacing: Realisation {} done".format(seed))
    totTime = time.time() - startTime
    return RunResults(totTime, fit_hist, best_fit, best_fit_spacing)


if __name__ == "__main__":
    num_realisations = 50
    resultList = []
    for real_idx in range(num_realisations):
        np.random.seed(real_idx)
        # Initialize
        runResults = AdaptiveSpacingOptimization(drops, seed=real_idx)
        resultList.append(runResults)
    avgResults: RunResults = RunResults.AverageResults(resultList)
    avg_fit_hist = avgResults.fit_hist
    avgResults.Describe()
    plt.plot(avg_fit_hist)
    plt.title('Fitness Over Iterations')
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.savefig('images/temp.png')
    plt.show()