# Utility functions to optimize droplet spacing for a given network structure

import numpy as np
from event_sequence_agent import RlAgent
from event_sequence_system import Environment, EventSequence

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
        if self.best_fit_spacing is not None:
            print("Best Fit Spacing: {}".format(self.best_fit_spacing.round(4)))

def NetworkMaxFitness(drops:list[list[int]], desired_output: dict[int,int], network_params: dict, seed: int = 0, returnAllStats = False, numIter = 100, debug_print: bool = False):
    rewards = np.zeros((numIter,))
    best_fit = -1
    best_spacing = None
    env, agent = SetUpEnv(drops, desired_output, network_params, seed = seed)
    if not env:
        return -1
    for iteration in range(numIter):
        state = env.reset()
        done = False
        r_sum = 0
        while not done:
            action = agent.pick_action(state)
            reward, next_state, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if abs(reward) == 100:
                r_sum += reward
        fit = env.GetFitness()
        if fit > best_fit:
            best_fit = fit
            best_spacing = env.GetSpacingValues()
        rewards[iteration] = fit
        if debug_print and (iteration + 1) % 25 == 0:
            print(f"StaSpacing Seed {seed}: {iteration + 1}th iteration done.")
    print(f"Max Fitness: {best_fit}")
    if returnAllStats:
        return RunResults(-1, rewards, best_fit, best_spacing)
    return best_fit

def GetNetworkPartialSequence(rlAgent: RlAgent, env: Environment, numSteps: int):
    state = env.reset()
    done = False
    while len(env.GetIntermediateEventSequence()) < numSteps:
        action = rlAgent.pick_action(state)
        reward, next_state, done = env.step(action)
        rlAgent.update(state, action, reward, next_state)
        state = next_state
        if done:
            # Ended prematurely
            return env.GetIntermediateEventSequence()
    return env.GetIntermediateEventSequence()

def SetUpEnv(drops:list[list[int]], desired_output: dict[int,int], network_params: dict, seed = 0):
    env = Environment(drops, desired_output, network_params)
    if not env.created:
        # Couldn't create environment because infeasible network
        return None, None
    agent = RlAgent(env.get_state_dimensions(), env.get_action_dimensions(), epsilon_decay=1e-6, seed = seed)
    return env, agent

def TrainAgentOnEventSequences(rlAgent: RlAgent, stateHistories: list[list[dict[int,int]]], actionHistories: list[list[int]], rewardHistories: list[list[float]]):
    for idx in range(len(stateHistories)):
        sequence = stateHistories[idx]
        rewards = rewardHistories[idx]
        actions = actionHistories[idx]
        for r_idx in range(len(rewards)):
            rlAgent.update(sequence[r_idx], actions[r_idx], rewards[r_idx], sequence[r_idx + 1])
