import random
import matplotlib.pyplot as plt
from simulation_code import Network
import rl_agent_util
import time
import numpy as np
import pickle

# Written for use in Dual Degree Project, 2021-2022 by J Mahesh, CH17B049, Dept. Chemical Engineering, IIT Madras
# Code for running and comparing the simulation results for a variety of input configurations using Network Construction Model and Random Enumeration Method

# =====================================
# INPUT PARAMETERS
# =====================================

# Shape: Number of Sources x Number of drops per source
resistance_ratios = [1, 2]
input_sequence = np.asarray([
    [1, 0, 0, 1]])

num_drps = np.product(input_sequence.shape)

# Same shape as input sequence
Spacing = np.asarray([
    [0, -0.06, -0.12, -0.18]])

# Shape: Number of branches for the grid dimension

Branch_Config = np.ones((1,28))

source_nodes = [1]

sink_nodes = [10,12]

grid_dim = 3

# This variable no longer has functionality, as the image & video generation logic has been removed from the code
should_gen_images = False

# Node Numbering
#
# 0          5           10 ...
#      3           8        ...
# 1          6           11 ...
#      4           9       ...
# 2          7           12 ...
#
# Branch Numbering
#
# *    2    *           *
#   1
# 0    *          *
#   4
# *    6    *           *
#   5
# 3    *          *
#   7
# *    8    *           *
#
#

# Setting this to false would make the rl_agent enumerate branches in a depth-first manner rather than breadth-first
use_radial_traversal = True

max_itr = 2000
max_seed = 100

overall_avg_fitness_lists = []
overall_max_fitness_lists = []

# ==========================================
# RL Agent: R Realisations over N iterations
# ==========================================
for seed in range(max_seed):
    print("RL Agent: For random seed", seed)    
    rl_agent_1 = rl_agent_util.RlNetworkAgent(grid_dim, 1, epsilon = 0.3, learning_rate= 0.01, baseline_learning_rate= 0.1, seed = seed, use_radial_traversal=use_radial_traversal)
    old_weights = 0
    fitness_list = []
    avg_fitness_list = []
    max_fitness_list = []
    max_fitness = 0.5
    random.seed(seed)
    np.random.seed(seed)

    # Run for max_itr number of iterations
    for itr_count in range(max_itr):
        interm_seed = seed * max_itr + itr_count

        random.seed(interm_seed)
        np.random.seed(interm_seed)
        # Construct the branch configuration using rl_agent
        var_strng = rl_agent_1.get_var_strng(seed = interm_seed)
        var_strng = [var_strng > 0]
        Branch_Config = np.asarray(var_strng, dtype= int)

        # "\1:Code for simulating droplets inside the branches with ideal length"
        network = Network(grid_dim=grid_dim, var_strng=Branch_Config, sim_drops=num_drps,source_nodes=source_nodes, sink_nodes=sink_nodes, arrange=input_sequence, value_spacing=Spacing,
                            length_factor=20 * (10 ** -3), Q_in=10 * 10**(-6)/3600, P_out=1.01325 * 10 ** 5,
                            width_channel=1 * (10 ** -3), height_channel=1 * (10 ** -3), interfacial_tension=45*0.001, rr=2, rr_list = resistance_ratios,
                            alpha=0.9, beta=1.5, vscty=50 * (10 ** -3), toterminate=False,
                            rn1=random.randint(1, 100), rn2=random.randint(1, 100), rn3=random.randint(1, 100),
                            subdue_print = True, output_img_ext = '.png', should_gen_images=should_gen_images)
        
        try:
            start = time.time()
            network.simulate_network_func()
            # print(f"iter{itr_count} TOE = {time.time() - start}")
            # fit.append(network.network_fitness_calc_fn())
            fitness = network.network_fitness_calc_fn()
        # In case simulation fails for any reason, default to minimum fitness value 0.5
        except Exception as e: 
            # print('\nException at iteration', itr_count)
            fitness = 0.5

        # Updating results info, such as max_fitness & avg fitness
        max_fitness = max(max_fitness, fitness)
        max_fitness_list.append(max_fitness)
        
        if ((itr_count+1) % 250 == 0):
            new_weights = np.concatenate([(rl_agent_1.weights + 0.0).flatten(), (rl_agent_1.bias + 0.0).flatten()]).flatten()
            norm_diff = np.linalg.norm(new_weights - old_weights)
            old_weights = new_weights
            print('Iteration %d: Max Fitness = %.4f, Norm Diff = %.4f'%(itr_count,max_fitness,norm_diff))
        fitness_list.append(fitness)
        avg_fitness_list.append(np.mean(fitness_list))
        rl_agent_1.update_weights_single(fitness)
        # rl_agent_2.update_weights_single(fitness)
        # rl_agent_3.update_weights_single(fitness)
    
    # End of iterations for one realisation. Updating averaged results info
    overall_avg_fitness_lists.append(fitness_list)
    overall_max_fitness_lists.append(max_fitness_list)

# End of all iterations
if (should_gen_images):
    network.gen_video()
    # print('Video ouput generation complete. Stored as project.avi.')

rlavg = overall_avg_fitness_lists.copy()
rlmax = overall_max_fitness_lists.copy()

overall_avg_fitness_lists = np.asarray(overall_avg_fitness_lists).mean(axis = 0)
overall_max_fitness_lists = np.asarray(overall_max_fitness_lists).mean(axis = 0)

# Plot RL agent-only results
plt.close()
plt.figure()
plt.plot(overall_avg_fitness_lists, label="Average Fitness")
plt.plot(overall_max_fitness_lists, label="Max Fitness")
plt.legend()
plt.title('RL Agent: Fitness Value over Iterations')
plt.savefig("RLAgent.png")
# plt.show()

rl_agent_avg_fitness = overall_avg_fitness_lists
rl_agent_max_fitness = overall_max_fitness_lists

overall_avg_fitness_lists = []
overall_max_fitness_lists = []

# ==========================================
# Random Enumeration: R Realisations over N iterations
# ==========================================

for seed in range(max_seed):
    print("Random enumeration: For random seed", seed)    
    fitness_list = []
    avg_fitness_list = []
    max_fitness_list = []
    max_fitness = 0.5
    random.seed(seed)
    np.random.seed(seed)
    for itr_count in range(max_itr):
        interm_seed = seed * max_itr + itr_count

        random.seed(interm_seed)
        np.random.seed(interm_seed)
        var_strng = np.random.rand(1,28) > 0.5
        Branch_Config = np.asarray(var_strng, dtype= int)
        # "\1:Code for simulating droplets inside the branches with ideal length"
        network = Network(grid_dim=grid_dim, var_strng=Branch_Config, sim_drops=num_drps,source_nodes=source_nodes, sink_nodes=sink_nodes, arrange=input_sequence, value_spacing=Spacing,
                            length_factor=20 * (10 ** -3), Q_in=10 * 10**(-6)/3600, P_out=1.01325 * 10 ** 5,
                            width_channel=1 * (10 ** -3), height_channel=1 * (10 ** -3), interfacial_tension=45*0.001, rr=2, rr_list = resistance_ratios,
                            alpha=0.9, beta=1.5, vscty=50 * (10 ** -3), toterminate=False,
                            rn1=random.randint(1, 100), rn2=random.randint(1, 100), rn3=random.randint(1, 100),
                            subdue_print = True, output_img_ext = '.png', should_gen_images=should_gen_images)
        try:
            start = time.time()
            network.simulate_network_func()
            # print(f"iter{itr_count} TOE = {time.time() - start}")
            # fit.append(network.network_fitness_calc_fn())
            fitness = network.network_fitness_calc_fn()
        except Exception as e: 
            # print('\nException at iteration', itr_count)
            fitness = 0.5
        max_fitness = max(max_fitness, fitness)
        max_fitness_list.append(max_fitness)
        if ((itr_count+1) % 250 == 0):
            print('Iteration %d: Max Fitness = %.4f, Norm Diff = %.4f'%(itr_count,max_fitness,norm_diff))
        fitness_list.append(fitness)
        avg_fitness_list.append(np.mean(fitness_list))
    overall_avg_fitness_lists.append(fitness_list)
    overall_max_fitness_lists.append(max_fitness_list)

randavg = overall_avg_fitness_lists.copy()
randmax = overall_max_fitness_lists.copy()

overall_avg_fitness_lists = np.asarray(overall_avg_fitness_lists).mean(axis = 0)
overall_max_fitness_lists = np.asarray(overall_max_fitness_lists).mean(axis = 0)

plt.close()

with open("results.pkl", 'wb') as handle:
    pickle.dump([rlavg, rlmax, randavg, randmax], handle)


# Plot Random enumeration-only results
plt.figure()
plt.plot(overall_avg_fitness_lists, label="Average Fitness")
plt.plot(overall_max_fitness_lists, label="Max Fitness")
plt.legend()
plt.title('Random Enumeration: Fitness Value over Iterations')
plt.savefig("Random.png")

# Plot Side-by-side comparison
plt.close()
fig, axs = plt.subplots(1,2, figsize = (15, 7), sharey = True)
axs[0].plot(rl_agent_avg_fitness, label="Average Fitness", linewidth = 3)
axs[0].plot(rl_agent_max_fitness, label="Max Fitness", linewidth = 3)
axs[0].legend()
axs[0].set_title('RL Agent: Fitness Value over Iterations')
axs[0].set_xlabel('Iteration', fontsize = 16)
axs[0].set_ylabel('Fitness', fontsize = 16)

axs[1].plot(overall_avg_fitness_lists, label="Average Fitness", linewidth = 3)
axs[1].plot(overall_max_fitness_lists, label="Max Fitness", linewidth = 3)
axs[1].legend()
axs[1].set_title('Random Enumeration: Fitness Value over Iterations')
axs[1].set_xlabel('Iteration', fontsize = 16)
plt.suptitle(f"Results for sequence [[1,2,2,1]]", fontsize = 24)

[ax.yaxis.set_tick_params(labelbottom=True) for ax in axs]
plt.savefig(f"{input_sequence}_combined.png")