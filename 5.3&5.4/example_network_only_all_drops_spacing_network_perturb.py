# Both spacing and network random seeds are perturbed for more robust solutions

from network_construction_agent import RlNetworkAgent
from network_simulation_utility import IsBranchConfigFeasible, SimulateNetwork
import numpy as np

np.random.seed(0)
def all_permute(n, l):
    ans = []
    def extend(sofar, n, rem_l):
        if rem_l == 0:
            ans.append(sofar.copy())
            return
        for idx in range(n):
            sofar.append(idx)
            extend(sofar, n, rem_l - 1)
            sofar.pop()
    extend([], n, l)
    return ans

grid_dim = 3
source_nodes = [1]
sink_nodes = [10, 12]
rr_list = [1, 2]
drop_perm_list = all_permute(2,4)
best_fitness = [0] * len(drop_perm_list)
best_networks = [[] for idx in range(len(drop_perm_list))] 
iter_count = 1000
idx = 0

net_perturb_count = 5
spa_perturb_count = 5
spacing= np.array(([[0,	-0.16933,	-0.410262,	-0.8505267]]))

def perturb_spacing(spacing: np.ndarray, magnitude = 0.05, seed = 0):
    my_state = np.random.get_state()
    np.random.seed(seed)
    spacing = spacing.copy()
    for idx in range(spacing.shape[0]):
        diff = np.diff(spacing[idx,:])
        if len(diff) == 0:
            mean_diff = spacing[idx, 0]
        else:
            mean_diff = np.mean(diff)
        for jdx in range(min(spacing.shape[1] - 1, 1), spacing.shape[1]):
            spacing[idx, jdx] += (1 - np.random.rand()) * 2 * magnitude * mean_diff

    np.random.set_state(my_state)
    return spacing


for idx,drops in enumerate(drop_perm_list):
    rl_agent = RlNetworkAgent(grid_dim, 1, epsilon = 0.3, learning_rate= 0.01, baseline_learning_rate= 0.1, seed = 0, use_radial_traversal=True)
    for iter_idx in range(iter_count):    
        branch_config = rl_agent.get_var_strng(seed = (idx + 1) * iter_count + iter_idx)
        if IsBranchConfigFeasible(grid_dim, branch_config, source_nodes, sink_nodes):
            fitness = 0
            for jdx in range(net_perturb_count):
                for kdx in range(spa_perturb_count):
                    params = {"branch_config": branch_config,
                    "rn_list": (1 + jdx,2 + 2 * jdx,3 + 3 * jdx)}
                    fitness += SimulateNetwork([drops], params=params, spacing = perturb_spacing(spacing, seed = kdx))
            fitness /= (net_perturb_count * spa_perturb_count)
        else:
            fitness = 0
        if fitness > best_fitness[idx]:
            best_fitness[idx] = fitness
            best_networks[idx] = branch_config
        if fitness >= 1:
            break
        rl_agent.update_weights_single(fitness)
        if (iter_idx + 1) % 100 == 0:
            print(f"{iter_idx + 1} Iteration completed")
    print("Drop Sequence:")
    print(drops)
    print("Best Fitness:")
    print(best_fitness[idx])
    print("Best Network:")
    print(best_networks[idx], "\n")
        

