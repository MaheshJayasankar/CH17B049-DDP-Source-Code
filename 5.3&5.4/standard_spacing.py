# Adaptive Spacing algorithm

import matplotlib.pyplot as plt
from event_sequence_utility import SetUpEnv, NetworkMaxFitness, RunResults
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


def SpacingOptimization(drops, params = ADA_DEFAULT_PARAMS, seed = 0, DROP_EXIT_DICT = ADA_DROP_EXIT_DICT):
    startTime = time.time()
    runResults = NetworkMaxFitness(drops, DROP_EXIT_DICT, params, seed= seed, numIter= 40, returnAllStats=True)
    print("StaSpacing: Realisation {} done".format(seed))
    totTime = time.time() - startTime
    return RunResults(totTime, runResults.fit_hist, runResults.best_fit, runResults.best_fit_spacing)


if __name__ == "__main__":
    num_realisations = 20
    resultList = []
    for real_idx in range(num_realisations):
        np.random.seed(real_idx)
        # Initialize
        runResults = SpacingOptimization(drops, seed=real_idx)
        resultList.append(runResults)
    avgResults: RunResults = RunResults.AverageResults(resultList)
    avg_fit_hist = avgResults.fit_hist
    avgResults.Describe()
    plt.plot(avg_fit_hist)
    plt.title('Fitness Over Iterations')
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.savefig('images/standard_spacing_result.png')
    plt.show()