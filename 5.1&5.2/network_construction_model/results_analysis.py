import pickle
import numpy as np

# Written for use in Dual Degree Project, 2021-2022 by J Mahesh, CH17B049, Dept. Chemical Engineering, IIT Madras
# Analyse the last amt_iter iterations of the two runs and find their average fitness mean & std values.

with open('results.pkl', 'rb') as handle:
    loadarr = pickle.load(handle)

rlavg, rlmax, randavg, randmax = loadarr

amt_iter = 50

rl_avg_vals = np.asarray(rlavg)[:,-amt_iter:]
rand_avg_vals = np.asarray(randavg)[:,-amt_iter:]

print("rlavg mean: ", np.mean(rl_avg_vals))
print("rlavg std: ", np.std(rl_avg_vals))

print("randavg mean: ", np.mean(rand_avg_vals))
print("randavg std: ", np.std(rand_avg_vals))