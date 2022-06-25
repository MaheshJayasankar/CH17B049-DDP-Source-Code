import matplotlib.pyplot as plt
import numpy as np
import pickle

def uncumulate(arr):
    n_arr = arr.copy()
    for idx in range(1,len(arr)):
        n = idx + 1
        n_arr[idx] = arr[idx] * n - arr[idx - 1] * (n - 1)
    for idx in range(len(arr)):
        arr[idx] = max(-1,min(1, n_arr[idx]))

droplet_type_list = [[1,2,2,1]]

batcha_files = ["batcha" + str(idx) + ".pkl" for idx in range(1,11)]
batchb_files = ["batchb" + str(idx) + ".pkl" for idx in range(1,11)]

combined_dict = {}

for filename in batcha_files:
    with open(filename, 'rb') as handle:
        load_dict = pickle.load(handle)
        combined_dict.update(load_dict)
    
avg1 = [combined_dict[seed]['avg'] for seed in combined_dict]
for arr in avg1:
    uncumulate(arr)
max1 = [combined_dict[seed]['max'] for seed in combined_dict]

combined_dict = {}
for filename in batchb_files:
    with open(filename, 'rb') as handle:
        load_dict = pickle.load(handle)
        combined_dict.update(load_dict)
    

avg2 = [combined_dict[seed]['avg'] for seed in combined_dict]
max2 = [combined_dict[seed]['max'] for seed in combined_dict]

for arr in avg2:
    uncumulate(arr)

oavg1 = np.mean(np.asarray(avg1), axis=0)
oavg2 = np.mean(np.asarray(avg2), axis=0)

omax1 = np.mean(np.asarray(max1), axis=0)
omax2 = np.mean(np.asarray(max2), axis=0)

amt_iter = 50

rl_avg_vals = np.asarray(avg1)[:,-amt_iter:]
rand_avg_vals = np.asarray(avg2)[:,-amt_iter:]

print("rlavg mean: ", np.mean(rl_avg_vals))
print("rlavg std: ", np.std(rl_avg_vals))

print("randavg mean: ", np.mean(rand_avg_vals))
print("randavg std: ", np.std(rand_avg_vals))

fig, axs = plt.subplots(1,2, figsize = (15, 7), sharey = True)
axs[0].plot(oavg1, label="Average Fitness", linewidth = 2)
axs[0].plot(omax1, label="Max Fitness", linewidth = 2)
axs[0].set_xlabel("Iteration Count", fontsize = 16)
axs[0].set_ylabel("Fitness", fontsize = 16)
axs[0].legend()
axs[0].set_title('Spacing Design: Fitness Value over Iterations')

axs[1].plot(oavg2, label="Average Fitness", linewidth = 2)
axs[1].plot(omax2, label="Max Fitness", linewidth = 2)
axs[1].set_xlabel("Iteration Count", fontsize = 16)
axs[1].legend()
axs[1].set_title('Random Enumeration: Fitness Value over Iterations')
plt.suptitle(f"Results for sequence {droplet_type_list}", fontsize = 24)

[ax.yaxis.set_tick_params(labelbottom=True) for ax in axs]
plt.savefig(f"{droplet_type_list}_combined_consolidated.png")