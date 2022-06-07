# Network-only algorithm

from dataclasses import replace
from network_simulation_utility import SimulateNetwork
from event_sequence_prediction_model.SpacingFeasibilityNew import FeasCheck
import numpy as np
import os
from plotting_utils.GifCreator import SaveToGif

import matplotlib.pyplot as plt

def PlotInequality2D(A, b, x, a, lb, ub, xlabel = 't1', ylabel = 't2', title = 't3 = 120', filename = 'test', folder = 't3'):
    plt.clf()
    xvals = np.linspace(lb[0],ub[0],1000)
    yvals = np.linspace(lb[1],ub[1],(1000))
    xx,yy = np.meshgrid(xvals,yvals)
    def is_feasible(x, y):
        ans = np.ones(x.shape).astype(bool)
        for idx, row in enumerate(A):
            ans &= (row[0] * x + row[1] * y - b[idx,0] <= 0)
        return ans
    plt.imshow( (is_feasible(xx,yy)).astype(int) , 
        extent=(xx.min(),xx.max(),yy.min(),yy.max()),origin="lower", cmap="Blues", alpha = 0.3)
    
    for idx, row in enumerate(A):
        if abs(row).max() < 1e-6:
            continue
        elif abs(row[0]) > abs(row[1]) and abs(row[1]) / abs(row[0]) < 1e-4:
            plt.plot([b[idx,0] / row[0]] * len(yvals), yvals)
        else:     
            plt.plot(xvals, (b[idx, 0] - row[0] * xvals) / row[1])
    plt.plot([x[0] + a, x[0] + a, x[0] - a, x[0] - a, x[0] + a], [x[1] + a, x[1] - a, x[1] - a, x[1] + a, x[1] + a], color='green',linewidth=3, label="Bound")
    plt.xlim(xx.min() - 1,xx.max() + 1)
    plt.ylim(yy.min() - 1, yy.max() + 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel),
    plt.title(title)
    plt.tight_layout()
    plt.savefig('images/{}/{}.png'.format(folder, filename))

def PlotInequalities(A, b, x, a, filename = 'test'):
    x1l, x2l, x3l = list(np.round(x - 20).flatten())
    x1u, x2u, x3u = list(np.round(x + 20).flatten())

    for idx in range(len(A)):
        max_val = np.abs(A[idx,:]).max()
        max_ten_val = np.floor(np.log10(max_val))
        A[idx,:] *= 10 ** (-max_ten_val)
        b[idx] *= 10 ** (-max_ten_val)
    with np.errstate(divide='raise'):
        PlotInequality2D(A[:,(0,1)], b - A[:,2:3].dot(x[2:3]), x[(0,1),:], a, [x1l, x2l], [x1u, x2u], 't1', 't2', 't3 = {:.2f}'.format(x.flatten()[2]), filename, 't3')
        PlotInequality2D(A[:,(1,2)], b - A[:,0:1].dot(x[0:1]), x[(1,2),:], a, [x2l, x3l], [x2u, x3u], 't2', 't3', 't1 = {:.2f}'.format(x.flatten()[2]), filename, 't1')
        PlotInequality2D(A[:,(2,0)], b - A[:,1:2].dot(x[1:2]), x[(2,0),:], a, [x3l, x1l], [x3u, x1u], 't3', 't1', 't2 = {:.2f}'.format(x.flatten()[2]), filename, 't2')

def MakeHistoryPlots(A, b, x_hist, a_hist):
    plt.figure(figsize=(8,8))
    os.makedirs('images/t1', exist_ok=True)
    os.makedirs('images/t2', exist_ok=True)
    os.makedirs('images/t3', exist_ok=True)
    jdx = 0
    for idx in range(len(x_hist)):
        x, a = x_hist[idx], a_hist[idx]
        # use this when using v1 Opti method
        # if idx > 0 and a_hist[idx] <= a_hist[idx - 1]:
        #     continue
        PlotInequalities(A, b, x, a, str('{:05d}'.format(jdx)))
        jdx += 1
    SaveToGif('t2t3', 'images', 'gifs', 10, replace_old=True)
    SaveToGif('t3t1', 'images', 'gifs', 10, replace_old=True)
    SaveToGif('t1t2', 'images', 'gifs', 10, replace_old=True)

def GetCornersOfCube():
    ans = []
    for idx in [-1, 1]:
            for jdx in [-1, 1]:
                for kdx in [-1, 1]:
                    ans.append(np.array([idx,jdx,kdx]).reshape((3,1)))
    return ans

def FindBestSpacingBoundsV2(A, b, x0):
    np.random.seed(0)
    # Optimize the spacing bounds to obtain the largest cubic bounds
    A = A[1:,1:]
    b = b[1:]
    x0 = x0[1:] # Initial guess for center is the simulation spacing
    reduced_A = []
    reduced_b = []
    norm_A = []
    for idx in range(A.shape[0]):
        norm_A_idx = np.linalg.norm(A[idx,:])
        if norm_A_idx > 1e-6:
            reduced_A.append(A[idx,:])
            norm_A.append(norm_A_idx)
            reduced_b.append(b[idx])
    reduced_A = np.asarray(reduced_A)
    reduced_b = np.asarray(reduced_b)
    norm_A = np.asarray(norm_A).reshape((reduced_A.shape[0]), 1)
    sqrt_2 = np.sqrt(2)
    obj_fun = lambda x: (((reduced_A.dot(x) - reduced_b) / norm_A) / sqrt_2)
    max_iter = 1000 # Iteration Count

    step_size = np.std(x0) * 1e-3
    noise_mult = 0.01

    x_hist = []
    a_hist = []

    best_x = x0
    x = x0
    best_a = -np.max(obj_fun(x0))

    x_hist.append(x.copy())
    a_hist.append(best_a.copy())
    print((x0,)) # Starting values
    for iter_idx in range(max_iter):
        distances = obj_fun(x)
        arg_min_dist = np.argmax(distances)
        min_norm_dir = (reduced_A[arg_min_dist] / norm_A[arg_min_dist]).reshape(x.shape)

        x -= min_norm_dir * step_size + np.random.uniform(-1, 1, (x.shape)) * step_size * noise_mult
        best_a = max(best_a, -max(distances))

        if (iter_idx + 1) % (100) == 0:
            x_hist.append(best_x.copy())
            a_hist.append(best_a.copy())
    MakeHistoryPlots(A, b, x_hist, a_hist)
    PlotInequalities(A, b, best_x, best_a, 'best')
    return best_x, best_a, x_hist, a_hist

def FindBestSpacingBounds(A, b, x0):
    np.random.seed(0)
    x_hist = []
    a_hist = []
    # Optimize the spacing bounds to obtain the largest cubic bounds
    A = A[1:,1:]
    b = b[1:]
    x0 = x0[1:] # Initial guess for center is the simulation spacing
    a0 = min(np.abs(np.diff(x0,axis=0))) * 0.001 # Initial guess
    is_feasible = lambda x: (A.dot(x) - b <= 0).all() # Constraint
    max_iter = 10000 # Iteration Count
    a_mult = 1.03 # Multiply side length per step
    a_div = 1.02 # Divide side length per failed step
    a_mult_noise = 0.02
    a_div_noise = 0.02
    off_mult = 0.25 # Amount of displacement of center point if an infeasibility is detected
    off_mult_noise = 1

    best_x, best_a = x0, 0
    x = x0
    a = a0

    x_hist.append(x.copy())
    a_hist.append(a.copy())
    print((x0, a0)) # Starting values
    cornerPointOffsets = GetCornersOfCube()
    for iter_idx in range(max_iter):
        # whether all 8 points are feasible
        feasible = True
        # which directions were infeasible
        offset = np.zeros(x.shape)
        for cornerPointOffset in cornerPointOffsets:

            if not is_feasible(x + cornerPointOffset * a):
                feasible = False
            offset += cornerPointOffset * a
        if feasible:
            a *= (a_mult + np.random.randn() * a_mult_noise)
            if a > best_a:
                best_a = a.copy()
                best_x = x.copy()
        else:
            a /= (a_div + np.random.randn() * a_div_noise)
            if not feasible:
                x -= (offset * off_mult)
                x += np.random.randn(3,1) * a * off_mult_noise
        if (iter_idx + 1) % (max_iter // 1000) == 0:
            x_hist.append(best_x.copy())
            a_hist.append(best_a.copy())
    MakeHistoryPlots(A, b, x_hist, a_hist)
    PlotInequalities(A, b, best_x, best_a, 'best')
    return best_x, best_a, x_hist, a_hist

def getSpacingFromRes(e_q, res):
    entrance_drops = set(range(len(e_q[0])))
    entry_idx = [0] * len(entrance_drops)
    for idx, event in enumerate(e_q):
        drop_pos = event
        for drop in drop_pos:
            if drop_pos[drop] != 'entrance' and drop in entrance_drops:
                entrance_drops.remove(drop)
                entry_idx[drop] = idx - 1
    spacing = []
    for entry in entry_idx:
        spacing.append(res.x[entry])
    return spacing

np.random.seed(0)

grid_dim = 3
source_nodes = [1]
sink_nodes = [10, 12]
rr_list = [1, 2]
drops = [[0,1,1,0]]
iter_count = 1000
branch_config = np.asarray([0]*28)
spacing= np.array(([[0,	-0.16933,	-0.410262,	-0.8505267]]))

var_strng = [1] * 28
# var_strng = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1,]
spacing = np.array(([[0,	-0.16933,	-0.410262,	-0.8505267]]))
params = {
    "branch_config": np.asarray(var_strng),
    "spacing": spacing,
    "length_factor": 0.015,
}
fit, e_q, t_s, s_n = SimulateNetwork(drops, params=params, ret_sequence= True, ret_network= True, gen_images = False)

myFeasChecker = FeasCheck(network=s_n, drops= drops)
results = myFeasChecker.GetEquations(e_q)
A, b = results['A'], results['b']
A = np.array(A).astype(float)
b = np.array(b).astype(float).reshape((len(b), 1))
entry_indices = results['variables']
actual_entry_times = np.array(list(t_s[idx] for idx in entry_indices)).reshape((4,1))

A_x = A.dot(actual_entry_times)

best_x, best_a, x_hist, a_hist = FindBestSpacingBoundsV2(A, b, actual_entry_times)

print("Best Spacing Values (sec):", best_x.flatten())
print("Robustness Value:", best_a)
