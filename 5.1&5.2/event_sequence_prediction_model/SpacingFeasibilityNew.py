# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:30:05 2021

@author: arune
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 22:23:25 2020

@author: admin
"""

import numpy as np
from NetworkSimulation_4 import Network
import matplotlib.pyplot as plt
import random
import copy
from sympy import symbols, solve
from anytree import Node, RenderTree
from collections import OrderedDict
from sympy import *
from scipy import optimize
## Importing the network
A=[];
while len(A)<1:
    rn1=18#random.randint(1,100)
    rn2=27#random.randint(1,100)
    rn3=46#random.randint(1,100)
    A.append([rn1,rn2,rn3])
Spacing= np.array(([[ 0.        ,-0.022719, -0.024844, -0.032665]]))
                        
for jdx in range(len(A)):    
    network = Network(grid_dim =3,var_strng=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],sim_drops=4,
                          Droplet_frqncy=0.05,source_nodes=[1],sink_nodes=[10,12],length_factor=1000 * (10 ** -6),Q_in=1* 10**(-6)/3600, P_out=1.01325 *10** 5 ,width_channel=200 * (10 ** -6),height_channel=200 * (10 ** -6),interfacial_tension=45*0.001,rr=1,arrange=np.array(([[0,0,1,0]])),
                          alpha=0.9,beta=1.4,vscty=50 * (10 ** -3),value_spacing=Spacing,toterminate=False,rn1=A[jdx][0],rn2=A[jdx][1],rn3=A[jdx][2])

network.node_branch_info()
network.node_connector_matrix(network.var_strng)
network.calc_length_branches()
network.remove_branches()
network.calc_length_branches()
network.incidence_matrix_gen()
network.branch_connectivity_fn()
length_branches=network.calc_length_branches()


Q_inlet=network.Q_in
vel_inlet=network.beta*(network.Q_in/(network.width_channel*network.height_channel))

# Event_8= {0: 6, 1: 12, 2: 'Exit', 3: 4}
# Event_7= {0: 6, 1: 12, 2: 'Exit', 3: 'entrance'}
# Event_6= {0: 6, 1: 12, 2: 15, 3: 'entrance'}
# Event_5= {0: 6, 1: 12, 2: 9, 3: 'entrance'}
# Event_4= {0: 6, 1: 12, 2: 4, 3: 'entrance'}
# Event_3= {0: 6, 1: 12, 2: 'entrance', 3: 'entrance'}
# Event_2= {0: 6, 1: 5, 2: 'entrance', 3: 'entrance'}
# Event_1= {0: 6, 1: 'entrance', 2: 'entrance', 3: 'entrance'}

# Event_16= {0: 'Exit', 1: 'Exit', 2: 'Exit', 3: 'Exit'}
# Event_15= {0: 'Exit', 1: 'Exit', 2: 'Exit', 3: 25}
# Event_14= {0: 'Exit', 1: 'Exit', 2: 'Exit', 3: 18}
# Event_13= {0: 'Exit', 1: 'Exit', 2: 22, 3: 18}
# Event_12= {0: 'Exit', 1: 'Exit', 2: 22, 3: 6}
# Event_11= {0: 'Exit', 1: 'Exit', 2: 17, 3: 6}
# Event_10= {0: 'Exit', 1: 'Exit', 2: 17, 3: 'entrance'}
# Event_9= {0: 'Exit', 1: 'Exit', 2: 6, 3: 'entrance'}
# Event_8= {0: 'Exit', 1: 'Exit', 2: 'entrance', 3: 'entrance'}
# Event_7= {0: 'Exit', 1: 22, 2: 'entrance', 3: 'entrance'}
# Event_6= {0: 'Exit', 1: 17, 2: 'entrance', 3: 'entrance'}
# Event_5= {0: 'Exit', 1: 6, 2: 'entrance', 3: 'entrance'}
# Event_4= {0: 22, 1: 6, 2: 'entrance', 3: 'entrance'}
# Event_3= {0: 22, 1: 'entrance', 2: 'entrance', 3: 'entrance'}
# Event_2= {0: 17, 1: 'entrance', 2: 'entrance', 3: 'entrance'}
# Event_1= {0: 6, 1: 'entrance', 2: 'entrance', 3: 'entrance'}
#######################################################################################################
def set_up_network(params):
    # Extract key value pairs
    if "Spacing" in params:
        Spacing = params["Spacing"]
    else:
        Spacing= np.array(([[ 0.        ,-0.022719, -0.024844, -0.032665]]))
    if "grid_dim" in params:
        grid_dim = params['grid_dim']
    else:
        grid_dim = 3
    if "var_strng" in params:
        var_strng = params["var_strng"]
    else:
        var_strng=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    if "sim_drops" in params:
        sim_drops = params["sim_drops"]
    else:
        sim_drops = 4
    if 'source_nodes' in params:
        source_nodes = params['source_nodes']
    else:
        source_nodes = [1]
    if 'sink_nodes' in params:
        sink_nodes = params['sink_nodes']
    else:
        sink_nodes = [10,12]
    if 'arrange' in params:
        arrange = params['arrange']
    else:
        arrange = np.array(([[0,0,1,0]]))
    if 'rr' in params:
        rr = params['rr']
    else:
        arrange = 1
    rn1=18#random.randint(1,100)
    rn2=27#random.randint(1,100)
    rn3=46#random.randint(1,100)
    
    network = Network(grid_dim =grid_dim,var_strng=var_strng,sim_drops=sim_drops,
                        Droplet_frqncy=0.05,source_nodes=source_nodes,sink_nodes=sink_nodes,length_factor=1000 * (10 ** -6),Q_in=1* 10**(-6)/3600, P_out=1.01325 *10** 5 ,width_channel=200 * (10 ** -6),height_channel=200 * (10 ** -6),interfacial_tension=45*0.001,rr=rr,arrange=arrange,
                        alpha=0.9,beta=1.4,vscty=50 * (10 ** -3),value_spacing=Spacing,toterminate=False,rn1=rn1,rn2=rn2,rn3=rn3)

    network.node_branch_info()
    network.node_connector_matrix(network.var_strng)
    network.calc_length_branches()
    network.remove_branches()
    network.calc_length_branches()
    network.incidence_matrix_gen()
    network.branch_connectivity_fn()
    length_branches=network.calc_length_branches()


    Q_inlet=network.Q_in
    vel_inlet=network.beta*(network.Q_in/(network.width_channel*network.height_channel))



#######################################################################################################
def drop_branch_To_branch_num_drops(drop_branch):
    branch_num_drps={}
    for jdx in range(network.num_branches):
        branch_num_drps.update({network.branches[jdx]: np.zeros(2, dtype=int)})
    for qdx in range(len(drop_branch)):
        if drop_branch[qdx]!='entrance' and drop_branch[qdx]!='Exit':
            
            if network.arrange[0][qdx]==0:
                branch_num_drps.update({drop_branch[qdx] : branch_num_drps[drop_branch[qdx]]+np.array([1,0])})
            elif network.arrange[0][qdx]==1:
                branch_num_drps.update({drop_branch[qdx] : branch_num_drps[drop_branch[qdx]]+np.array([0,1])})
    return branch_num_drps
##############################################################################################
def velocity_calculation(branch_num_drps):
    network.resistance_calc_func(branch_num_drps)
    network.network_solution_calc_func()
    network.velocity_calc_func()
    k= copy.deepcopy(network.Branch_velocity)
    return k

def Moving_node_to(branch):
    if network.Branch_velocity[branch]>0:
        approaching_node=network.branch_cnctvty_nodes_cnctd[branch][1]
    else:
        approaching_node=network.branch_cnctvty_nodes_cnctd[branch][0]
    return approaching_node

def Decision_making_drop(i):
    drop_branch=globals()["Event_"+str(i)]
    drop_branch_prev=globals()["Event_"+str(i-1)]
    for j in range(network.sim_drops):
        if drop_branch[j]!=drop_branch_prev[j]:
            decision_making_drop=j
    return decision_making_drop

def Time_entry_event(X,i):
    drop_position={}
    decision_making_drop=Decision_making_drop(i)
    drop_branch=globals()["Event_"+str(i)]
    for j in range(0,network.sim_drops):
        if drop_branch[j]!='Exit' and drop_branch[j]!='entrance' and j!=decision_making_drop:
            drop_position.update({j:X[j]})
        else:
            drop_position.update({j:X[j]})
    return drop_position
    


def Time_next_event(drop_position,i,t_current,vel_dict):
    drop_branch=globals()["Event_"+str(i)]
    decision_making_drop=Decision_making_drop(i+1)
    vel=vel_dict[i]#velocity_calculation(drop_branch_To_branch_num_drops(drop_branch))
    
    # drop_branch_prev=globals()["Event_"+str(i-1)]
    if drop_branch[decision_making_drop]!='entrance':
        if vel_dict[i][drop_branch[decision_making_drop]]>0:
            time=t_current+((length_branches[drop_branch[decision_making_drop]]-drop_position[decision_making_drop])/(network.beta*vel[drop_branch[decision_making_drop]]))
        else:
            time=t_current+((drop_position[decision_making_drop])/(network.beta*vel[drop_branch[decision_making_drop]]))

    else:
        time=symbols("t_"+str(i+1))
        
    return time

def drop_position_update(drop_position,i,t_current):
    drop_branch=globals()["Event_"+str(i)]
    drop_branch_next=globals()["Event_"+str(i+1)]
    decision_making_drop=Decision_making_drop(i+1)
    vel=velocity_calculation(drop_branch_To_branch_num_drops(drop_branch))
        
    for k in range(0,network.sim_drops):
        if k!=decision_making_drop and drop_branch[k]!='Exit' and drop_branch[k]!='entrance':
            drop_position.update({k:drop_position[k]+((t_current[-1]-t_current[-2])*network.beta*vel[drop_branch[k]])})

        elif k==decision_making_drop and drop_branch[k]!='Exit' and drop_branch[k]!='entrance':
            
            if drop_branch_next[k]!='Exit':
                if vel[drop_branch[k]]>0 :
                    drop_position.update({k:0})
                elif vel[drop_branch[k]]<0:
                    drop_position.update({k:length_branches[drop_branch_next[k]]})
            else:
                drop_position.update({k:0})
                
        elif k==decision_making_drop and drop_branch[k]=='entrance':
            if vel[drop_branch_next[k]]>0 :
                drop_position.update({k:0})
            # NOTE: If drop branch is entrance, check velocity of branch where it goes next
            elif vel[drop_branch_next[k]]<0:
                drop_position.update({k:length_branches[drop_branch_next[k]]})         
                # print('hi')
        elif k!=decision_making_drop and (drop_branch[k]=='Exit' or drop_branch[k]!='entrance'):
            drop_position.update({k:0})
            
    return drop_position


def pick_fastest_branch(droplet_positions, dm_drop_position, target_branches):
    target_nodes = list(target_branches.keys())
    vel_dict = velocity_calculation(drop_branch_To_branch_num_drops(droplet_positions))

    cur_branch_vel = vel_dict[dm_drop_position]
    if (cur_branch_vel < 0):
        chosen_node = min(target_nodes)
        chosen_node_branches = [branch for branch in target_branches[chosen_node] if branch != dm_drop_position]
        local_branch_vels = {branch: vel_dict[branch] for branch in vel_dict if branch in chosen_node_branches}
        chosen_branch = min(local_branch_vels, key=local_branch_vels.get)
    else:
        chosen_node = max(target_nodes)
        chosen_node_branches = [branch for branch in target_branches[chosen_node] if branch != dm_drop_position]
        local_branch_vels = {branch: vel_dict[branch] for branch in vel_dict if branch in chosen_node_branches}
        chosen_branch = max(local_branch_vels, key=local_branch_vels.get)
    return chosen_branch

def get_velocities(droplet_positions):
    return velocity_calculation(drop_branch_To_branch_num_drops(droplet_positions))

def check_feasibility(event_sequence):
    event_len = len(event_sequence)
    for idx in range(event_len):
        globals()["Event_"+str(idx+1)] = event_sequence[idx]
    
    globals()["Event_0"]={0:'entrance',1:'entrance',2:'entrance',3:'entrance'}
    #Event_1={0:6,1:'entrance',2:'entrance',3:'entrance'}
    #Event_2={0:17,1:'entrance',2:'entrance',3:'entrance'}
    #Event_3={0:22,1:'entrance',2:'entrance',3:'entrance'}
    #Event_4={0:'Exit',1:'entrance',2:'entrance',3:'entrance'}
    #Event_5={0:'Exit',1:6,2:'entrance',3:'entrance'}
    #Event_6={0:'Exit',1:18,2:'entrance',3:'entrance'}
    #Event_7={0:22,1:18,2:12,3:'entrance'}
    #Event_8={0:'Exit',1:18,2:12,3:'entrance'}
    #Event_9={0:'Exit',1:25,2:12,3:'entrance'}
    #Event_10={0:'Exit',1:25,2:21,3:'entrance'}
    #Event_11={0:'Exit',1:25,2:21,3:6}
    #Event_12={0:'Exit',1:'Exit',2:21,3:6}
    #Event_13={0:'Exit',1:'Exit',2:'Exit',3:6}
    #Event_14={0:'Exit',1:'Exit',2:'Exit',3:17}
    #Event_15={0:'Exit',1:'Exit',2:'Exit',3:22}
    #Event_16={0:'Exit',1:'Exit',2:'Exit',3:'Exit'}
    drop_position={0: 0, 1: 0, 2: 0, 3: 0}
    equations=[]
    entry_times=[]
    t_current=[0]
    vel_dict={}

    for j in range(1,event_len):
        globals()["t_"+str(1)]=0
        decision_making_drop=Decision_making_drop(j+1)
        # print(decision_making_drop)
        drop_branch=globals()["Event_"+str(j+1)]
        drop_branch_prev=globals()["Event_"+str(j)]
        
        vel_dict.update({j:velocity_calculation(drop_branch_To_branch_num_drops(drop_branch_prev))})
        
        globals()["t_"+str(j+1)]=Time_next_event(drop_position,j,t_current[-1],vel_dict)
        t_current.append(globals()["t_"+str(j+1)])
        drop_position_update(drop_position,j,t_current)
        
        if drop_branch_prev[decision_making_drop]=='entrance':
            entry_times.append(globals()["t_"+str(j+1)])
            equations.append(globals()["t_"+str(j)]-globals()["t_"+str(j+1)])
        for k in range(network.sim_drops):
            if drop_branch[k]!='entrance' and drop_branch[k]!='Exit':
                if drop_position[k]!=0 and drop_position[k]!=length_branches[drop_branch[k]]:
                    if vel_dict[j][drop_branch[k]]>0:
                        equations.append(drop_position[k]-length_branches[drop_branch[k]])
                    else:
                        equations.append((length_branches[drop_branch[k]]-drop_position[k])-length_branches[drop_branch[k]])
            


    A, b = linear_eq_to_matrix(equations, entry_times)


    A=np.array(A)
    b=np.array(b)

    # If no constraints
    if (b.size <= 0):
        return True, 0
        
    aa=(np.where(b == np.amax(b)))
    dd=(np.where(b == np.amin(b)))
    # print(A)
    if np.amax(b)>0:
        #print('max(b)>0')
        
    # #print('aa',aa[0][0])
        cc=np.amax(abs(b/b[aa[0][0]]))
        # print('cc',cc)
        
        b_mult=-np.identity(b.shape[0])

        for hh in range(b.shape[0]):
            b_mult[hh,aa[0][0]]=1
        #print('b_mult',b_mult)
        
    else:
        #print('max(b)<0')
        b_mult=np.identity(b.shape[0])
        for hh in range(b.shape[0]):
            b_mult[hh,dd[0][0]]=-1
    A=np.concatenate((A,np.identity(A.shape[0])),axis=1)    


    A=np.matmul(b_mult,A)
    b=np.matmul(b_mult,b)  
    # #print(b)
    A_hat=np.concatenate((A,np.identity(A.shape[0])),axis=1)
    #print('A_hat',A_hat)
    c=np.zeros((A_hat.shape[1]))
    c[A.shape[1]:A_hat.shape[1]]=1

    # c=np.ones((A.shape[1],1))
    # c[-A.shape[0]:-1]=0
    #print('c',c)
    res=optimize.linprog(c, A_ub=None, b_ub=None, A_eq=A_hat, b_eq=b, bounds=None, method='simplex', callback=None, options=None, x0=None)

    return (res.fun < 1e-6), res.fun