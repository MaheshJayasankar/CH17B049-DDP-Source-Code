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

import copy

import numpy as np
import itertools
from numpy.lib.arraysetops import isin
from scipy import optimize
from scipy.optimize.optimize import OptimizeWarning
from sympy import *
from sympy import solve, symbols
import warnings


from .NetworkSimulation_4 import Network

#######################################################################################################

class FeasCheck:
    def __init__(self, network: Network, drops: "list[list[int]]"):
        self.network = network
        self.length_branches = network.calc_length_branches()
        self.drops = drops
        self.flat_range = list(itertools.chain(*drops))
        self.sim_drops = len(self.flat_range)
        self.drop_type_count = self.network.droplet_type_count
        self.entry_event_idx = []

    #######################################################################################################
    def drop_branch_To_branch_num_drops(self, drop_branch):
        network = self.network
        arrange = self.flat_range
        drop_type_count = self.drop_type_count

        branch_num_drps={}
        for jdx in range(network.num_branches):
            branch_num_drps.update({network.branches[jdx]: np.zeros(drop_type_count, dtype=int)})
        for qdx in range(len(drop_branch)):
            if drop_branch[qdx]!='entrance' and drop_branch[qdx]!='Exit':
                branch_num_drps[drop_branch[qdx]][arrange[qdx]] += 1
        return branch_num_drps
    def velocity_calculation(self, branch_num_drps):
        network = self.network
        network.resistance_calc_func(branch_num_drps)
        network.network_solution_calc_func()
        network.velocity_calc_func()
        k= copy.deepcopy(network.Branch_velocity)
        return k

    def Moving_node_to(self, branch):
        network = self.network
        if network.Branch_velocity[branch]>0:
            approaching_node=network.branch_cnctvty_nodes_cnctd[branch][1]
        else:
            approaching_node=network.branch_cnctvty_nodes_cnctd[branch][0]
        return approaching_node

    def Decision_making_drop(self, i):
        network = self.network
        sim_drops = self.sim_drops

        drop_branch=globals()["Event_"+str(i)]
        drop_branch_prev=globals()["Event_"+str(i-1)]
        for j in range(sim_drops):
            if drop_branch[j]!=drop_branch_prev[j]:
                decision_making_drop=j
        return decision_making_drop

    def Time_entry_event(self, X,i):
        sim_drops = self.sim_drops
        network = self.network
        drop_position={}
        decision_making_drop = self.Decision_making_drop(i)
        drop_branch=globals()["Event_"+str(i)]
        for j in range(0,sim_drops):
            if drop_branch[j]!='Exit' and drop_branch[j]!='entrance' and j!=decision_making_drop:
                drop_position.update({j:X[j]})
            else:
                drop_position.update({j:X[j]})
        return drop_position
    
    def Time_next_event(self, drop_position,i,t_current,vel_dict):
        network = self.network
        length_branches = self.length_branches

        drop_branch=globals()["Event_"+str(i)]
        decision_making_drop=self.Decision_making_drop(i+1)
        vel=vel_dict[i]
        
        # drop_branch_prev=globals()["Event_"+str(i-1)]
        if drop_branch[decision_making_drop]!='entrance':
            if vel_dict[i][drop_branch[decision_making_drop]]>0:
                time=t_current+((length_branches[drop_branch[decision_making_drop]]-drop_position[decision_making_drop])/(network.beta*vel[drop_branch[decision_making_drop]]))
            else:
                time=t_current+((drop_position[decision_making_drop])/(network.beta*vel[drop_branch[decision_making_drop]]))

        else:
            time=symbols("t_"+str(i+1))
            
        return time

    def drop_position_update(self, drop_position,i,t_current):
        network = self.network
        length_branches = self.length_branches
        sim_drops = self.sim_drops

        drop_branch=globals()["Event_"+str(i)]
        drop_branch_next=globals()["Event_"+str(i+1)]
        decision_making_drop=self.Decision_making_drop(i+1)
        vel=self.get_velocities(drop_branch)
            
        for k in range(0,sim_drops):
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

    def pick_fastest_branch(self, droplet_positions, dm_drop_position, target_branches):
        target_nodes = list(target_branches.keys())
        vel_dict = self.get_velocities(droplet_positions)

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

    def get_velocities(self,droplet_positions):
        return self.velocity_calculation(self.drop_branch_To_branch_num_drops(droplet_positions))

    def check_feasibility(self, event_sequence, return_whole_res = False) -> tuple[bool, any]:
        equations, entry_times = self.GenerateConstraints(event_sequence)

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            res=optimize.linprog(c, A_ub=None, b_ub=None, A_eq=A_hat, b_eq=b, bounds=None, method='simplex', callback=None, options=None, x0=None)

        if return_whole_res:
            return (res.fun < 1e-5), res
        return (res.fun < 1e-5), res.fun

    def GetEquations(self, event_sequence):
        equations, entry_times = self.GenerateConstraints(event_sequence)
        A, b = linear_eq_to_matrix(equations, entry_times)
        return {
            'A': A,
            'b': b,
            'variables': self.entry_event_idx
        }

    def GenerateConstraints(self, event_sequence):
        network = self.network
        length_branches = self.length_branches
        sim_drops = self.sim_drops

        event_len = len(event_sequence)
        for idx in range(event_len):
            globals()["Event_"+str(idx+1)] = event_sequence[idx]
        
        globals()["Event_0"]={0:'entrance',1:'entrance',2:'entrance',3:'entrance'}

        drop_position={0: 0, 1: 0, 2: 0, 3: 0}
        equations=[]
        entry_times=[]
        t_current=[0]
        vel_dict={}

        for j in range(1,event_len):
            globals()["t_"+str(1)]=0
            decision_making_drop=self.Decision_making_drop(j+1)
            # print(decision_making_drop)
            drop_branch=globals()["Event_"+str(j+1)]
            drop_branch_prev=globals()["Event_"+str(j)]
            
            vel_dict.update({j:self.get_velocities(drop_branch_prev)})
            
            globals()["t_"+str(j+1)]=self.Time_next_event(drop_position,j,t_current[-1],vel_dict)
            t_current.append(globals()["t_"+str(j+1)])
            self.drop_position_update(drop_position,j,t_current)
            
            if drop_branch_prev[decision_making_drop]=='entrance':
                entry_times.append(globals()["t_"+str(j+1)])
                self.entry_event_idx.append(j - 1)
                equations.append(globals()["t_"+str(j)]-globals()["t_"+str(j+1)])
            for k in range(sim_drops):
                if drop_branch[k]!='entrance' and drop_branch[k]!='Exit':
                    if drop_position[k]!=0 and drop_position[k]!=length_branches[drop_branch[k]]:
                        if vel_dict[j][drop_branch[k]]>0:
                            equations.append(drop_position[k]-length_branches[drop_branch[k]])
                        else:
                            equations.append((length_branches[drop_branch[k]]-drop_position[k])-length_branches[drop_branch[k]])
        return equations,entry_times
