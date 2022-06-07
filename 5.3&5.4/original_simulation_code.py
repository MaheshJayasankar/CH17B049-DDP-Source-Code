# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:31:29 2020

@author: arune
"""
# Modified for use in Dual Degree Project (2021-2022) by J Mahesh, CH17B049.

# import glob
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import glob

# NOTE: If the numpy version installed is higher than 1.18, the algorithm may not work as intended

class Network:
    """
    Network Class that can run the simulation, calculate fitness, and generate video output.
    Parameters
    ==========
    Major Parameters
    ----------
    `grid_dim`: Dimension of the square grid.

    `var_strng`: List of 0-1 values indicating if a branch is present or not

    `sim_drops`: Total number of droplets 

    `source_nodes`: List source nodes

    `sink_nodes`: List of sink nodes

    `arrange`: The sequence of droplets as a list, for each sink, bundled as a list of lists. Value 0 <=> Type A, 1 <=> Type B.

    `value_spacing`: Spacing values of droplets, i.e. relative position from source node at t=0. Stored as a list, in negative float values.

    Significant Parameters
    ----------
    `length_factor`: Length of a horizontal/vertical branch

    `Q_in`: Fluid inlet flow rate

    `P_out`: Pressure at the sink nodes (all share same value)

    `width_channel`: Width of each channel

    `height_channel`: Height of each channel

    `rr_list`: Resistance ratio between all droplet types, normalised to type A.

    `beta`: Slip factor, ratio of droplet velocity to fluid velocity. Keep above 1.

    `vscty`: Viscosity in SI units.

    Other Parameters
    ----------
    `toterminate`: Used to prevent trapping issue.

    `rn1`,`rn2`,`rn3`: Random seed values to add noise to every branch length, to prevent ties in droplet events.

    Display Only Parameters
    ----------
    `subdue_print`: Don't output any print statements that were used for debugging.

    `output_img_ext`: The extension with which to save image files. Incase .jpg doesn't work, use .png.

    `should_gen_images`: Whether or not to save images created during the simulation. This is necessary to generate video later.

    Unused Parameters
    ----------

    `alpha`: Currently unused.

    `rr`: Resistance ratio between droplet type B with type A... replaced with rr_list

    `Droplet_frqncy`: A parameter that was used to determine spacing, but now we are manually inputting spacing through value_spacing.

    `interfacial_tension`: We are ignoring the effects of interfacial tension.
    
    External Functions
    ==========
    `simulate_network_func`: Run the simulation

    `network_fitness_calc_fn`: Find the fitness value after simulation has finished

    `gen_video`: Generate video from image output.

    """

    def __init__(self,
                 grid_dim, var_strng, sim_drops, source_nodes, sink_nodes,arrange,value_spacing,
                 length_factor, Q_in, P_out, width_channel, height_channel,rr_list,beta,vscty,
                 toterminate,rn1,rn2,rn3,
                 rr = 1,alpha = 0.4,subdue_print = True, output_img_ext = '.jpg', should_gen_images = False,
                 Droplet_frqncy = 0.05, interfacial_tension = 0.045):
        
        # 450micro litre/hr is the flow rate
        
        self.grid_dim = grid_dim
        self.var_strng= var_strng
        self.Droplet_frqncy = Droplet_frqncy
        self.sim_drops = sim_drops
        self.source_nodes = source_nodes
        self.sink_nodes = sink_nodes
        self.arrange=np.array(arrange)
        self.length_factor = length_factor
        self.Q_in = Q_in
        self.P_out = P_out
        self.width_channel = width_channel
        self.height_channel = height_channel
        self.vscty = vscty
        self.interfacial_tension=interfacial_tension
        srs_prt=0
        for n_srsindx in range(1, 100, 2):
            srs_prt = srs_prt + (1 / n_srsindx ** 5) * (192 / np.pi ** 5) * (self.height_channel) / (self.width_channel) * np.tanh((n_srsindx * np.pi * width_channel) / (2 * height_channel))
        self.srs_prt=srs_prt
        self.R_d1 = (12 * self.vscty * self.length_factor / ((self.width_channel) * self.height_channel ** 3)) * (1 - self.srs_prt) ** -1# 3.15*1.5*interfacial_tension*((self.vscty* self.Q_in/(self.width_channel* self.height_channel*interfacial_tension))**(2/3))/(self.Q_in*self.width_channel)
        self.rr = rr
        self.rr_list = rr_list

        self.droplet_type_count = len(rr_list)
        if (self.arrange.max() > self.droplet_type_count - 1):
            raise ValueError("Not enough Resistance Ratio values provided for all droplet types.")
        
        self.R_d_list = [self.R_d1]
        for i in range(1,self.droplet_type_count):
            self.R_d_list.append(self.R_d1 * rr_list[i])

        self.alpha= alpha
        self.beta= beta
        self.value_spacing=value_spacing
        self.toterminate = toterminate
        self.rn1=rn1
        self.rn2=rn2
        self.rn3=rn3
        self.subdue_print = subdue_print
        self.output_img_ext = output_img_ext
        self.should_gen_images = should_gen_images
        if (not(self.subdue_print)):
            print('srs_prt',self.srs_prt)
            print('drop_resistance',self.R_d1)
            print('rn1, rn2,rn3',self.rn1,self.rn2,self.rn3)

        self.event_sequence = [{0:'entrance',1:'entrance',2:'entrance',3:'entrance'}]
        self.raw_event_sequence = []
        self.ex0 = len(var_strng)
        self.branch_lengths_found = False
        self.A = None
        self.branch_node_list = None
        self.node_branch_info()
        self.sign_convention: dict[int, int] = None
        self.EstablishSignConvention()

    def node_branch_info(self):
        """
        The function returns a matrix(A) of dim (total_num_branches,2)
        A(i,0) & A(i,1) give the two nodes between which the branch exists
        """

        grid_dim = self.grid_dim

        total_branches = (grid_dim - 1) * (6 * grid_dim - 4)  # Total number of branches for a given grid_dim
        intersection_node_matrix = np.zeros((grid_dim - 1,grid_dim - 1), dtype=int)
        for idx in range(grid_dim - 1):
            for jdx in range(grid_dim - 1):
                intersection_node_matrix[idx][jdx] = idx + grid_dim + jdx*(2*grid_dim - 1)
        A = np.zeros((total_branches, 2), dtype=int)
        jdx = 0  # branch number
        idx = 0  # node number
        while jdx <= total_branches - 1:
            if (idx >= ((2*grid_dim - 1)*(grid_dim - 1))) and (idx < (2*(grid_dim ** 2) - 2*grid_dim)):
                # Last column nodes
                A[jdx][0] = idx
                A[jdx][1] = idx + 1
                jdx = jdx + 1
            elif (idx >= grid_dim) and (idx in intersection_node_matrix.flatten()) and (idx <= (2*(grid_dim**2) - 3*grid_dim)):
                # intersection nodes
                A[jdx][0] = idx
                A[jdx][1] = idx + (grid_dim - 1)
                A[jdx + 1][0] = idx
                A[jdx + 1][1] = idx + grid_dim
                jdx = jdx + 2
            elif (idx == 0) or (idx % (2*grid_dim - 1) == 0):
                # Top nodes
                A[jdx][0] = idx
                A[jdx][1] = idx + 1
                A[jdx + 1][0] = idx
                A[jdx + 1][1] = idx + grid_dim
                A[jdx + 2][0] = idx
                A[jdx + 2][1] = idx + (2*grid_dim - 1)
                jdx = jdx + 3
            elif (idx + 1 - grid_dim) % (2*grid_dim - 1) == 0:
                # bottom nodes
                A[jdx][0] = idx
                A[jdx][1] = idx + grid_dim - 1
                A[jdx + 1][0] = idx
                A[jdx + 1][1] = idx + (2*grid_dim - 1)
                jdx = jdx + 2
            else:
                # Middle nodes
                A[jdx][0] = idx
                A[jdx][1] = idx + 1
                A[jdx + 1][0] = idx
                A[jdx + 1][1] = idx + grid_dim - 1
                A[jdx + 2][0] = idx
                A[jdx + 2][1] = idx + grid_dim
                A[jdx + 3][0] = idx
                A[jdx + 3][1] = idx + (2*grid_dim - 1)
                jdx = jdx + 4

            idx = idx + 1
            # An array which gives the nodes a branch runs through
        self.intersection_node_matrix = intersection_node_matrix
        self.A = A
        self.branch_node_list = A   
        return A

    def node_connector_matrix(self, var_strng):
        """
        adjacency matrix
        returns a matrix A of dimensions (num_nodes,num_nodes)
        A(i,j) = 1 if there exist a branch between nodes i & j

        """
        grid_dim = self.grid_dim
        #print(var_strng)
        num_nodes_total = grid_dim ** 2 + (grid_dim - 1)**2
        num_branchs_total = (grid_dim - 1) * (6 * grid_dim - 4)
        A = np.zeros((num_nodes_total, num_nodes_total), dtype=int)
        idx = 0  # node
        jdx = 0  # branch
        while jdx <= num_branchs_total - 1:
            if idx >= ((2*grid_dim - 1)*(grid_dim - 1)) and idx < (2*(grid_dim ** 2) - 2*grid_dim):
                # Last column nodes
                A[idx][idx + 1] = self.var_strng[jdx]
                A[idx + 1][idx] = self.var_strng[jdx]
                jdx = jdx + 1
            elif idx >= grid_dim and idx in self.intersection_node_matrix.flatten() and idx <= (2*(grid_dim**2) - 3*grid_dim):
                # intersection nodes
                A[idx][idx + (self.grid_dim - 1)] = self.var_strng[jdx]
                A[idx + (self.grid_dim - 1)][idx] = self.var_strng[jdx]
                A[idx][idx + self.grid_dim] = self.var_strng[jdx + 1]
                A[idx + self.grid_dim][idx] = self.var_strng[jdx + 1]
                jdx = jdx + 2
            elif idx == 0 or idx % (2*grid_dim - 1) == 0:
                # Top nodes
                A[idx][idx + 1] = self.var_strng[jdx]
                A[idx + 1][idx] = self.var_strng[jdx]
                A[idx][idx + self.grid_dim] = self.var_strng[jdx + 1]
                A[idx + self.grid_dim][idx] = self.var_strng[jdx + 1]
                A[idx][idx + (2*self.grid_dim - 1)] = self.var_strng[jdx + 2]
                A[idx + (2*self.grid_dim - 1)][idx] = self.var_strng[jdx + 2]
                jdx = jdx + 3
            elif (idx + 1 - grid_dim) % (2*grid_dim - 1) == 0:
                # bottom nodes
                A[idx][idx + self.grid_dim - 1] = self.var_strng[jdx]
                A[idx + self.grid_dim - 1][idx] = self.var_strng[jdx]
                A[idx][idx + (2*self.grid_dim - 1)] = self.var_strng[jdx + 1]
                A[idx + (2*self.grid_dim - 1)][idx] = self.var_strng[jdx + 1]
                jdx = jdx + 2
            else:
                # Middle nodes
                A[idx][idx + 1] = self.var_strng[jdx]
                A[idx + 1][idx] = self.var_strng[jdx]
                A[idx][idx + self.grid_dim - 1] = self.var_strng[jdx + 1]
                A[idx + self.grid_dim - 1][idx] = self.var_strng[jdx + 1]
                A[idx][idx + grid_dim] = self.var_strng[jdx + 2]
                A[idx + grid_dim][idx] = self.var_strng[jdx + 2]
                A[idx][idx + (2*self.grid_dim - 1)] = self.var_strng[jdx + 3]
                A[idx + (2*self.grid_dim - 1)][idx] = self.var_strng[jdx + 3]
                jdx = jdx + 4

            idx = idx + 1
        #print(A)
        #y = A.tolist()
        #print(y)
        #A[A==2] = 1
        #print(A)
        return A

    def calc_length_branches(self):
        """
        Internally called by network_simulation_func to decide the length of the branch.
        Here the variables rn1 rn2 rn3 are used to introduce perturbation in the lengths of each branch.
        """
        orig_randstate = np.random.get_state()
        for_struct = []
        # P=[]
        # Q=[]
        node_cordinate={}
        if self.should_gen_images:
            plt.xlim(-2 * self.length_factor, (self.grid_dim + 1) * self.length_factor)
            plt.ylim(-2 * self.length_factor, (self.grid_dim + 1) * self.length_factor)
        # L=(1/1000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000)), int(self.length_factor*(1000)))
        #print(L)
        np.random.seed(self.rn1)
        #imp_node1 = np.multiply(L,(1.01-0.99)*np.random.random_sample(3)+0.99)#np.multiply(L, np.array(([0.99999972, 0.99999775, 0.99999615]))) #0.99999546, 0.99999864])))
        #G=np.multiply(((1/1000000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000000)), int(self.length_factor*(1000000)))),imp_node1)
        G=np.multiply(((1/1000000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000000)), int(self.length_factor*(1000000)))),(1.0005-0.9995)*np.random.random_sample(self.grid_dim)+0.9995)#,0.99999546, 0.99999864])));
        #print(G)
        np.random.seed(self.rn2)
        M=np.multiply(((1/1000000)*np.arange((self.grid_dim - 1)* int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)))),(1.0005-0.9995)*np.random.random_sample(self.grid_dim)+0.9995)#, 1.00000546, 1.00000348])))
        #M=np.multiply(((1/1000000)*np.arange((self.grid_dim - 1)* int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)))),np.array(([1.00000067, 0.99999825, 1.00000435])))#, 1.00000546, 1.00000348])))
        #print(M)
        for i in np.arange(0,len(G)):
            for k in np.arange(0,len(M)):
                for_struct.append([G[i], M[k]])
                #P[i][k]=G[i]
                #Q[i][k]=M[k]
                
            if i!=len(G)-1:
                d=i#for d in np.arange(0,len(G)-1,1) :
                #if G[d]==i:
                np.random.seed(self.rn3)
                RN=(1.0005-0.9995)*np.random.random_sample((self.grid_dim-1)**2)+0.9995
                for j in np.arange(0,len(M)-1,1):
                    #print("j",j)
                    A=np.array(( [[ (M[j+1]-M[j]) / (G[d+1]-G[d]), -1], [ (M[j]-M[j+1]) / (G[d+1]-G[d]) , -1]]))
                    b=np.array(([[G[d]*(( M[j+1] - M[j] )/(G[d+1]-G[d]))-M[j]],[G[d]*(( M[j] - M[j+1] )/(G[d+1]-G[d]))-M[j+1]]]))
                    X=(np.linalg.inv(A)).dot(b)
                    #print(X[0])
                    for_struct.append([np.float(X[0])*RN[(self.grid_dim-1)*d+j], np.float(X[1])*RN[-((self.grid_dim-1)*d+j)]])
                    #P[d][k]=G[i]
                    #Q[i][k]=M[k]
        for key in np.arange(0,self.grid_dim **2+(self.grid_dim-1)**2):
            node_cordinate.update({key:for_struct[key]})

        length_branches=np.zeros(len(self.A))
           
        for j in np.arange(0,len(self.A)):

            length_branches[j]=(((node_cordinate[self.A[j][0]][0]-node_cordinate[self.A[j][1]][0])**2)+((node_cordinate[self.A[j][0]][1]-node_cordinate[self.A[j][1]][1])**2))** 0.5
        #print('Logkjgdh',length_branches)
        #        A = self.node_branch_info()
        #        grid_dim = self.grid_dim
        #        
        #        def myfunc(branch):
        #            node1 = branch[0]
        #            node2 = branch[1]
        #            if (node2 - node1 == 1):
        #                return self.length_factor
        #            elif (node2 - node1 == grid_dim):
        #                return self.length_factor / (2**0.5)
        #            elif (node2 - node1 == grid_dim - 1):
        #                return self.length_factor / (2 ** 0.5)
        #            elif (node2 - node1 == (2*grid_dim - 1)):
        #                return self.length_factor

        #        length_branches = np.reshape(np.apply_along_axis(myfunc, axis=1, arr=A), (A.shape[0], -1))
        #        print("length branch",length_branches)
        #        print("length_branches",length_branches)
        np.random.set_state(orig_randstate)
        return length_branches

    def remove_branches(self):

        """
        Called internally during network construction.
        """

        A = self.node_branch_info()
        branches = np.flatnonzero(np.array(self.var_strng))
        bypass_branches = np.where(np.array(self.var_strng) == 2)[0]
        num_branches = len(branches)
        nodes = np.unique(A[branches, :])
        num_nodes = len(nodes)
        node_indices = {}
        branch_indices = {}
        # length_branches = self.calc_length_branches()
        # print(length_branches[branches])
        for idx in range(num_nodes):
            node_indices.update({nodes[idx]: idx})
        for jdx in range(num_branches):
            branch_indices.update(({branches[jdx]: jdx}))

        self.branches = branches
        self.bypass_branches = bypass_branches
        self.nodes = nodes
        self.num_branches = num_branches
        self.num_nodes = num_nodes
        self.node_indices = node_indices
        self.branch_indices = branch_indices

    def incidence_matrix_gen(self):
        """
        Returns the node connector matrix of given network
        Node connector matrix is of dimesnions (num_nodes x num_nodes), where
        A(i,j) = 1 if there exists a branch between nodes i & j
        """

        grid_dim = self.grid_dim
        num_original_nodes = self.grid_dim ** 2 + (self.grid_dim - 1)**2
        total_branches = (grid_dim - 1) * (6 * grid_dim - 4)
        # A = np.zeros((total_branches, num_original_nodes), dtype=int)
        B = self.node_branch_info()
        self.remove_branches()
        # print(self.branches)
        #print(self.nodes)
        A = np.zeros((self.num_branches, self.num_nodes), dtype=int)
        for idx in range(self.num_branches):
            A[idx][self.node_indices[B[self.branches[idx]][0]]] = 1
            A[idx][self.node_indices[B[self.branches[idx]][1]]] = -1

        incidence_matrix = A
        #print(incidence_matrix.tolist())
        Q_in = self.Q_in
        P_out = self.P_out
        num_nodes = self.num_nodes
        #print('num nodes',num_nodes)
        sink_nodes = self.sink_nodes
        source_nodes = self.source_nodes
        dim_D = len(self.sink_nodes)
        dim_Pbc = len(self.sink_nodes)
        D_matrix = np.asarray(np.zeros((dim_D, num_nodes)))
        C_matrix = np.asarray(np.zeros((num_nodes - dim_Pbc, num_nodes)))
        Pressure_bc = P_out * np.ones((len(sink_nodes), 1))
        flow_bc = np.zeros((num_nodes - dim_Pbc, 1))
        jdx = 0
        kdx = 0

        for idx in range(num_nodes):
            if (self.nodes[idx] not in sink_nodes) and (self.nodes[idx] not in source_nodes):
                C_matrix[jdx][idx] = 1
                jdx = jdx + 1
            elif self.nodes[idx] in sink_nodes:
                D_matrix[kdx][idx] = 1
                kdx = kdx + 1
            elif self.nodes[idx] in source_nodes:
                C_matrix[jdx][idx] = 1
                flow_bc[jdx][0] = Q_in
                jdx = jdx + 1

        # print(flow_bc)
        self.incidence_matrix = incidence_matrix
        self.C_matrix = C_matrix
        self.D_matrix = D_matrix
        self.Pressure_bc = Pressure_bc
        self.flow_bc = flow_bc

        return
        
    def branch_connectivity_fn(self):
        """
        This function tells the nodes connecting any given branch and the branches
        connected to each of the nodes
        """
        if (not(self.subdue_print)):
            print("i see")
        num_branches = self.num_branches
        branches = self.branches
        num_nodes = self.num_nodes
        nodes = self.nodes
        incidence_matrix = self.incidence_matrix
        branch_cnctvty_nodes_cnctd = {}
        branch_cnctvty_branchs_cnctd = {}

        for idx in range(num_nodes):
            # Each node what branches are connected
            branch_cnctvty_branchs_cnctd.update({nodes[idx]: branches[np.flatnonzero(incidence_matrix[:, idx])]})

        for idx in range(num_branches):
            # Each branch what nodes are connected
            branch_cnctvty_nodes_cnctd.update({branches[idx]: nodes[np.flatnonzero(incidence_matrix[idx, :])]})

        self.branch_cnctvty_nodes_cnctd = branch_cnctvty_nodes_cnctd
        self.branch_cnctvty_branchs_cnctd = branch_cnctvty_branchs_cnctd

        return

    def resistance_calc_func(self, branch_num_drps = None):
        """
        Calculate the resistance in each branch, using positions of all droplets.
        """

        #print("resistance_calc_func-block runs")
        width_channel = self.width_channel
        w2 = self.width_channel/2
        height_channel = self.height_channel
        vscty = self.vscty
        # R_d1 = self.R_d1
        # R_d2 = R_d1 * self.rr
        cross_area = width_channel * height_channel
        srs_prt = 0
        srs_prt1 = 0
        length_branches = self.calc_length_branches()
        #print(length_branches)

        for n_srsindx in range(1, 100, 2):
            srs_prt = srs_prt + (1 / n_srsindx ** 5) * (192 / np.pi ** 5) * (height_channel) / (
                width_channel) * np.tanh((n_srsindx * np.pi * width_channel) / (2 * height_channel))
            srs_prt1 = srs_prt1 + (1 / n_srsindx ** 5) * (192 / np.pi ** 5) * (height_channel) / (w2) * np.tanh((n_srsindx * np.pi * w2) / (2 * height_channel))
        if not branch_num_drps:
            branch_num_drps = self.branch_num_drps
        resistances = np.zeros(self.num_branches)

        for idx in range(self.num_branches):
            resistances_only_branch = 12 * vscty * length_branches[self.branches[idx]] / ((width_channel) * height_channel ** 3) * (1 - srs_prt) ** -1
            if self.branches[idx] in self.bypass_branches:
                resistances[idx] = 12 * vscty * length_branches[self.branches[idx]] / ((w2 * height_channel ** 3) * (1 - srs_prt1) ** -1)
            else:
                branch_resistances_sum = 0
                resistances[idx] = 12 * vscty * length_branches[self.branches[idx]] / ((width_channel) * height_channel ** 3) * (1 - srs_prt) ** -1 
                # + branch_num_drps[self.branches[idx]][0] * R_d1 + branch_num_drps[self.branches[idx]][1] * R_d2
                for jdx in range(self.droplet_type_count):
                    branch_resistances_sum += branch_num_drps[self.branches[idx]][jdx] * self.R_d_list[jdx]
                # Added support for multiple droplet types
                resistances[idx] += branch_resistances_sum

        self.resistances = resistances
        self.resistances_only_branch = resistances_only_branch
        #print("branch resistance",self.resistances)

        return

    def network_solution_calc_func(self):

        """
        Solve the network equation and determine all flow rates as per VLDMI paper.
        """

        resistance_normalized = np.diag(self.resistances) / max(self.resistances)
        Pressure_bc_normalized = self.Pressure_bc / max(self.resistances)
        trans_incidence_matrix = np.transpose(self.incidence_matrix)
        row_C = self.C_matrix.shape[0]
        col_C = self.C_matrix.shape[1]
        row_D = self.D_matrix.shape[0]
        col_D = self.D_matrix.shape[1]
        num_branches = self.num_branches
        num_nodes = self.num_nodes
        nodes = self.nodes
        branches = self.branches
        Final_matrix = np.zeros((2 * num_nodes + num_branches, 2 * num_nodes + num_branches))

        for idx in range(2 * num_nodes + num_branches):
            if idx < num_branches:
                Final_matrix[idx][:num_nodes] = self.incidence_matrix[idx, :]
                Final_matrix[idx][num_nodes:num_nodes + num_branches] = -resistance_normalized[idx, :]

            elif idx >= num_branches and idx < (num_branches + num_nodes):
                Final_matrix[idx][num_nodes:num_nodes + num_branches] = trans_incidence_matrix[idx - num_branches, :]
                Final_matrix[idx][num_nodes + num_branches:] = -np.eye(num_nodes)[idx - num_branches, :]

            elif idx >= num_nodes + num_branches and idx < num_nodes + num_branches + row_C:
                Final_matrix[idx][2 * num_nodes + num_branches - col_C:] = self.C_matrix[idx - num_nodes - num_branches,
                                                                           :]
            elif idx >= num_nodes + num_branches + row_C:

                Final_matrix[idx][0:col_D] = self.D_matrix[idx - num_nodes - num_branches - row_C, :]

        Final_matrix = np.asmatrix(Final_matrix)
        Right_matrix = np.zeros((2 * num_nodes + num_branches, 1))
        Right_matrix[num_branches + num_nodes:num_branches + num_nodes + len(self.flow_bc)] = self.flow_bc
        Right_matrix[num_branches + num_nodes + len(self.flow_bc):, :] = Pressure_bc_normalized
        inv_Final_matrix = np.linalg.inv(np.asmatrix(Final_matrix))
        Network_solution = np.matmul(inv_Final_matrix, Right_matrix)

        Network_solution[:num_nodes, :] = Network_solution[:num_nodes, :] * max(self.resistances)
        self.Network_solution = Network_solution

        return

    def velocity_calc_func(self):
        """
        After solving for flow rates, find velocity of each drop
        """

        cross_area = self.width_channel * self.height_channel
        Branch_velocity = {}

        for idx in range(self.num_branches):
            Branch_velocity.update({self.branches[idx]: self.Network_solution[self.num_nodes + idx, 0] / cross_area})
        self.Branch_velocity = Branch_velocity

        #print(self.branch_cnctvty_nodes_cnctd)

        #print(Branch_velocity)
        return

    def drop_movement(self, node_or_branch, val):
        """
        Internally used to obtain possible droplet movements for tmin calculation
        """

        if val == 0:
            """New drop entering the network"""

            branch_options = self.branch_cnctvty_branchs_cnctd[node_or_branch]
            node_options = np.array([], dtype=int)
            #print(branch_options)
            for idx in range(len(branch_options)):
                tmp = np.setdiff1d(self.branch_cnctvty_nodes_cnctd[branch_options[idx]], np.array([node_or_branch]))[0]
                node_options = np.append(node_options, tmp)

            #print(node_options)
            del_nodes_indices = np.array([], dtype=int)

            for idx in range(len(node_options)):
                for jdx in range(len(self.bypass_branches)):
                    if branch_options[idx] == self.bypass_branches[jdx]:
                        del_nodes_indices = np.append(del_nodes_indices, idx)
                if self.Network_solution[self.node_indices[node_options[idx]]] - self.Network_solution[self.node_indices[node_or_branch]] > 0:
                    del_nodes_indices = np.append(del_nodes_indices, idx)
    
            node_options = np.delete(node_options, del_nodes_indices)
            branch_options = np.delete(branch_options, del_nodes_indices)
            del_branch_indices = np.array([], dtype=int)
            for idx in range(len(branch_options)):
                if self.Branch_velocity[branch_options[idx]] == 0:
                    del_branch_indices = np.append(del_branch_indices, idx)

            node_options = np.delete(node_options, del_branch_indices)
            branch_options = np.delete(branch_options, del_branch_indices)
            velocity_arr1 = np.array([])
            for idx in range(len(node_options)):
                velocity_arr1 = np.append(velocity_arr1, abs(self.Branch_velocity[branch_options[idx]]))

            #print(velocity_arr1)
            indx = np.argmax(velocity_arr1)
            #print(indx)
            moving_branch = branch_options[indx]
            node_travelling_to = node_options[indx]

            return moving_branch, [node_or_branch, node_travelling_to]

        elif val == 1:
            """Drop leaving the network"""
            moving_branch1 = node_or_branch
            node_travelling_from1 = np.setdiff1d(self.branch_cnctvty_nodes_cnctd[node_or_branch], self.sink_nodes)[0]
            node_travelling_to1 = np.intersect1d(self.branch_cnctvty_nodes_cnctd[node_or_branch], self.sink_nodes)[0]

            return moving_branch1, [node_travelling_from1, node_travelling_to1]



        elif val == 2:
            "Droplet changes branch within the network"
            if (not(self.subdue_print)):
                print(node_or_branch)
            #print(self.Network_solution)
            c_1 = np.argmin([self.Network_solution[self.node_indices[x]] for x in self.branch_cnctvty_nodes_cnctd[node_or_branch]])
            c_2 = np.argmax([self.Network_solution[self.node_indices[x]] for x in self.branch_cnctvty_nodes_cnctd[node_or_branch]])
            node_travelling_from = self.branch_cnctvty_nodes_cnctd[node_or_branch][c_1]
            node_travelling_from_prime = self.branch_cnctvty_nodes_cnctd[node_or_branch][c_2]
            #print('Dissection starts')
            branch_options = self.branch_cnctvty_branchs_cnctd[node_travelling_from]
            #print(branch_options)
            if self.Branch_velocity_prev[node_or_branch]*self.Branch_velocity[node_or_branch] > 0:
                for vdx in range(len(branch_options)):
                    if branch_options[vdx] == node_or_branch:
                        del_indx = vdx
            branch_options = np.delete(branch_options,del_indx)
            #print(del_indx)
            #print(node_travelling_from)
            #print(branch_options)
            node_options = np.array([], dtype=int)
            for idx in range(len(branch_options)):
                for jdx in range(2):
                    if self.branch_cnctvty_nodes_cnctd[branch_options[idx]][jdx] != node_travelling_from:
                        node_options = np.append(node_options,
                                                 self.branch_cnctvty_nodes_cnctd[branch_options[idx]][jdx])
            if (not(self.subdue_print)):
                print(node_options)
            # print(self.Network_solution[:7,:])
            # print(self.Network_solution[self.node_indices[node_options[0]]])
            # node_options = np.array([np.setdiff1d(self.branch_cnctvty_nodes_cnctd(x),self.branch_cnctvty_nodes_cnctd[node_travelling_from])[0] for x in branch_options])
            del_nodes_indices = np.array([], dtype=int)
            for idx in range(len(node_options)):
                for jdx in range(len(self.bypass_branches)):
                    if branch_options[idx] == self.bypass_branches[jdx]:
                        del_nodes_indices = np.append(del_nodes_indices, idx)
                if self.Network_solution[self.node_indices[node_options[idx]]] - self.Network_solution[self.node_indices[node_travelling_from]] >= 0:
                        #print('check now')
                        del_nodes_indices = np.append(del_nodes_indices, idx)

            node_options = np.delete(node_options, del_nodes_indices)
            branch_options = np.delete(branch_options, del_nodes_indices)
            #print(branch_options)

            del_branch_indices = np.array([], dtype=int)
            for idx in range(len(branch_options)):
                if self.Branch_velocity[branch_options[idx]] == 0:
                    del_branch_indices = np.append(del_branch_indices, idx)
            node_options = np.delete(node_options, del_branch_indices)
            branch_options = np.delete(branch_options, del_branch_indices)
            #print("branch options",branch_options)
            #print("node options",node_options)
            velocity_arr = np.array([])

            for idx in range(len(node_options)):
                velocity_arr = np.append(velocity_arr, self.Branch_velocity[branch_options[idx]])
            #print(velocity_arr)
            #print('yuck')
            #print(node_travelling_from_prime)
            #print(node_travelling_from)
            if len(velocity_arr) == 0:
                if (not(self.subdue_print)):
                    print("rude")
                if self.Network_solution[self.node_indices[node_travelling_from]] - self.Network_solution[self.node_indices[node_travelling_from_prime]] > 0:
                    return [node_or_branch], [node_travelling_from_prime]
                elif self.Network_solution[self.node_indices[node_travelling_from]] - self.Network_solution[self.node_indices[node_travelling_from_prime]] < 0:
                    return [node_or_branch], [node_travelling_from]
                elif self.Network_solution[self.node_indices[node_travelling_from]] - self.Network_solution[self.node_indices[node_travelling_from_prime]] == 0:
                    return [node_or_branch], [node_travelling_from]

            else:
                indx = np.argmax(abs(velocity_arr))
                #print("index",indx)
                moving_branch = branch_options[indx]
                node_travelling_to = node_options[indx]
                #print('current branch',node_or_branch)
                #print('moving-branch',moving_branch)
                #print('node to',node_travelling_to)

                return [node_or_branch, moving_branch], [node_travelling_from, node_travelling_to]

        elif val == 3:
            "Trapping event"
            branch_options = self.branch_cnctvty_branchs_cnctd[node_or_branch]
            node_options = np.array([], dtype=int)
            #print("branch options",branch_options)

            for idx in range(len(branch_options)):
                tmp = np.setdiff1d(self.branch_cnctvty_nodes_cnctd[branch_options[idx]], np.array([node_or_branch]))[0]
                node_options = np.append(node_options, tmp)
            #print(node_options)
            del_nodes_indices = np.array([], dtype=int)

            for idx in range(len(node_options)):
                for jdx in range(len(self.bypass_branches)):
                    if branch_options[idx] == self.bypass_branches[jdx]:
                        del_nodes_indices = np.append(del_nodes_indices, idx)
                if self.Network_solution[self.node_indices[node_options[idx]]] - self.Network_solution[self.node_indices[node_or_branch]] >= 0:
                    del_nodes_indices = np.append(del_nodes_indices, idx)

            node_options = np.delete(node_options, del_nodes_indices)
            branch_options = np.delete(branch_options, del_nodes_indices)
            del_branch_indices = np.array([], dtype=int)
            for idx in range(len(branch_options)):
                if self.Branch_velocity[branch_options[idx]] == 0:
                    del_branch_indices = np.append(del_branch_indices, idx)

            node_options = np.delete(node_options, del_branch_indices)
            branch_options = np.delete(branch_options, del_branch_indices)
            #print(branch_options)
            #print(node_options)
            velocity_arr2 = np.array([])
            for idx in range(len(node_options)):
                velocity_arr2 = np.append(velocity_arr2, self.Branch_velocity[branch_options[idx]])

            #print(velocity_arr2)
            if len(velocity_arr2) == 0:
                return [], []
            indx = np.argmax(velocity_arr2)
            #print(indx)
            moving_branch = branch_options[indx]
            node_travelling_to = node_options[indx]

            return [moving_branch], [node_travelling_to]

    def delt_Tmin_calc(self, flag_first):
        """
        Finds out which event is going to occur next, given position velocities of each droplet, and at what time it will occur.
        """

        incidence_matrix = self.incidence_matrix
        Network_solution = self.Network_solution
        cross_area = self.width_channel * self.height_channel
        decision = {'val': 0, 'branch': [], 'node_num': []}
        source_velocity = self.Q_in / cross_area
        # Droplet_frqncy = self.Droplet_frqncy
        sim_drps = self.sim_drops
        self.velocity_calc_func()
        self.Branch_velocity_prev = {}
        length_branches = self.calc_length_branches()
        to_insert = False

        if flag_first == 1:
            c = []
            for item in self.branch_drp_num.values():
                c = np.append(c, list(item))
            #print(c)
            self.c = c
            dup_list = []
            if len(c) > 1:
                new_list = sorted(c)
                for i in range(len(new_list)):
                    if list(c).count(new_list[i]) > 1:
                        if new_list[i] not in dup_list:
                            dup_list.append(new_list[i])
           # print(dup_list)
            self.dup_list = dup_list
            if len(dup_list) > 0:
                br_opt = []
                a1 = []
                for ndx in range(len(dup_list)):
                    br_opt = np.append(br_opt,[k for k,v in self.branch_drp_num.items() if dup_list[ndx] in v])
                    #print(br_opt)
                    for adx in range(len(br_opt)):
                        ind_to_add = self.branch_drp_num[br_opt[adx]]
                        #print(ind_to_add)
                        if dup_list[ndx] in list(ind_to_add):
                            a1.append(list(ind_to_add).index(dup_list[ndx]))
                        else:
                            continue
                        #print(a1)
                #print(br_opt)
                #print(a1)
                self.br_opt = br_opt
                self.a1 = a1
                for hdx in range(len(self.nodes)):
                    if all(item in list(self.br_opt) for item in list(self.branch_cnctvty_branchs_cnctd[self.nodes[hdx]])) is True:
                    #if list(self.branch_cnctvty_branchs_cnctd[self.nodes[hdx]]) == list(br_opt):
                        br_mod = list(self.branch_cnctvty_branchs_cnctd[self.nodes[hdx]])
                        a1_indic = [list(br_opt).index(i) for i in br_mod]
                        a1_mod = [a1[j] for j in a1_indic]
                        node_branch = self.nodes[hdx]
                        #print("hey")
                        #print(self.branch_drp_pstn)
                        decision['branch'],decision['node_num'] = self.drop_movement(node_branch,3)
                        if len(decision['branch']) == 0:
                            tempo = decision['branch']
                            #print("not this")
                            to_insert = True
                            self.to_insert = to_insert
                            self.tempo = tempo
                            for cdx in range(len(br_mod)):
                                self.branch_drp_pstn[br_mod[cdx]] = np.delete(self.branch_drp_pstn[br_mod[cdx]], a1_mod[cdx])
                            continue

                        else:
                            to_insert = False
                            self.to_insert = to_insert
                            moving_branch_to = decision['branch'][0]
                            node_to = decision['node_num']
                            moving_branch_from = list(br_opt)
                            node_from = node_branch
                            self.node_from = node_from
                            #print(node_from)
                            #print(moving_branch_to)
                            #print(node_to)
                            node_options = []
                            for mdx in range(len(br_opt)):
                                tmp = np.setdiff1d(self.branch_cnctvty_nodes_cnctd[br_opt[mdx]],np.array([node_from]))[0]
                                node_options = np.append(node_options, tmp)
                            #print(node_options)
                            self.node_options = node_options
                            tochoose_moving_branch = [moving_branch_from[n] for n in range(len(moving_branch_from)) if self.Network_solution[self.node_indices[node_from]] - self.Network_solution[self.node_indices[node_options[n]]] < 0]
                            tochoose_moving_branch_indx = [n for n in range(len(moving_branch_from)) if self.Network_solution[self.node_indices[node_from]] - self.Network_solution[self.node_indices[node_options[n]]] < 0]
                           # print(tochoose_moving_branch)
                            #print(tochoose_moving_branch_indx)
                            x1 = self.branch_drp_type[tochoose_moving_branch[0]][a1[tochoose_moving_branch_indx[0]]]  #, list(self.branch_drp_time[tochoose_moving_branch])[tochoose_moving_branch_indx], list(self.branch_drp_num[tochoose_moving_branch])[tochoose_moving_branch_indx]]
                            #print(x1)
                            y1 = self.branch_drp_time[tochoose_moving_branch[0]][a1[tochoose_moving_branch_indx[0]]]
                            #print(y1)
                            z1 = self.branch_drp_num[tochoose_moving_branch[0]][a1[tochoose_moving_branch_indx[0]]]
                            #print(z1)

                            for key in self.branch_drp_time:
                                self.branch_drp_time[key] = self.branch_drp_time[key]

                            for key in self.branch_drp_pstn:
                                if self.Branch_velocity[key] * self.Branch_velocity_prev[key] > 0:
                                    self.branch_drp_pstn[key] = self.branch_drp_pstn[key]
                                else:
                                    self.branch_drp_pstn[key] = 1 - self.branch_drp_pstn[key]
                                    # print("why here")
                                    self.branch_drp_pstn[key] = self.branch_drp_pstn[key]

                            for ldx in range(len(moving_branch_from)):
                                self.branch_drp_pstn[moving_branch_from[ldx]] = np.delete(self.branch_drp_pstn[moving_branch_from[ldx]], a1[ldx])
                                self.branch_drp_type[moving_branch_from[ldx]] = np.delete(self.branch_drp_type[moving_branch_from[ldx]], a1[ldx])
                                self.branch_drp_time[moving_branch_from[ldx]] = np.delete(self.branch_drp_time[moving_branch_from[ldx]], a1[ldx])
                                self.branch_drp_num[moving_branch_from[ldx]] = np.delete(self.branch_drp_num[moving_branch_from[ldx]], a1[ldx])


                            if self.Network_solution[self.node_indices[node_from]] - self.Network_solution[self.node_indices[node_to[0]]] > 0:
                                self.branch_drp_pstn[moving_branch_to] = np.insert(self.branch_drp_pstn[moving_branch_to], 0, 0)
                                self.branch_drp_type[moving_branch_to] = np.insert(self.branch_drp_type[moving_branch_to], 0, x1)
                                self.branch_drp_time[moving_branch_to] = np.insert(self.branch_drp_time[moving_branch_to], 0, y1)
                                self.branch_drp_num[moving_branch_to] = np.insert(self.branch_drp_num[moving_branch_to], 0, z1)
                            elif self.Network_solution[self.node_indices[node_from]] - self.Network_solution[self.node_indices[node_to[0]]] < 0:
                                self.branch_drp_pstn[moving_branch_to] = np.insert(self.branch_drp_pstn[moving_branch_to], -1, 1)
                                self.branch_drp_type[moving_branch_to] = np.insert(self.branch_drp_type[moving_branch_to], -1, x1)
                                self.branch_drp_time[moving_branch_to] = np.insert(self.branch_drp_time[moving_branch_to], -1, y1)
                                self.branch_drp_num[moving_branch_to] = np.insert(self.branch_drp_num[moving_branch_to], -1, z1)


                            #print(self.branch_drp_pstn)
                            #print(self.branch_drp_num)
                            #print(self.branch_drp_type)
                            #print(self.branch_drp_time)


            #print(self.branch_drp_pstn)

            self.Branch_velocity_prev = self.Branch_velocity
            self.velocity_calc_func()
            sink_node_branchs = np.array([])
            for idx in range(len(self.sink_nodes)):
                sink_node_branchs = np.append(sink_node_branchs,self.branch_cnctvty_branchs_cnctd[self.sink_nodes[idx]])
            #print('func_delt_tmin')
            delt_Tmin_branchs = {}
            delt_Tmin_source_nodes = {}
            for idx in range(self.num_branches):
                """There are drops in the branch and the branch velocity is not q"""
                if len(self.branch_drp_pstn[self.branches[idx]]) != 0 and self.Branch_velocity[self.branches[idx]] != 0:
                    delt_Tmin_branchs.update({self.branches[idx]: abs(1 - self.branch_drp_pstn[self.branches[idx]][-1]) * length_branches[self.branches[idx]] / (self.beta*abs(self.Branch_velocity[self.branches[idx]]))})
                """elif len(self.branch_drp_pstn[self.branches[idx]]) != 0 and self.Branch_velocity[self.branches[idx]] < 0:
                    delt_Tmin_branchs.update({self.branches[idx]: abs(self.branch_drp_pstn[self.branches[idx]][-1]) * length_branches[self.branches[idx]] / abs(self.Branch_velocity[self.branches[idx]])})"""

            for idx in range(len(self.source_nodes)):
                if len(self.entrance_drps[self.source_nodes[idx]]) > 0:
                    delt_Tmin_source_nodes.update({self.source_nodes[idx]: abs(self.entrance_drp_pstn[self.source_nodes[idx]][0]) / (self.beta*source_velocity)})
            temp_delt_branchs = delt_Tmin_branchs.copy()
            self.temp_delt_branchs = temp_delt_branchs
            dict_temp_delt = sorted(temp_delt_branchs.items(), key=lambda x: x[1])
            sortd_dict_temp_tmin = []
            for mdx in range(len(dict_temp_delt)):
                sortd_dict_temp_tmin.append(dict_temp_delt[mdx][0])
            self.sortd_dict_temp_tmin = sortd_dict_temp_tmin
            #print(sortd_dict_temp_tmin)
            #print(delt_Tmin_branchs)
            if len(sortd_dict_temp_tmin) == 0 and len(delt_Tmin_branchs) == 0:
                self.toterminate = True
                #print('nothing in here')
                return
                #sys.exit("droplet trapped")
            if len(dup_list) > 0:
                if sortd_dict_temp_tmin[0] in self.br_opt and len(sortd_dict_temp_tmin) > 1:
                    self.toterminate = True
                    #print("sober")
                    return
                elif sortd_dict_temp_tmin[0] in self.br_opt and len(sortd_dict_temp_tmin) == 1:
                    self.toterminate = True
                    #print('sober_2')
                    return
                else:
                    if len(decision['branch']) == 0:
                        to_remove = []
                        for key in delt_Tmin_branchs:
                            if key in self.br_opt:
                                to_remove.append(key)
                        #print(to_remove)
                        self.to_remove = to_remove
                        for gdx in range(len(to_remove)):
                            del delt_Tmin_branchs[to_remove[gdx]]
            #print(delt_Tmin_branchs.keys())
            delt_Tmin = np.concatenate((np.ravel(np.array(list(delt_Tmin_branchs.values()))), np.array(list(delt_Tmin_source_nodes.values()))))
            #print(delt_Tmin_source_nodes)
            #print(delt_Tmin_branchs)
            self.delt_Tmin_branchs = delt_Tmin_branchs
            dict_delt = sorted(delt_Tmin_branchs.items(), key=lambda x: x[1])
            sortd_dict_tmin = []
            for mdx in range(len(dict_delt)):
                sortd_dict_tmin.append(dict_delt[mdx][0])
            self.sortd_dict_tmin = sortd_dict_tmin
            #print(sortd_dict_tmin)
            #print(tmp_intersection_branches)
            #print(delt_Tmin)
            indx = np.argmin(delt_Tmin)
            #print(delt_Tmin)
            #print(indx)
            if indx >= len(delt_Tmin_branchs.keys()):
                """check if new drop enters the network"""
                """node_or_branch is one of the source node"""
                node_or_branch = np.array(list(delt_Tmin_source_nodes.keys()))[indx - len(delt_Tmin_branchs.keys())]

                self.delt_Tmin_val = delt_Tmin_source_nodes[node_or_branch]
                decision['val'] = 0
                decision['branch'], decision['node_num'] = self.drop_movement(node_or_branch, 0)
                self.decision = decision

            else:
                """check if existing one goes out of network"""
                """node_or_branch is a branch"""
                node_or_branch = np.ravel(np.array(list(delt_Tmin_branchs.keys())))[indx]
                #print(node_or_branch)
                #node_or_branch1 = np.ravel(np.array(list(delt_Tmin_branchs.keys())))
                """if node_or_branch in tmp_intersection_branches:
                    intersection_change = True"""
                self.delt_Tmin_val = delt_Tmin_branchs[node_or_branch]

                if node_or_branch in sink_node_branchs:
                    # Drop exits the network
                    self.delt_Tmin_val = delt_Tmin_branchs[node_or_branch]
                    decision['val'] = 1
                    decision['branch'], decision['node_num'] = self.drop_movement(node_or_branch, 1)
                    self.decision = decision


                else:
                    """Drop changes from one branch to another within the network"""
                    #print('phew')
                    decision['val'] = 2
                    self.delt_Tmin_val = delt_Tmin_branchs[sortd_dict_tmin[0]]
                    #branches_reject = []
                    #for bdx in range(len(sortd_dict_tmin)):
                    decision['branch'], decision['node_num'] = self.drop_movement(sortd_dict_tmin[0],2)
                    #print(type(decision['branch']))
                    self.decision = decision


        elif flag_first == 0:
            dup_list = []
            trap_secnd = int()
            self.dup_list = dup_list
            self.trap_secnd = trap_secnd

            self.velocity_calc_func()
            Branch_velocity_prev = {}
            for idx in range(self.num_branches):
                Branch_velocity_prev.update({self.branches[idx]: 0.})
                """else:
                    Branch_velocity_prev.update({self.branches[idx]: self.Branch_velocity[idx]})"""
            self.Branch_velocity_prev = Branch_velocity_prev
            #print(self.Branch_velocity_prev)
            #print("jhk")
            delt_Tmin = np.array([])
            for idx in range(len(self.source_nodes)):
                if (len(self.entrance_drp_pstn[self.source_nodes[idx]]) > 0):
                    delt_Tmin = np.append(delt_Tmin, abs(self.entrance_drp_pstn[self.source_nodes[idx]][0]) / (self.beta*source_velocity))
                else:
                    delt_Tmin = np.append(delt_Tmin, float('inf'))

            indx = np.argmin(delt_Tmin)
            node_or_branch = self.source_nodes[indx]
            decision['val'] = 0
            decision['branch'], decision['node_num'] = self.drop_movement(node_or_branch, 0)
            self.delt_Tmin_val = delt_Tmin[indx]
            self.decision = decision

        return

    def Droplet_Topology_matrix(self, flag_first):

        """
        Once the delta_tmin is calculated, and the next event is determined, the droplet will move to a different branch.
        """

        cross_area = self.width_channel * self.height_channel
        source_velocity = self.Q_in / cross_area
        length_branches = self.calc_length_branches()
        branches = self.branches
        nodes = self.nodes
        branch_drp_pstn = self.branch_drp_pstn
        branch_drp_num = self.branch_drp_num
        branch_drp_type = self.branch_drp_type
        branch_drp_time = self.branch_drp_time
        branch_num_drps = self.branch_num_drps


        # Move the drops in the source channel

        for key in branch_drp_time:

            branch_drp_time[key] = branch_drp_time[key] + self.delt_Tmin_val

        for key in branch_drp_pstn:
            if len(self.dup_list) > 0:
                if len(self.sortd_dict_temp_tmin) > 1:
                    if key == self.trap_secnd:
                        if self.trap_secnd in self.br_opt:
                            if self.Branch_velocity[key] * self.Branch_velocity_prev[key] >= 0:
                                branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * self.beta*abs(self.Branch_velocity[key]) / \
                                                       length_branches[key] - self.width_channel
                            else:
                                branch_drp_pstn[key] = 1 - branch_drp_pstn[key]
                                #print("why here")
                                branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * self.beta* abs(self.Branch_velocity[key]) / \
                                                       length_branches[key] - self.width_channel
                    else:
                        if self.Branch_velocity[key] * self.Branch_velocity_prev[key] >= 0:
                            branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * self.beta*abs(self.Branch_velocity[key]) / \
                                                   length_branches[key]
                        else:
                            branch_drp_pstn[key] = 1 - branch_drp_pstn[key]
                            # print("why here")
                            branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * self.beta*abs(self.Branch_velocity[key]) / \
                                                   length_branches[key]

            else:
                if self.Branch_velocity[key] * self.Branch_velocity_prev[key] >= 0:
                    branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * self.beta*abs(self.Branch_velocity[key]) / \
                                           length_branches[key]
                else:
                    branch_drp_pstn[key] = 1 - branch_drp_pstn[key]
                    # print("why here")
                    branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * self.beta*abs(self.Branch_velocity[key]) / \
                                           length_branches[key]

            """else:
                if key in self.branches_reject:
                    branch_drp_pstn[key] = branch_drp_pstn[key]


                elif (self.Branch_velocity[key] * self.Branch_velocity_prev[key] > 0):
                    branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * abs(self.Branch_velocity[key]) / \
                                           length_branches[key]

                else:
                    branch_drp_pstn[key] = 1 - branch_drp_pstn[key]
                    #print("shit man")
                    branch_drp_pstn[key] = branch_drp_pstn[key] + self.delt_Tmin_val * abs(self.Branch_velocity[key]) / \
                                   length_branches[key]"""

        for key in branch_drp_pstn:
            indices_sort = np.argsort(branch_drp_pstn[key])
            branch_drp_pstn[key] = branch_drp_pstn[key][indices_sort]

        if len(self.source_nodes) == 1:
            self.entrance_drp_pstn[self.source_nodes[0]] = self.entrance_drp_pstn[self.source_nodes[
                0]] + self.delt_Tmin_val * self.beta*source_velocity
        else:
            for idx in range(len(self.source_nodes)):
                self.entrance_drp_pstn[self.source_nodes[idx]] = self.entrance_drp_pstn[self.source_nodes[
                    idx]] + self.delt_Tmin_val * self.beta*source_velocity

        for idx in range(len(self.sink_nodes)):
            if len(self.exit_drp_res_time[self.sink_nodes[idx]]) > 0:
                self.exit_drp_pstn[self.sink_nodes[idx]] = self.exit_drp_pstn[self.sink_nodes[idx]] + (self.delt_Tmin_val * self.sink_velocity[self.sink_nodes[idx]])

        if flag_first == 1:
            if len(self.dup_list) > 0:
                #print(self.to_insert)
                for bdx in range(len(self.sink_nodes)):
                    if self.to_insert is True and (len(self.c) + len(self.exit_drp_res_time[self.sink_nodes[bdx]]))> 4:
                        #print("help")
                        for hdx in range(len(self.nodes)):
                            if all(item in list(self.br_opt) for item in list(self.branch_cnctvty_branchs_cnctd[self.nodes[hdx]])) is True:
                                #print('let it')
                                br_mod = list(self.branch_cnctvty_branchs_cnctd[self.nodes[hdx]])
                                a1_indic = [list(self.br_opt).index(i) for i in br_mod]
                                a1_mod = [self.a1[j] for j in a1_indic]
                                node_branch = self.nodes[hdx]
                                node_opt = []
                                #print(self.to_insert)
                                #print(self.temp_delt_branchs.keys())
                                #print(self.delt_Tmin_branchs.keys())
                                for mdx in range(len(br_mod)):
                                    tmp = np.setdiff1d(self.branch_cnctvty_nodes_cnctd[br_mod[mdx]], np.array([node_branch]))[0]
                                    node_opt = np.append(node_opt, tmp)
                                #print(node_branch)
                                #print(node_opt)
                                for adx in range(len(br_mod)):
                                    if self.Network_solution[self.node_indices[node_branch]] - self.Network_solution[self.node_indices[node_opt[adx]]] < 0:
                                        self.branch_drp_pstn[br_mod[adx]] = np.insert(self.branch_drp_pstn[br_mod[adx]], a1_mod[adx], 1)
                                    elif self.Network_solution[self.node_indices[node_branch]] - self.Network_solution[self.node_indices[node_opt[adx]]] > 0:
                                        self.branch_drp_pstn[br_mod[adx]] = np.insert(self.branch_drp_pstn[br_mod[adx]], a1_mod[adx], 0)
                                    elif self.Network_solution[self.node_indices[node_branch]] - self.Network_solution[self.node_indices[node_opt[adx]]] == 0:
                                        self.branch_drp_pstn[br_mod[adx]] = np.insert(self.branch_drp_pstn[br_mod[adx]], a1_mod[adx], 1)
                                continue

            if self.decision['val'] == 0:
                # Droplet enters
                self.entrance_drp_cntr = self.entrance_drp_cntr + 1
                moving_branch_to = self.decision['branch']
                source_num = self.decision['node_num'][0]

                self.branch_drp_pstn[moving_branch_to] = np.insert(self.branch_drp_pstn[moving_branch_to], 0, 0)
                self.branch_drp_type[moving_branch_to] = np.insert(self.branch_drp_type[moving_branch_to], 0,
                                                                   self.entrance_drp_type[source_num][0])
                self.branch_drp_time[moving_branch_to] = np.insert(self.branch_drp_time[moving_branch_to], 0, 0)
                self.branch_drp_num[moving_branch_to] = np.insert(self.branch_drp_num[moving_branch_to], 0,
                                                                  self.entrance_drps[source_num][0])

                self.entrance_drps[source_num] = np.delete(self.entrance_drps[source_num], 0)
                self.entrance_drp_type[source_num] = np.delete(self.entrance_drp_type[source_num], 0)
                self.entrance_drp_pstn[source_num] = np.delete(self.entrance_drp_pstn[source_num], 0)

            elif self.decision['val'] == 1:
                # Droplet exits
                self.exit_drp_cntr = self.exit_drp_cntr + 1
                moving_branch_from = self.decision['branch']
                node_from = self.decision['node_num'][0]
                sink_num = self.decision['node_num'][1]

                self.exit_drps[sink_num] = np.append(self.exit_drps[sink_num], self.branch_drp_num[moving_branch_from][-1])
                self.exit_drp_type[sink_num] = np.append(self.exit_drp_type[sink_num], self.branch_drp_type[moving_branch_from][-1])
                self.exit_drp_res_time[sink_num] = np.append(self.exit_drp_res_time[sink_num], self.branch_drp_time[moving_branch_from][-1])

                x = self.exit_drps[sink_num]
                self.d.update({x[-1]: [moving_branch_from, node_from, sink_num]})

                if len(self.exit_drp_pstn[sink_num]) != 0:
                    self.exit_drp_pstn[sink_num] = self.exit_drp_pstn[sink_num].tolist()
                    #print(self.exit_drp_pstn[sink_num][0][0])
                    if self.exit_drp_pstn[sink_num][0][0] > 0:
                        # One drop already exists in the sink node branch
                        self.exit_drp_spacing[sink_num] = np.append(self.exit_drp_spacing[sink_num], self.exit_drp_pstn[sink_num][0][0])
                else:
                    if self.exit_drp_pstn[sink_num] > 0:
                        # One drop already exists in the sink node branch
                        self.exit_drp_spacing[sink_num] = np.append(self.exit_drp_spacing[sink_num], self.exit_drp_pstn[sink_num])

                for key in self.exit_drp_pstn:
                    if key == sink_num:
                        self.exit_drp_pstn[key] = np.insert(self.exit_drp_pstn[key], 0, 0)


                if len(self.branch_drp_type[moving_branch_from]) > 1:
                    self.branch_drp_pstn[moving_branch_from] = np.delete(self.branch_drp_pstn[moving_branch_from], -1)
                    self.branch_drp_type[moving_branch_from] = np.delete(self.branch_drp_type[moving_branch_from], -1)
                    self.branch_drp_time[moving_branch_from] = np.delete(self.branch_drp_time[moving_branch_from], -1)
                    self.branch_drp_num[moving_branch_from] = np.delete(self.branch_drp_num[moving_branch_from], -1)
                else:
                    self.branch_drp_pstn[moving_branch_from] = np.delete(self.branch_drp_pstn[moving_branch_from], 0)
                    self.branch_drp_type[moving_branch_from] = np.delete(self.branch_drp_type[moving_branch_from], 0)
                    self.branch_drp_time[moving_branch_from] = np.delete(self.branch_drp_time[moving_branch_from], 0)
                    self.branch_drp_num[moving_branch_from] = np.delete(self.branch_drp_num[moving_branch_from], 0)
                if self.exit_drp_cntr == self.entrance_drp_cntr and self.exit_drp_cntr < self.sim_drops:
                    flag_first = 0
                    

            elif self.decision['val'] == 2:
                if len(self.decision['branch']) == 2:

                    moving_branch_from = self.decision['branch'][0]
                    moving_branch_to = self.decision['branch'][1]

                    if moving_branch_to != moving_branch_from:
                        self.branch_drp_pstn[moving_branch_to] = np.insert(self.branch_drp_pstn[moving_branch_to], 0, 0)
                        self.branch_drp_type[moving_branch_to] = np.insert(self.branch_drp_type[moving_branch_to], 0,
                                                                           self.branch_drp_type[moving_branch_from][-1])
                        self.branch_drp_time[moving_branch_to] = np.insert(self.branch_drp_time[moving_branch_to], 0,
                                                                           self.branch_drp_time[moving_branch_from][-1])
                        self.branch_drp_num[moving_branch_to] = np.insert(self.branch_drp_num[moving_branch_to], 0,
                                                                          self.branch_drp_num[moving_branch_from][-1])

                        if len(self.branch_drp_type[moving_branch_from]) > 1:
                            self.branch_drp_pstn[moving_branch_from] = np.delete(self.branch_drp_pstn[moving_branch_from], -1)
                            self.branch_drp_type[moving_branch_from] = np.delete(self.branch_drp_type[moving_branch_from], -1)
                            self.branch_drp_time[moving_branch_from] = np.delete(self.branch_drp_time[moving_branch_from], -1)
                            self.branch_drp_num[moving_branch_from] = np.delete(self.branch_drp_num[moving_branch_from], -1)
                        else:
                            self.branch_drp_pstn[moving_branch_from] = np.delete(self.branch_drp_pstn[moving_branch_from], 0)
                            self.branch_drp_type[moving_branch_from] = np.delete(self.branch_drp_type[moving_branch_from], 0)
                            self.branch_drp_time[moving_branch_from] = np.delete(self.branch_drp_time[moving_branch_from], 0)
                            self.branch_drp_num[moving_branch_from] = np.delete(self.branch_drp_num[moving_branch_from], 0)
                    elif moving_branch_to == moving_branch_from:
                        if (not(self.subdue_print)):
                            print('here hain')
                        #print(self.branch_drp_pstn)
                        #print(self.branch_drp_type)
                        #print(self.branch_drp_time)
                        #print(self.branch_drp_num)
                        #self.branch_drp_pstn[moving_branch_from] = self.branch_drp_pstn


                elif len(self.decision['branch']) == 1:
                    #print("yourehere")
                    moving_branch_from = self.decision['branch'][0]
                    moving_branch_to = self.branch_cnctvty_branchs_cnctd[self.decision['node_num'][0]]
                    node_opt = []
                    for mdx in range(len(moving_branch_to)):
                        tmp = np.setdiff1d(self.branch_cnctvty_nodes_cnctd[moving_branch_to[mdx]], np.array([self.decision['node_num'][0]]))[0]
                        node_opt = np.append(node_opt, tmp)
                    #print(node_opt)
                    #print(self.branch_drp_pstn)
                    #print(self.branch_drp_num)
                    for jdx in range(len(moving_branch_to)):
                        if self.Network_solution[self.node_indices[self.decision['node_num'][0]]] - self.Network_solution[self.node_indices[node_opt[jdx]]] <= 0:
                            if moving_branch_to[jdx] == moving_branch_from:
                                continue
                            else:
                                if len(self.branch_drp_num[moving_branch_to[jdx]]) > 0:
                                    self.branch_drp_pstn[moving_branch_to[jdx]] = np.insert(self.branch_drp_pstn[moving_branch_to[jdx]], len(self.branch_drp_pstn[moving_branch_to[jdx]]), 1)
                                    self.branch_drp_type[moving_branch_to[jdx]] = np.insert(self.branch_drp_type[moving_branch_to[jdx]], len(self.branch_drp_type[moving_branch_to[jdx]]), self.branch_drp_type[moving_branch_from][-1])
                                    self.branch_drp_time[moving_branch_to[jdx]] = np.insert(self.branch_drp_time[moving_branch_to[jdx]], len(self.branch_drp_time[moving_branch_to[jdx]]), self.branch_drp_time[moving_branch_from][-1])
                                    self.branch_drp_num[moving_branch_to[jdx]] = np.insert(self.branch_drp_num[moving_branch_to[jdx]], len(self.branch_drp_num[moving_branch_to[jdx]]), self.branch_drp_num[moving_branch_from][-1])
                                else:
                                    self.branch_drp_pstn[moving_branch_to[jdx]] = np.insert(self.branch_drp_pstn[moving_branch_to[jdx]], 0, 1)
                                    self.branch_drp_type[moving_branch_to[jdx]] = np.insert(self.branch_drp_type[moving_branch_to[jdx]], 0, self.branch_drp_type[moving_branch_from][-1])
                                    self.branch_drp_time[moving_branch_to[jdx]] = np.insert(self.branch_drp_time[moving_branch_to[jdx]], 0, self.branch_drp_time[moving_branch_from][-1])
                                    self.branch_drp_num[moving_branch_to[jdx]] = np.insert(self.branch_drp_num[moving_branch_to[jdx]], 0, self.branch_drp_num[moving_branch_from][-1])
                        elif self.Network_solution[self.node_indices[self.decision['node_num'][0]]] - self.Network_solution[self.node_indices[node_opt[jdx]]] >= 0:

                            self.branch_drp_pstn[moving_branch_to[jdx]] = np.insert(self.branch_drp_pstn[moving_branch_to[jdx]], 0, 0)
                            self.branch_drp_type[moving_branch_to[jdx]] = np.insert(self.branch_drp_type[moving_branch_to[jdx]], 0, self.branch_drp_type[moving_branch_from][-1])
                            self.branch_drp_time[moving_branch_to[jdx]] = np.insert(self.branch_drp_time[moving_branch_to[jdx]], 0, self.branch_drp_time[moving_branch_from][-1])
                            self.branch_drp_num[moving_branch_to[jdx]] = np.insert(self.branch_drp_num[moving_branch_to[jdx]], 0, self.branch_drp_num[moving_branch_from][-1])



            # branch_num_drps = {}
            new_branch_num_drps = {}
            # TODO: Support for multiple droplet types
            for idx in range(self.num_branches):
                # branch_num_drps.update({self.branches[idx]: np.zeros(2, dtype=int)})
                # branch_num_drps[self.branches[idx]][0] = len(np.where(branch_drp_type[self.branches[idx]] == 0)[0])
                # branch_num_drps[self.branches[idx]][1] = len(np.where(branch_drp_type[self.branches[idx]] == 1)[0])

                new_branch_num_drps.update({self.branches[idx]: np.zeros(self.droplet_type_count, dtype=int)})
                for jdx in range(self.droplet_type_count):
                    new_branch_num_drps[self.branches[idx]][jdx] = len(np.where(branch_drp_type[self.branches[idx]] == jdx)[0])

            self.branch_num_drps = new_branch_num_drps

        else:
            self.entrance_drp_cntr = self.entrance_drp_cntr + 1
            source_num = self.decision['node_num'][0]
            moving_branch_to = self.decision['branch']
            self.branch_drp_pstn[moving_branch_to] = np.append(self.branch_drp_pstn[moving_branch_to], 0)
            self.branch_drp_type[moving_branch_to] = np.append(self.branch_drp_type[moving_branch_to],
                                                               self.entrance_drp_type[source_num][0])
            self.branch_drp_time[moving_branch_to] = np.append(self.branch_drp_time[moving_branch_to], 0)
            self.branch_drp_num[moving_branch_to] = np.append(self.branch_drp_num[moving_branch_to],
                                                              self.entrance_drps[source_num][0])

            if self.entrance_drp_type[source_num][0] == 0:
                self.branch_num_drps[moving_branch_to][0] += 1
            else:
                self.branch_num_drps[moving_branch_to][1] += 1

            self.entrance_drps[source_num] = np.delete(self.entrance_drps[source_num], 0)
            self.entrance_drp_type[source_num] = np.delete(self.entrance_drp_type[source_num], 0)
            self.entrance_drp_pstn[source_num] = np.delete(self.entrance_drp_pstn[source_num], 0)

            flag_first = 1

        return flag_first

    def network_structure(self):

        """
        Used for plotting the network during image generation.
        """
        orig_randstate = np.random.get_state()
        for_struct = []
        P=[]
        Q=[]
        node_cordinate={}
        if self.should_gen_images:
            plt.xlim(-2 * self.length_factor, (self.grid_dim + 1) * self.length_factor)
            plt.ylim(-2 * self.length_factor, (self.grid_dim + 1) * self.length_factor)
        L=(1/1000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000)), int(self.length_factor*(1000)))
        #print(L)
        np.random.seed(self.rn1)
        #imp_node1 = np.multiply(L,(1.01-0.99)*np.random.random_sample(3)+0.99)
        #imp_node1 =np.multiply(L, np.array(([0.99999972, 0.99999775, 0.99999615]))) #0.99999546, 0.99999864])))
        G=np.multiply(((1/1000000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000000)), int(self.length_factor*(1000000)))),(1.0005-0.9995)*np.random.random_sample(self.grid_dim)+0.9995)
        #imp_node1 = np.multiply(L, np.array(([0.99999972, 0.99999775, 0.99999615])))#, 0.99999546, 0.99999864])))
        #G=np.multiply(((1/1000000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000000)), int(self.length_factor*(1000000)))),np.array(([0.99999972, 0.99999775, 0.99999615])))#,0.99999546, 0.99999864])));
        #print(G)
        np.random.seed(self.rn2)
        M=np.multiply(((1/1000000)*np.arange((self.grid_dim - 1)* int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)))),(1.0005-0.9995)*np.random.random_sample(self.grid_dim)+0.9995)
        #M=np.multiply(((1/1000000)*np.arange((self.grid_dim - 1)* int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)))),np.array(([1.00000067, 0.99999825, 1.00000435])))#,1.00000546, 1.00000348])))
        #print('M',M)
        for i in np.arange(0,len(G)):
            for k in np.arange(0,len(M)):
                for_struct.append([G[i], M[k]])
                #P[i][k]=G[i]
                #Q[i][k]=M[k] 
                
            if i!=len(G)-1:
                d=i#for d in np.arange(0,len(G)-1,1) :
                #if G[d]==i:
                np.random.seed(self.rn3)
                RN=(1.0005-0.9995)*np.random.random_sample((self.grid_dim-1)**2)+0.9995
                for j in np.arange(0,len(M)-1,1):
                    #print("j",j)
                    A=np.array(( [[ (M[j+1]-M[j]) / (G[d+1]-G[d]), -1], [ (M[j]-M[j+1]) / (G[d+1]-G[d]) , -1]]))
                    b=np.array(([[G[d]*(( M[j+1] - M[j] )/(G[d+1]-G[d]))-M[j]],[G[d]*(( M[j] - M[j+1] )/(G[d+1]-G[d]))-M[j+1]]]))
                    X=(np.linalg.inv(A)).dot(b)
                    #print(X[0])
                    #for_struct.append([np.float(X[0]), np.float(X[1])])
                    for_struct.append([np.float(X[0])*RN[(self.grid_dim-1)*d+j], np.float(X[1])*RN[-((self.grid_dim-1)*d+j)]])
                    #P[d][k]=G[i]
                    #Q[i][k]=M[k]
        for key in np.arange(0,self.grid_dim **2):
            node_cordinate.update({key:for_struct[key]})
        #print("node_cordinate",node_cordinate)
        
        #print("hahaah'",for_struct)
        #int(self.incidence_matrix)
        #print(self.branch_indices)
        #print(self.nodes)
        for jdx in self.branches:
            n1, n2 = int(
                np.where(self.incidence_matrix[self.branch_indices[jdx], :] == 1)[0]), int(
                np.where(self.incidence_matrix[self.branch_indices[jdx], :] == -1)[0])
            n1, n2 = self.nodes[n1], self.nodes[n2]
            #print(n1,n2)
            x1, y1, x2, y2 = for_struct[n1][0], for_struct[n1][1], for_struct[n2][0], for_struct[n2][1]
            if self.should_gen_images:
                plt.plot([x1, x2], [y1, y2], linewidth=5.0, color='cyan', zorder=1, alpha=0.4)
            #print("hahaah'",for_struct)
        if len(self.bypass_branches) != 0:
            for mdx in self.bypass_branches:
                n1, n2 = self.branch_cnctvty_nodes_cnctd[mdx][0], self.branch_cnctvty_nodes_cnctd[mdx][1]
                x1, y1, x2, y2 = for_struct[n1][0], for_struct[n1][1], for_struct[n2][0], for_struct[n2][1]
                if self.should_gen_images:
                    plt.plot([x1, x2], [y1, y2], linewidth=5.0, color='r', zorder=1, alpha=0.1)

        for idx in self.nodes:
            if idx in self.source_nodes:
                x1, y1 = for_struct[idx][0], for_struct[idx][1]
                if self.should_gen_images:
                    plt.plot([x1, x1 - 10 * self.length_factor], [y1, y1], linewidth=5.0, color='y', zorder=1, alpha=0.4)
            if idx in self.sink_nodes:
                x1, y1 = for_struct[idx][0], for_struct[idx][1]
                if self.should_gen_images:
                    plt.plot([x1, x1 + 10 * self.length_factor], [y1, y1], linewidth=5.0, color='y', zorder=1, alpha=0.4)

        self.for_struct = for_struct
        np.random.set_state(orig_randstate)
        return plt.plot

    def gen_images(self):
        """
        Generates the generated/defined network and runs the corresponding video of drop movement in the network
        """
        k = str(self.itr_num)
        for_struct = []
        plt.xlim(-2 * self.length_factor, (self.grid_dim + 1) * self.length_factor)
        plt.ylim(-2 * self.length_factor, (self.grid_dim + 1) * self.length_factor)
        imp_node = (1/100000)*np.arange(0, self.grid_dim * self.length_factor*(100000), self.length_factor*(100000))
        for i in (1/100000)*np.arange(0, (self.grid_dim - 0.5) * self.length_factor*(100000), self.length_factor*(100000) / 2):
            if i in imp_node:
                for j in (1/100000)*np.arange((self.grid_dim - 1) * self.length_factor*(100000), -self.length_factor*(100000), -self.length_factor*(100000)):
                    for_struct.append([i, j])
            else:
                for m in (1/100000)*np.arange((self.grid_dim - 1.5) * self.length_factor*(100000), -self.length_factor*(100000) / 2, -self.length_factor*(100000)):
                    for_struct.append([i, m])


        for jdx in self.branches:
            n1, n2 = int(np.where(self.incidence_matrix[self.branch_indices[jdx],:] == 1)[0]), int(np.where(self.incidence_matrix[self.branch_indices[jdx],:] == -1)[0])
            n1, n2 = self.nodes[n1], self.nodes[n2]
            #print('n1 n2',n1,n2)
            x1, y1, x2, y2 = for_struct[n1][0], for_struct[n1][1], for_struct[n2][0], for_struct[n2][1]
            #print('x1 y1',x1, y1, x2, y2)
            plt.plot([x1, x2], [y1, y2], linewidth=5.0, color='cyan', zorder=1, alpha=0.4)
            plt.text(x1,y1,n1,fontsize=4)
            plt.text(x2,y2,n2,fontsize=4)
            plt.text((x1+x2)/2,(y1+y2)/2, self.branch_indices[jdx],fontsize=4)                
        #print(for_struct)
        if len(self.bypass_branches) != 0:
            for mdx in self.bypass_branches:
                n1, n2 = self.branch_cnctvty_nodes_cnctd[mdx][0], self.branch_cnctvty_nodes_cnctd[mdx][1]
                x1, y1, x2, y2 = for_struct[n1][0], for_struct[n1][1], for_struct[n2][0], for_struct[n2][1]
                plt.plot([x1, x2], [y1, y2], linewidth=5.0, color='r', zorder=1, alpha=0.1)
                
        for idx in self.nodes:
            if idx in self.source_nodes:
                x1, y1 = for_struct[idx][0], for_struct[idx][1]
                plt.plot([x1, x1 - 10*self.length_factor], [y1, y1], linewidth=5.0, color='y', zorder=1, alpha=0.4)
            if idx in self.sink_nodes:
                x1, y1 = for_struct[idx][0], for_struct[idx][1]
                plt.plot([x1, x1 + 0.5*self.length_factor], [y1, y1], linewidth=5.0, color='y', zorder=1, alpha=0.4)

        for key in self.entrance_drp_pstn:
            for ndx in range(len(self.entrance_drp_pstn[key])):
                self.branch_drp_details.setdefault(k, []).append(np.array(['still in entrance', self.entrance_drps[key][ndx], self.entrance_drp_pstn[key][ndx], for_struct[key][1], self.beta*(self.Q_in / (self.width_channel * self.height_channel)) , 'still in entrance']))
                plt.scatter(self.entrance_drp_pstn[key][ndx], for_struct[key][1], s=10, facecolor='k', zorder=2)
                plt.text(self.entrance_drp_pstn[key][ndx], for_struct[key][1], self.entrance_drps[key][ndx], style='italic')

        for key in self.exit_drp_pstn:
            if len(self.exit_drp_pstn[key]) > 0:
                for adx in range(len(self.exit_drp_pstn[key])):
                    #print(self.exit_drp_pstn[key][adx])
                    if type(self.exit_drp_pstn[key][adx]) is np.float64 and self.exit_drp_pstn[key][adx] == 0.0:
                        #print(self.exit_drp_pstn[key][adx])
                        #print(adx, 'exit branch', self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - adx], for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], self.Branch_velocity[self.decision['branch']], self.decision['branch'])
                        self.branch_drp_details.setdefault(k, []).append(np.array(['exit branch', self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - adx], for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], self.beta*self.Branch_velocity[self.decision['branch']], self.decision['branch']]))
                        plt.scatter(for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], s=10, facecolor='k', zorder=2)
                        plt.text(for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - adx], style='italic')
                    else:
                        #print(self.exit_drp_pstn[key][adx])
                        if type(self.exit_drp_pstn[key][adx]) is np.float64 and self.exit_drp_pstn[key][adx] != 0.0:
                            if (not(self.subdue_print)):
                                print(adx, 'exit branch', self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - adx], for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], self.sink_velocity[key][0][0], 'exit branch')
                            self.branch_drp_details.setdefault(k, []).append(np.array(['exit branch', self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - adx], for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], self.beta*self.sink_velocity[key][0][0], 'exit branch']))
                            plt.scatter(for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], s=10, facecolor='k', zorder=2)
                            plt.text(for_struct[key][0] + self.exit_drp_pstn[key][adx], for_struct[key][1], self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - adx], style='italic')
                        else:
                            for bdx in range(len(self.exit_drp_pstn[key][adx])):
                                #print(adx, 'exit branch', self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - bdx], for_struct[key][0] + self.exit_drp_pstn[key][adx][bdx], for_struct[key][1], self.sink_velocity[key][0][0], 'exit branch')
                                self.branch_drp_details.setdefault(k, []).append(np.array(['exit branch', self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - bdx], for_struct[key][0] + self.exit_drp_pstn[key][adx][bdx], for_struct[key][1], self.beta*self.sink_velocity[key][0][0], 'exit branch']))
                                plt.scatter(for_struct[key][0] + self.exit_drp_pstn[key][adx][bdx], for_struct[key][1], s=10, facecolor='k', zorder=2)
                                plt.text(for_struct[key][0] + self.exit_drp_pstn[key][adx][bdx], for_struct[key][1], self.exit_drps[key][len(self.exit_drp_pstn[key]) - 1 - bdx], style='italic')

        for key in self.branch_drp_pstn:
            if len(self.branch_drp_pstn[key]) != 0:
                if self.decision['val'] == 0 or self.decision['val'] == 1:
                    branch_no = self.decision['branch']
                elif self.decision['val'] == 2:
                    branch_no = self.decision['branch'][0]
                branch_to_move = key
                drop_no = self.branch_drp_num[key]
                drop_postns = self.branch_drp_pstn[key]
                if len(drop_no) > 1:
                    for i in range(len(drop_no)):
                        if self.Branch_velocity[branch_to_move] >= 0:
                            [x1, y1] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][0]]
                            [x2, y2] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][1]]
                            x, y = x1 - drop_postns[i]*(x1-x2), y1 - drop_postns[i]*(y1-y2)
                            if self.decision['val'] == 0 or self.decision['val'] == 1:
                                if branch_to_move != self.decision['branch']:
                                    self.branch_drp_details.setdefault(k,[]).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                                else:
                                    self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                            else:
                                if branch_to_move not in self.decision['branch']:
                                    self.branch_drp_details.setdefault(k,[]).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                                else:
                                    self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                            plt.scatter(x,y, s=10, facecolor='k', zorder=2)
                            plt.text(x, y, drop_no[i], style='italic')
                        elif self.Branch_velocity[branch_to_move] < 0:
                            [x1, y1] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][1]]
                            [x2, y2] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][0]]
                            x, y = x1 - drop_postns[i] * (x1 - x2), y1 - drop_postns[i] * (y1 - y2)
                            if self.decision['val'] == 0 or self.decision['val'] == 1:
                                if branch_to_move != self.decision['branch']:
                                    self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                                else:
                                    self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                            else:
                                if branch_to_move not in self.decision['branch']:
                                    self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                                else:
                                    self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no[i], x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                            plt.scatter(x,y, s=10, facecolor='k', zorder=2)
                            plt.text(x, y, drop_no[i], style='italic')
                else:
                    if self.Branch_velocity[branch_to_move] >= 0:

                        [x1, y1] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][0]]
                        [x2, y2] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][1]]
                        x, y = x1 - drop_postns[0] * (x1 - x2), y1 - drop_postns[0] * (y1 - y2)
                        if self.decision['val'] == 0 or self.decision['val'] == 1:
                            if branch_to_move != self.decision['branch']:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                            else:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                        else:
                            if branch_to_move not in self.decision['branch']:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                            else:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                        plt.scatter(x,y, s=10, facecolor='k', zorder=2)
                        plt.text(x,y, drop_no[0], style='italic')
                    elif self.Branch_velocity[branch_to_move] < 0:
                        #print('br no', branch_no)
                        #print('br mo', branch_to_move)
                        if branch_no == branch_to_move:
                            [x1, y1] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][1]]
                            [x2, y2] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][0]]
                        else:
                            [x1, y1] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][1]]
                            [x2, y2] = for_struct[self.branch_cnctvty_nodes_cnctd[branch_to_move][0]]

                        x, y = x1 - drop_postns[0] * (x1 - x2), y1 - drop_postns[0] * (y1 - y2)
                        if self.decision['val'] == 0 or self.decision['val'] == 1:
                            if branch_to_move != self.decision['branch']:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                            else:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                        else:
                            if branch_to_move not in self.decision['branch']:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_to_move], branch_to_move]))
                            else:
                                self.branch_drp_details.setdefault(k, []).append(np.array([branch_to_move, drop_no, x, y, self.beta*self.Branch_velocity[branch_no], branch_no]))
                        plt.scatter(x,y, s=10, facecolor='k', zorder=2)
                        plt.text(x, y, drop_no[0])

        self.branch_drp_details = {int(key):self.branch_drp_details[key] for key in self.branch_drp_details}
        outpath = "image_folder"
        plt.savefig(os.path.join(outpath,"{}".format(.001*float(k)) + self.output_img_ext), dpi=400)
        plt.clf()

    def additional_images(self):
        """
        Additional images are generated to smoothen the transition from one event to another.
        This function will generate many in-between frames for video purposes.
        """
        length_branches = self.calc_length_branches()
        for key in self.branch_drp_details:
            #if key >= 1 and key <= 5 :
            if key != list(self.branch_drp_details.keys())[-1]:
                total_frames = math.floor(self.delt_tmin_details[key-1] / self.time_step)
                x_s = np.zeros((int(total_frames), len(self.branch_drp_details[key])))
                if (not(self.subdue_print)):
                    print(x_s.shape)
                #print(self.branch_drp_details[key])
                y_s = np.zeros((int(total_frames), len(self.branch_drp_details[key])))
                itr_frame = 0
                while itr_frame < total_frames-1:
                    for qdx in range(len(self.branch_drp_details[key])):
                        note_drop_no = self.branch_drp_details[key][qdx][1]
                        self.note_drop_no = note_drop_no
                        if self.branch_drp_details[key][qdx][0] == 'still in entrance':
                            br_vel = float(self.branch_drp_details[key][qdx][4])
                            if itr_frame == 0:
                                x_s[itr_frame][qdx], y_s[itr_frame][qdx] = float(self.branch_drp_details[key][qdx][2]), float(self.branch_drp_details[key][qdx][3])
                            else:
                                x_s[itr_frame][qdx] = x_s[itr_frame - 1][qdx] + (self.time_step * br_vel)
                                y_s[itr_frame][qdx] = y_s[itr_frame - 1][qdx]
                                plt.scatter(x_s[itr_frame][qdx],y_s[itr_frame][qdx], s=10, facecolor='k', zorder=2)
                                plt.text(x_s[itr_frame][qdx], y_s[itr_frame][qdx], self.note_drop_no, style='italic')
                        elif self.branch_drp_details[key][qdx][0] == 'exit branch':
                            for zdx in range(len(self.branch_drp_details[key + 1])):
                                if self.branch_drp_details[key + 1][zdx][1] == note_drop_no:
                                    br_vel = self.branch_drp_details[key + 1][zdx][4]
                                if itr_frame == 0:
                                    x_s[itr_frame][qdx], y_s[itr_frame][qdx] = self.branch_drp_details[key][qdx][2], self.branch_drp_details[key][qdx][3]
                                else:
                                    x_s[itr_frame][qdx] = x_s[itr_frame - 1][qdx] + (self.time_step * float(br_vel))
                                    y_s[itr_frame][qdx] = y_s[itr_frame - 1][qdx]
                                    plt.scatter(x_s[itr_frame][qdx],y_s[itr_frame][qdx], s=10, facecolor='k', zorder=2)
                                    plt.text(x_s[itr_frame][qdx], y_s[itr_frame][qdx], self.note_drop_no, style='italic')

                        else:
                            for ldx in range(len(self.branch_drp_details[key+1])):
                                if int(self.branch_drp_details[key+1][ldx][1]) == note_drop_no:
                                    br_vel = self.branch_drp_details[key+1][ldx][4]
                                    len_br = length_branches[int(self.branch_drp_details[key+1][ldx][5])]
                            if self.branch_drp_details[key][qdx][0] == self.branch_drp_details[key][qdx][5]:
                                x1_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][5]][0]][0]
                                x2_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][5]][1]][0]
                                y1_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][5]][0]][1]
                                y2_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][5]][1]][1]
                            if self.branch_drp_details[key][qdx][0] != self.branch_drp_details[key][qdx][5]:
                                x1_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][0]][0]][0]
                                x2_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][0]][1]][0]
                                y1_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][0]][0]][1]
                                y2_s = self.for_struct[self.branch_cnctvty_nodes_cnctd[self.branch_drp_details[key][qdx][0]][1]][1]
                            if itr_frame == 0:
                                x_s[itr_frame][qdx], y_s[itr_frame][qdx] = self.branch_drp_details[key][qdx][2], self.branch_drp_details[key][qdx][3]
                            else:
                                x_s[itr_frame][qdx] = x_s[itr_frame-1][qdx] - ((self.time_step*float(br_vel))/len_br)*(x1_s - x2_s)
                                y_s[itr_frame][qdx] = y_s[itr_frame-1][qdx] - ((self.time_step * float(br_vel)) / len_br)*(y1_s - y2_s)

                                plt.scatter(x_s[itr_frame][qdx],y_s[itr_frame][qdx], s=10, facecolor='k', zorder=2)
                                plt.text(x_s[itr_frame][qdx], y_s[itr_frame][qdx], self.note_drop_no, style='italic')

                    #print('x_s', x_s)
                    #print(str(self.note_drop_no))
                    self.network_structure()
                    plt.scatter(x_s[itr_frame,:], y_s[itr_frame,:], s=10, facecolor='k', zorder=2)
                    #plt.text(x_s[itr_frame,:], y_s[itr_frame,:], self.note_drop_no, style='italic', zorder=3)
                    outpath = "image_folder"
                    plt.savefig(os.path.join(outpath, "{}".format(key*(.001)+0.00000001*itr_frame) + self.output_img_ext), dpi=400)
                    itr_frame = itr_frame + 1
                    plt.clf()
                    #elif key == list(self.branch_drp_details.keys())[-1]:

    def gen_video(self):
        """
        Generates video, if the gen_images and additional_images functions were used. Stores under project.avi name.
        """
        frame_array = []
        pathIn = 'image_folder'
        # files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
        data_path = os.path.join('image_folder', '*' + self.output_img_ext)
        jpg_files = glob.glob(data_path)
        # for sorting the file names properly
        # files.sort(key=lambda x: int(x[5:-4]))
        # jpg_files.sort()
        if (not(self.subdue_print)):
            print(jpg_files)
        
        object_filename_mapping = {}
        filename_float_mapping = {}

        for filename in jpg_files:
            filename_float_mapping[filename] = float(filename.split('\\')[-1].split(self.output_img_ext)[0])
        
        sorted_filenames = sorted(filename_float_mapping, key=filename_float_mapping.get)

        filename = jpg_files[0]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        out = cv2.VideoWriter(pathIn+'/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 8, size)

        for i in range(len(sorted_filenames)):
            filename = sorted_filenames[i]
            # filename = os.path.join(pathIn, filename)
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            # writing to a image array
            out.write(img)
            
        out.release()

    def simulate_network_func(self):
        """
        Main method used to perform simulation from start to finish.
        """
        # Droplet_frqncy = self.Droplet_frqncy
        sim_drps = self.sim_drops
        cross_area = self.width_channel * self.height_channel
        self.incidence_matrix_gen()
        self.branch_connectivity_fn()
        entrance_drps = {}
        entrance_drp_pstn = {}
        entrance_drp_type = {}
        exit_drps = {}
        exit_drp_type = {}
        exit_drp_res_time = {}
        exit_drp_spacing = {}
        exit_drp_pstn = {}
        sink_node_branchs = {}
        d = {}
        branch_drp_details = {}
        drop_dict = {}
        drop_pos = {}
        delt_tmin_details = []
        source_velocity = self.Q_in / cross_area
        """(450 micro L * (10^-3L/m3) /cross_area)* Droplet_frqncy"""
        #source_spacing = source_velocity * Droplet_frqncy
        #print(source_spacing)
        #source_spacing_mod = list(source_spacing*np.arange(0,9,1))
        #exit_spacing = source_spacing ** 0.5
        sink_velocity = {}
        prev_drops_sum = 0
        for idx in range(len(self.source_nodes)):
            entrance_drps.update({self.source_nodes[idx]: np.arange(prev_drops_sum, prev_drops_sum + len(self.value_spacing[idx]), 1)})
            prev_drops_sum += len(self.value_spacing[idx])
            entrance_drp_pstn.update({self.source_nodes[idx]: self.value_spacing[idx]})
            entrance_drp_type.update({self.source_nodes[idx]: self.arrange[idx]})
            #odd_indices = np.arange(0, sim_drps, 2)
            #entrance_drp_type[self.source_nodes[idx]][odd_indices] = 1
        #print(entrance_drp_pstn)

        self.entrance_drps = entrance_drps
        self.entrance_drp_type = entrance_drp_type
        self.entrance_drp_pstn = entrance_drp_pstn

        for idx in range(len(self.sink_nodes)):
            exit_drps.update({self.sink_nodes[idx]: np.array([], dtype=int)})
            exit_drp_res_time.update({self.sink_nodes[idx]: np.array([])})
            exit_drp_type.update({self.sink_nodes[idx]: np.array([], dtype=int)})
            exit_drp_pstn.update({self.sink_nodes[idx]: np.array([])})
            sink_node_branchs.update({self.sink_nodes[idx]: self.branch_cnctvty_branchs_cnctd[self.sink_nodes[idx]]})
            exit_drp_spacing.update({self.sink_nodes[idx]: np.array([])})

        self.exit_drps = exit_drps
        self.exit_drp_res_time = exit_drp_res_time
        self.exit_drp_type = exit_drp_type
        self.exit_drp_pstn = exit_drp_pstn
        #self.sink_velocity = sink_velocity
        self.exit_drp_spacing = exit_drp_spacing

        flag_first = 0
        self.entrance_drp_cntr = 0
        self.exit_drp_cntr = 0

        self.branch_num_drps = {}
        self.branch_drp_pstn = {}
        self.branch_drp_type = {}
        self.branch_drp_time = {}
        self.branch_drp_num = {}


        for idx in range(self.num_branches):
            self.branch_num_drps.update({self.branches[idx]: np.zeros(self.droplet_type_count, dtype=int)})
            self.branch_drp_pstn.update({self.branches[idx]: np.array([])})
            self.branch_drp_type.update({self.branches[idx]: np.array([], dtype=int)})
            self.branch_drp_time.update({self.branches[idx]: np.array([])})
            self.branch_drp_num.update({self.branches[idx]: np.array([], dtype=int)})

        self.branch_drp_details = branch_drp_details

        for key, value in self.exit_drps.items():
            for idx in range(len(value)):
                d.update({value[idx]: np.array([])})
        self.d = d

        self.raw_event_sequence.append(self.get_raw_state())

        itr_num = 1
        while self.exit_drp_cntr < sim_drps:
            if (not(self.subdue_print)):
                print('\n')
                print(itr_num)
            self.node_branch_info()
            self.remove_branches()
            #print(self.branches)
            #print(self.bypass_branches)
            #print(self.nodes)
            self.resistance_calc_func()
            #print(self.resistances)
            self.network_solution_calc_func()
            #print(self.Network_solution.flatten())
            for idx in range(len(self.sink_nodes)):
                sink_velocity[self.sink_nodes[idx]] = np.array(abs(self.Network_solution[self.num_branches + self.num_nodes + self.node_indices[self.sink_nodes[idx]]]) / cross_area)
                #print(type(sink_velocity[self.sink_nodes[idx]]))
                #sink_velocity.update({self.sink_nodes[idx]: abs(self.Network_solution[self.num_branches + self.num_nodes + self.node_indices[self.sink_nodes[idx]]]) / cross_area})
            self.sink_velocity = sink_velocity
            self.delt_Tmin_calc(flag_first)
            #print(self.decision)
            #print(self.decision['node_num'][0])
            #print(self.delt_Tmin_val)
            delt_tmin_details.append(np.array(self.delt_Tmin_val))
            self.delt_tmin_details = delt_tmin_details
            if (self.toterminate):
                self.toterminate = False
                return
            flag_first = self.Droplet_Topology_matrix(flag_first)
            self.velocity_calc_func()
            #print(self.Branch_velocity)
            #print(self.exit_drp_spacing)
            #print(self.exit_drp_res_time)
            #print(self.branch_drp_pstn)
            #print(self.branch_drp_type)
            #print(self.branch_drp_num)
            #print(source_spacing)
            #print(source_velocity)
            #print(sink_velocity)
            #print(self.exit_drp_pstn)
            self.itr_num = itr_num
            #print(self.itr_num)
            if self.should_gen_images:
                self.network_structure()
                self.gen_images()
            #print(self.exit_drp_cntr)


            # with open('simulation_file_no_bend.txt', 'a') as f:
            #     f.write('\n ITERATION NUMBER:' + str(itr_num) + '\nNodes:' + str(self.nodes) + '\nBranches:' + str(
            #         self.branches) + '\nBypass Branches:' + str(self.bypass_branches) + '\nentrance drop positn:' + str(self.entrance_drp_pstn) + '\nBRANCH DECISION:' + str(
            #         self.decision) + '\nFlow Rate in Branches:' + str(
            #         self.Network_solution[self.num_nodes:self.num_nodes + self.num_branches, 0]) + '\nTIME:' + str(
            #         self.delt_Tmin_val)
            #             + '\ndrop position:' + str(self.branch_drp_pstn) + '\ndrop number:' + str(
            #         self.branch_drp_num) + '\ndrop branch time:' + str(self.branch_drp_time) + '\ndrop type:' + str(
            #         self.branch_drp_type) + '\nexit drop:' + str(self.exit_drps) + '\nexit drop type:' + str(
            #         self.exit_drp_type)
            #             + '\nexit drop time:' + str(self.exit_drp_res_time) + '\nexit drop spacing:' + str(
            #         self.exit_drp_spacing) + '\nNumber of droplet entered into the network:' + str(
            #         self.entrance_drp_cntr)
            #             + '\nNumber of droplet left the network:' + str(self.exit_drp_cntr) + '\nresistance:' + str(
            #         self.resistances) + '\nPressure at nodes:' + str(
            #         self.Network_solution[:self.num_nodes, 0]) + '\n\n')
            self.event_sequence.append(self.get_new_state())
            self.raw_event_sequence.append(self.get_raw_state())
            itr_num = itr_num + 1

        self.delt_tmin_details = np.delete(self.delt_tmin_details, 0)
        #print(self.delt_tmin_details)
        time_step = 1*np.min(self.delt_tmin_details)
        self.time_step = time_step
        #print(self.time_step)
        #print("drop_resisitance",self.R_d1)
        if (self.should_gen_images):
            self.additional_images()
            #print(self.branch_drp_details)
            print(self.gen_video())
        #print("branch_cnctvty_nodes_cnctd",self.branch_cnctvty_nodes_cnctd)
        #print('incidence matrix',self.incidence_matrix)

        #print("time elapsed: {:.2f}s".format(time.time() - start_time))
        "Fitness Calculation function"    

    def get_new_state(self):
        drop_pos = {0:'entrance',1:'entrance',2:'entrance',3:'entrance'}
        entrance_drops = self.entrance_drps
        exit_drops = self.exit_drps
        branch_drops = self.branch_drp_num
        for ent_idx in entrance_drops:
            for drop_idx in entrance_drops[ent_idx]:
                drop_pos[drop_idx] = 'entrance'
        for branch_idx in branch_drops:
            for drop_idx in branch_drops[branch_idx]:
                drop_pos[drop_idx] = branch_idx
        for exit_idx in exit_drops:
            for drop_idx in exit_drops[exit_idx]:
                drop_pos[drop_idx] = 'Exit'
        return drop_pos

    def get_raw_state(self):
        drop_pos = {}
        entrance_drops = self.entrance_drps
        exit_drops = self.exit_drps
        branch_drops = self.branch_drp_num
        for s_idx, ent_idx in enumerate(entrance_drops):
            for drop_idx in entrance_drops[ent_idx]:
                drop_pos[drop_idx] = -s_idx - 1
        for branch_idx in branch_drops:
            for drop_idx in branch_drops[branch_idx]:
                drop_pos[drop_idx] = branch_idx
        for e_idx, exit_idx in enumerate(exit_drops):
            for drop_idx in exit_drops[exit_idx]:
                drop_pos[drop_idx] = e_idx + self.ex0
        return dict(sorted(drop_pos.items()))

    def network_fitness_calc_fn(self) -> float:
        """
        Determines the fitness value after simulation has finished.
        Criterion for good fit: All droplets of type A in one sink, all droplets of type B in another sink.

        Change to define a new fitness function.
        """
        # The default objective function used is all A drops in top sink & all B drops in the bottom
        err1 = np.sum(self.exit_drp_type[self.sink_nodes[0]]==1)
        #       print('err1',err1)
        #       print('err1',len(Network.exit_drp_type[self.sink_nodes[0]]))
        err2 = np.sum(self.exit_drp_type[self.sink_nodes[1]]==0)
        #       print('err2',err2)
        #       print('err1',len(Network.exit_drp_type[self.sink_nodes[1]]))
        #       err1 : number of drops of type 2 gone into 1st sink
        #       err2 : number of drops of type 1 gone into 2nd sink
        if len(self.exit_drp_type[self.sink_nodes[0]]) > 0 and len(self.exit_drp_type[self.sink_nodes[1]]) > 0:
            fitness1 = (1-err1/len(self.exit_drp_type[self.sink_nodes[0]])) + (1-err2/len(self.exit_drp_type[self.sink_nodes[1]]))
            fitness1 = fitness1/2
            
            fitness2 = (err1/len(self.exit_drp_type[self.sink_nodes[0]])) + (err2/len(self.exit_drp_type[self.sink_nodes[1]]))
            fitness2 = fitness2/2
            
            fitness=max(fitness1,fitness2)
        elif len(self.exit_drp_type[self.sink_nodes[0]]) == 0 or len(self.exit_drp_type[self.sink_nodes[1]]) == 0:
            err00 = np.sum(self.exit_drp_type[self.sink_nodes[0]]==0)
            err01 = np.sum(self.exit_drp_type[self.sink_nodes[0]]==1)
            err10 = np.sum(self.exit_drp_type[self.sink_nodes[1]]==0)
            err11 = np.sum(self.exit_drp_type[self.sink_nodes[1]]==1)
            
            fitness=max(err00,err01,err10,err11)/self.sim_drops
        
            
        else:
            fitness = 0.5
                
        return fitness

    def linked_branches(self, node: int):
        if self.branch_node_list is None:
            self.node_branch_info()
        return [branch for branch in self.branch_node_list[node] if self.var_strng[branch] > 0]

    def EstablishSignConvention(self):
        """
        Establish sign convention for velocities in and out of each node.
        Run during initialisation of simulation environment
        """
        node_branch_mat = self.node_branch_info()
        branch_node_list = self.branch_node_list

        self.sign_convention = {}
        for node, connected_branches in enumerate(branch_node_list):
            node_id = node
            node_sign_convention = {}
            # For each node, find the connected branches
            for branch in connected_branches:
                # Find all nodes connected to this branch.
                nodes_connected_to_this_branch = node_branch_mat[branch]
                # Either the current node is the larger of the two indices, or it is not
                if (node_id >= max(nodes_connected_to_this_branch)):
                    # This would mean that fluid will flow into this node from this branch.
                    node_sign_convention[branch] = -1
                else:
                    # This would mean that fluid will flow out of this node from this branch.
                    node_sign_convention[branch] = 1
            self.sign_convention[node_id] = node_sign_convention