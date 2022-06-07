# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:11:10 2020

@author: arune
"""

# Modified for use in Dual Degree Project (2021-2022) by J Mahesh, CH17B049.

import numpy as np

class Network:
	def __init__(self,
				 grid_dim: int, var_strng: "list[int]", source_nodes: "list[int]", sink_nodes: "list[int]",
				 length_factor: float, Q_in: float, P_out: float, width_channel: float,height_channel: float,
				 beta: float,rr_list: "list[float]",vscty: float, rn1: int=1, rn2: int=21, rn3: int=321):
		
		# 450micro litre/hr is the flow rate
		
		self.grid_dim = grid_dim
		self.var_strng= var_strng
		self.source_nodes = source_nodes
		self.sink_nodes = sink_nodes
		self.length_factor = length_factor
		self.Q_in = Q_in
		self.P_out = P_out
		self.width_channel = width_channel
		self.height_channel = height_channel
		self.vscty = vscty
		srs_prt=0
		for n_srsindx in range(1, 100, 2):
			srs_prt = srs_prt + (1 / n_srsindx ** 5) * (192 / np.pi ** 5) * (self.height_channel) / (self.width_channel) * np.tanh((n_srsindx * np.pi * width_channel) / (2 * height_channel))
		self.srs_prt=srs_prt
		self.R_d1 = (12 * self.vscty * self.length_factor / ((self.width_channel) * self.height_channel ** 3)) * (1 - self.srs_prt) ** -1# 3.15*1.5*interfacial_tension*((self.vscty* self.Q_in/(self.width_channel* self.height_channel*interfacial_tension))**(2/3))/(self.Q_in*self.width_channel)
		
		self.droplet_type_count = len(rr_list)
		self.beta = beta

		self.R_d_list = [self.R_d1]
		for i in range(1,self.droplet_type_count):
			self.R_d_list.append(self.R_d1 * rr_list[i])

		self.rn1=rn1
		self.rn2=rn2
		self.rn3=rn3

		self.branch_lengths_found = False
		self.A = None
		self.branch_node_list = None
		self.node_branch_info()
		self.sign_convention: dict[int, int] = None
		self.EstablishSignConvention()

	def linked_branches(self, node: int):
		if self.branch_node_list == None:
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

	def node_branch_info(self):
		"""
		The function returns a matrix(A) of dim (total_num_branches,2)
		A(i,0) & A(i,1) give the two nodes between which the branch exists
		"""

		if self.A is not None:
			return self.A

		grid_dim = self.grid_dim

		total_branches = (grid_dim - 1) * (6 * grid_dim - 4)  # Total number of branches for a given grid_dim
		intersection_node_matrix = np.zeros((grid_dim - 1,grid_dim - 1), dtype=int)
		for idx in range(grid_dim - 1):
			for jdx in range(grid_dim - 1):
				intersection_node_matrix[idx][jdx] = idx + grid_dim + jdx*(2*grid_dim - 1)
		A = np.zeros((total_branches, 2), dtype=int)
		max_nodes = (2 * grid_dim - 1) * (grid_dim - 1) + grid_dim
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
		B = [[] for node in range(max_nodes)]
		for idx in range(A.shape[0]):
			for jdx in range(A.shape[1]):
				B[A[idx,jdx]].append(idx)
		self.branch_node_list = B
		return A

	def node_connector_matrix(self, var_strng):
		"""adjacency matrix
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
		y = A.tolist()
		#print(y)
		#A[A==2] = 1
		#print(A)
		return A

	def calc_length_branches(self):

		true_random_state = np.random.get_state()

		if self.branch_lengths_found:
			return self.length_branches
		for_struct = []
		P=[]
		Q=[]
		node_cordinate={}
		L=(1/1000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000)), int(self.length_factor*(1000)))
		#print(L)
		np.random.seed(self.rn1)
		#imp_node1 = np.multiply(L,(1.01-0.99)*np.random.random_sample(3)+0.99)#np.multiply(L, np.array(([0.99999972, 0.99999775, 0.99999615]))) #0.99999546, 0.99999864])))
		#G=np.multiply(((1/1000000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000000)), int(self.length_factor*(1000000)))),imp_node1)
		G=np.multiply(((1/1000000)*np.arange(0, self.grid_dim * int(self.length_factor*(1000000)), int(self.length_factor*(1000000)))),(1.0005-0.9995)*np.random.random_sample(3)+0.9995)#,0.99999546, 0.99999864])));
		#print(G)
		np.random.seed(self.rn2)
		M=np.multiply(((1/1000000)*np.arange((self.grid_dim - 1)* int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)),-int(self.length_factor*(1000000)))),(1.0005-0.9995)*np.random.random_sample(3)+0.9995)#, 1.00000546, 1.00000348])))
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
		self.length_branches = length_branches

		np.random.set_state(true_random_state)

		return length_branches
	
	def remove_branches(self):

		A = self.node_branch_info()
		branches = np.flatnonzero(np.array(self.var_strng))
		bypass_branches = np.where(np.array(self.var_strng) == 2)[0]
		num_branches = len(branches)
		nodes = np.unique(A[branches, :])
		num_nodes = len(nodes)
		node_indices = {}
		branch_indices = {}
		length_branches = self.calc_length_branches()
		# print(length_branches[branches])
		for idx in range(num_nodes):
			node_indices.update({nodes[idx]: idx})
		for jdx in range(num_branches):
			branch_indices.update(({branches[jdx]: jdx}))
			
		self.num_branches=num_branches
		self.branches = branches
		self.bypass_branches = bypass_branches
		self.nodes = nodes
		self.num_branches = num_branches
		self.num_nodes = num_nodes
		self.node_indices = node_indices
		self.branch_indices = branch_indices
		# Need to recalculate branch lengths
		self.branch_lengths_found = False
		
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
		D_matrix = np.zeros((dim_D, num_nodes))
		C_matrix = np.zeros((num_nodes - dim_Pbc, num_nodes))
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
	   # print('self.incidence_matrix',self.incidence_matrix)
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
		#print("i see")
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
		# print('dsagfd',self.branch_cnctvty_branchs_cnctd)
		return    

		
	def resistance_calc_func(self,branch_num_drps):
		"""
		Calculate the resistance in each branch, using positions of all droplets.
		"""

		# Number of droplet types need not be equal to number of resistance ratios
		# Need to externally ensure that index error doesn't happen

		#print("resistance_calc_func-block runs")
		width_channel = self.width_channel
		w2 = self.width_channel/2
		height_channel = self.height_channel
		vscty = self.vscty
		cross_area = width_channel * height_channel
		srs_prt = 0
		srs_prt1 = 0
		length_branches = self.calc_length_branches()
		#print(length_branches)

		for n_srsindx in range(1, 100, 2):
			srs_prt = srs_prt + (1 / n_srsindx ** 5) * (192 / np.pi ** 5) * (height_channel) / (
				width_channel) * np.tanh((n_srsindx * np.pi * width_channel) / (2 * height_channel))
			srs_prt1 = srs_prt1 + (1 / n_srsindx ** 5) * (192 / np.pi ** 5) * (height_channel) / (w2) * np.tanh((n_srsindx * np.pi * w2) / (2 * height_channel))
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
		
		cross_area = self.width_channel * self.height_channel
		Branch_velocity = {}

		for idx in range(self.num_branches):
			Branch_velocity.update({self.branches[idx]: self.Network_solution[self.num_nodes + idx, 0] / cross_area})
		self.Branch_velocity = Branch_velocity

		#print(self.branch_cnctvty_nodes_cnctd)

		#print(Branch_velocity)
		return
	def Simlulate_fn(self):
		self.node_branch_info()
		self.node_connector_matrix(self.var_strng)
		self.calc_length_branches()
		self.remove_branches()
		self.calc_length_branches()
		self.incidence_matrix_gen()
		self.branch_connectivity_fn()
		self.network_solution_calc_func()
		self.velocity_calc_func()

	def GetFitness(self, exit_pattern):
		err1 = np.sum(exit_pattern[0]==1)
		err2 = np.sum(exit_pattern[1]==0)
		if len(exit_pattern[0]) > 0 and len(exit_pattern[1]) > 0:
			fitness1 = (1-err1/len(exit_pattern[0])) + (1-err2/len(exit_pattern[1]))
			fitness1 = fitness1/2
			fitness2 = (err1/len(exit_pattern[0])) + (err2/len(exit_pattern[1]))
			fitness2 = fitness2/2			
			fitness=max(fitness1,fitness2)
		elif len(exit_pattern[0]) == 0 or len(exit_pattern[1]) == 0:
			err00 = np.sum(exit_pattern[0]==0)
			err01 = np.sum(exit_pattern[0]==1)
			err10 = np.sum(exit_pattern[1]==0)
			err11 = np.sum(exit_pattern[1]==1)
			fitness=max(err00,err01,err10,err11)/len(exit_pattern)
		else:
			fitness = 0.5
		return fitness
