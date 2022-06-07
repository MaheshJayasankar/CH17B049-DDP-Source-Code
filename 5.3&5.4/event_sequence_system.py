# Utility functions for running network simulations

import itertools
import string
from typing import Any
import numpy as np
import copy

from feasibility_simulation_code import Network
from event_sequence_prediction_model.SpacingFeasibilityNew import FeasCheck
from network_simulation_utility import CreateNetwork

class DropType:
	def __init__(self, idx = 0, rr = 1):
		self.idx = idx
		self.rr = rr

class Drop:
	def __init__(self, idx = 0, drop_type = 0):
		self.idx = idx
		self.type = drop_type

class Event:
	def __init__(self, dm_drop: int, from_branch: int, parent_seq: "EventSequence"):
		self.drop = dm_drop
		self.from_branch = from_branch
		self.to_branch = parent_seq.PickFastestBranch(dm_drop)

class EventSequence:
	def __init__(self, network: Network, drop_types: "list[list[int]]"):
		self.network = network
		self.drop_types = drop_types
		self.source_count = len(network.source_nodes)
		self.sink_count = len(network.sink_nodes)
		self.flat_range: "list[int]" = list(itertools.chain(*drop_types))
		self.drop_type_count = self.network.droplet_type_count
		# index of first exit
		self.ex0 = len(network.var_strng)

		self.state: dict[int, int] = {}
		idx = 0
		for (s_idx, s_drops) in enumerate(drop_types):
			for drop in s_drops:
				self.state[idx] = -s_idx - 1
				idx += 1
		
		# Last state for which velocity has been calculated
		self.lv_state: dict[int, int] = None
		self.last_velocity: dict[int, float] = None

		# Last state for which available events are presented
		self.l_ev_state: dict[int, int] = None
		self.available_events = None
		self.GetAvailableEvents()

		self.state_history: list[dict[int, 'int | str']] = []
		self.raw_state_history: list[dict[int, int]] = []

		self.state_history.append(self.Simplify(self.state))
		self.raw_state_history.append(self.state.copy())

		self.feas_checker = FeasCheck(network, drop_types)

	def Simplify(self, state: dict[int,int] = None):
		"""
		Simplify state representation to use in feasibility checking
		"""
		if state == None:
			state = self.state
		simple_state: dict[int, 'int | str'] = state.copy()
		for drop in simple_state:
			if simple_state[drop] < 0:
				simple_state[drop] = 'entrance'
			elif simple_state[drop] >= self.ex0:
				simple_state[drop] = 'Exit'
		return simple_state

	def GetAvailableEvents(self) -> dict[int, Event]:
		state = self.state

		if (self.l_ev_state is state or self.l_ev_state == state):
			return self.available_events

		event_dict = {}
		idx = 0
		for (s_idx, s_drops) in enumerate(self.drop_types):
			for jdx, drop in enumerate(s_drops):
				if state[idx + jdx] >= self.ex0:
					# Skip this drop because it has already exit.
					continue
				new_event = Event(idx + jdx, state[idx + jdx], self)
				event_dict[idx + jdx] = new_event
				if state[idx + jdx] < 0:
					# Skip other drops if this drop is yet to enter. As the other drops cannot enter
					idx += len(s_drops)
					break
				
			# All drops from this source accounted for. Now to next source
			idx += len(s_drops)
		self.available_events = event_dict
		return self.available_events
	
	def AddEventWithDrop(self, dm_drop: int):
		"""
		Retuns:\n
		-1 => Infeasible, so didn't add\n
		0 => Event not available for Droplet\n
		1 => Successfully added
		"""
		available_events = self.GetAvailableEvents()
		if dm_drop not in available_events:
			return 0
		new_state = self.state.copy()
		new_state[dm_drop] = available_events[dm_drop].to_branch
		self.state_history.append(self.Simplify(new_state))
		self.raw_state_history.append(self.state.copy())

		if self.IsFeasible():
			self.state = new_state
			return 1
		else:
			self.state_history.pop()
			return -1

	def IsFeasible(self) -> bool:
		is_feasible, _ = self.feas_checker.check_feasibility(self.state_history)
		return is_feasible
	
	def GetSpacingSolution(self):
		results = self.feas_checker.GetEquations(self.state_history)
		_, res = self.feas_checker.check_feasibility(self.state_history, True)

		A, b = results['A'], results['b']
		A = np.array(A).astype(float)
		b = np.array(b).astype(float).reshape((len(b), 1))

		num_entered = A.shape[1]
		if num_entered == 1:
			raise RuntimeError("At least two droplets should enter the network. Please increase partial sequence length")

		# Calculate spacing from solution
		def expandSolutionIfNotFull(droplet_count, x_sol, A, b, num_entered):
			if num_entered < droplet_count:
				# Need to expand A,
				entry_times = list(x_sol)
				std_droplets = np.std(x_sol)
				while len(entry_times) < droplet_count:
					n_s, n_d = A.shape
					entry_times.append(entry_times[-1] + std_droplets)
					new_A = np.zeros((n_s + 1, n_d + 1))
					new_b = np.zeros((new_A.shape[0], 1))
					# New droplet should enter only after the previous droplet. Hence, t_i - t_(i+1) < 0
					new_A[-1, -2] = 1
					new_A[-1, -1] = -1
					new_A[:-1, :-1] = A
					new_b[:-1, :] = b
					A = new_A
					b = new_b
				actual_entry_times = np.asarray(entry_times).reshape((4,1))
			else:
				actual_entry_times = x_sol.reshape((4,1))
			return actual_entry_times, A, b

		def getSpacingFromTime(actual_entry_times):
			"""
			Function to extract spacing from feasibility solution
			"""
			b = self.network.width_channel
			h = self.network.height_channel
			Q = self.network.Q_in
			beta = self.network.beta
			return (-beta * Q / (b*h) * actual_entry_times).T
		droplet_count = len(self.flat_range)
		actual_entry_times, A, b = expandSolutionIfNotFull(droplet_count, res.x[:num_entered],  A, b, num_entered)
		return getSpacingFromTime(actual_entry_times)
					
	def CalculateVelocity(self, state = None):
		"""
		Calculate and cache velocity
		"""
		if state is None:
			state = self.state
		if self.lv_state is state or self.lv_state == state:
			return self.last_velocity
		def drop_branch_To_branch_num_drops(drop_branch):
			network = self.network
			arrange = self.flat_range
			drop_type_count = self.drop_type_count

			branch_num_drps={}
			for jdx in range(network.num_branches):
				branch_num_drps.update({network.branches[jdx]: np.zeros(drop_type_count, dtype=int)})
			for qdx in range(len(drop_branch)):
				if drop_branch[qdx]>=0 and drop_branch[qdx]< self.ex0:
					branch_num_drps[drop_branch[qdx]][arrange[qdx]] += 1
			return branch_num_drps
		def velocity_calculation(branch_num_drps):
			network = self.network
			network.resistance_calc_func(branch_num_drps)
			network.network_solution_calc_func()
			network.velocity_calc_func()
			k= copy.deepcopy(network.Branch_velocity)
			return k
		self.last_velocity = velocity_calculation(drop_branch_To_branch_num_drops(state))
		self.lv_state = state
		return self.last_velocity

	def PickFastestBranch(self, dm_drop: int, state: dict = None):
		"""
		Uses velocity calculations to find the branch the dm_drop will move towards next.
		This is based on the Maximum Flowrate assumption discussed in the report.
		Return of -1 means that drop has already exit.
		"""
		network = self.network
		if state is None:
			state = self.state
		if (state[dm_drop] >= self.ex0):
			return -1

		dm_drop_position = state[dm_drop]
		if (dm_drop_position < 0):
			branch_nodes = [network.source_nodes[-dm_drop_position - 1], network.source_nodes[-dm_drop_position - 1]]
		else:
			# What are the two nodes at the ends of this branch
			branch_nodes = network.node_branch_info()[dm_drop_position]
		target_branches = {}

		for branch_node in branch_nodes:
			# What all branches can be targetted from this branch, based on branch nodes
			target_branches[branch_node] = network.linked_branches(branch_node)
		
		return self.DropletVelocityArgmax(state, dm_drop_position, target_branches)

	def DropletVelocityArgmax(self, state, dm_drop_position, target_branches) -> int:
		"""
		Internal function used to find branch with maximum velocity
		"""
		target_nodes = list(target_branches.keys())
		vel_dict = self.CalculateVelocity(state)
		sink_nodes = self.network.sink_nodes
		sign_convention = self.network.sign_convention

		# Special case for entrance branch. An entrance branch will have positive velocity
		# Any velocity that is positive is enough
		if (dm_drop_position >= 0):
			cur_branch_vel = vel_dict[dm_drop_position]
		else:
			cur_branch_vel = 100
		
		chosen_branch: int = -1
		if (cur_branch_vel < 0):
			# Case when the droplet is travelling towards the lower node number of the connecting nodes
			chosen_node = min(target_nodes)
			if (chosen_node in sink_nodes):
				# If it is a sink node that was chosen, we must additionally calculate velocity of the outlet tube
				chosen_node_branches = [branch for branch in target_branches[chosen_node]]
				local_branch_vels = {branch: vel_dict[branch] * sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
				# Exit velocity is the of sum of all other velocities
				exit_branch_vel = sum(local_branch_vels.values())

				# Now that we have the exit velocity, we can remove the original branch the droplet is in.
				# Because it cannot move back to that branch
				chosen_node_branches.remove(dm_drop_position)
				del local_branch_vels[dm_drop_position]
				
				exit_branch_idx = self.ex0 + sink_nodes.index(chosen_node)
				local_branch_vels[exit_branch_idx] = exit_branch_vel
				chosen_branch = min(local_branch_vels, key=local_branch_vels.get)

			else:
				# Case when not a sink node
				chosen_node_branches = [branch for branch in target_branches[chosen_node] if branch != dm_drop_position]
				local_branch_vels = {branch: vel_dict[branch] * sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
				chosen_branch = min(local_branch_vels, key=local_branch_vels.get)
		else:
			chosen_node = max(target_nodes)
			if (chosen_node in sink_nodes):
				chosen_node_branches = [branch for branch in target_branches[chosen_node]]
				local_branch_vels = {branch: vel_dict[branch] * sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
				# Exit velocity is the of sum of all other velocities
				exit_branch_vel = -sum(local_branch_vels.values())
				
				# Now that we have the exit velocity, we can remove the original branch the droplet is in. 
				# Because it cannot move back to that branch
				chosen_node_branches.remove(dm_drop_position)
				del local_branch_vels[dm_drop_position]

				exit_branch_idx = self.ex0 + sink_nodes.index(chosen_node)
				local_branch_vels[exit_branch_idx] = exit_branch_vel
				chosen_branch = max(local_branch_vels, key=local_branch_vels.get)
			else:
				chosen_node_branches = [branch for branch in target_branches[chosen_node] if branch != dm_drop_position]
				local_branch_vels = {branch: vel_dict[branch] * sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
				chosen_branch = max(local_branch_vels, key=local_branch_vels.get)
		return chosen_branch

class Environment:
	def __init__(self, drop_list: list[list[int]] = None, desired_outputs: dict[int,int] = None, network_param_dict: dict = None):

		self.params = network_param_dict
		self.network = CreateNetwork(self.params)
		self.created = self.network is not None
		if not self.created:
			return
		if drop_list is None or not isinstance(drop_list, list) or not all([isinstance(item, list) for item in drop_list]):
			# Default: 0 1 1 0
			drop_list = []
			for _ in range(len(self.network.source_nodes)):
				drop_list.append([0, 1, 1, 0])
		num_drop_types = max([max(elist) for elist in drop_list]) + 1
		if num_drop_types > len(self.network.R_d_list):
			raise ValueError(f"Not enough resistances to account for all droplet types. Given {len(self.network.R_d_list)}. Need {num_drop_types}")

		if desired_outputs == None:
			desired_outputs = {}
			for idx in range(len(self.network.sink_nodes)):
				desired_outputs[idx] = idx
		
		self.drops = drop_list

		self.feas_checker = FeasCheck(self.network, self.drops)
		self.event_sequence = EventSequence(self.network, self.drops)
		
		self.exit_pattern = []
		self.offset = len(self.network.source_nodes)
		for drop_type in self.event_sequence.flat_range:
			self.exit_pattern.append(desired_outputs[drop_type])

		self.single_drop_reward = 100
		self.infeasible_reward = -1000

		# 
		self.num_drops = len(self.event_sequence.flat_range)
		self.exit_count = 0

	def get_state_dimensions(self):
		drop_states = len(self.network.source_nodes) + self.event_sequence.ex0 + len(self.network.sink_nodes)
		return tuple([drop_states for idx in range(self.num_drops)])

	def get_action_dimensions(self):
		return (self.num_drops)

	def get_actions(self):
		return list(self.event_sequence.GetAvailableEvents().keys())

	def reset(self):
		self.feas_checker = FeasCheck(self.network, self.drops)
		self.event_sequence = EventSequence(self.network, self.drops)
		self.num_drops = len(self.event_sequence.flat_range)
		self.exit_count = 0
		return self.pack_state(self.event_sequence.state)

	def pack_state(self, state):
		return tuple([pos + self.offset for pos in state.values()])

	def step(self, action) -> tuple[float, tuple[int], bool]:
		# Try to Add Event
		result = self.event_sequence.AddEventWithDrop(action)
		if result <= 0:
			# Failed
			return self.infeasible_reward, self.pack_state(self.event_sequence.state), False
		done = False
		reward = 0
		next_state = self.event_sequence.state
		if result > 0:
			# Check if exit
			if self.event_sequence.state[action] >= self.event_sequence.ex0:
				self.exit_count += 1
				if self.exit_count == self.num_drops:
					done = True
				# Check if correct exit
				if self.event_sequence.state[action] - self.event_sequence.ex0 == self.exit_pattern[action]:
					# correct exit
					reward = self.single_drop_reward
				else:
					reward = -self.single_drop_reward
			return reward, self.pack_state(next_state), done
	
	def GetIntermediateEventSequence(self):
		return self.event_sequence.state_history
	
	def GetSpacingValues(self):
		return self.event_sequence.GetSpacingSolution()

	def GetFitness(self):
		state = self.event_sequence.state
		drop_types = np.hstack(self.event_sequence.drop_types)
		num_sinks = len(self.network.sink_nodes)
		exits = [[] for _ in range(num_sinks)]
		for drop in state:
			cur_exit = state[drop] - self.event_sequence.ex0
			if cur_exit >= 0 and cur_exit < len(exits):
				exits[cur_exit].append(drop_types[drop])
		for idx in range(len(exits)):
			exits[idx] = np.asarray(exits[idx])
		return self.network.GetFitness(exits)
