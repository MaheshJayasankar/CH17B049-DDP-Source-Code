from old_rl_agent_code import NodeClass
import numpy as np
import random
import matplotlib.pyplot as plt
# from cv2 import cv2
import os
from collections import Counter
from SpacingFeasibilityNew import check_feasibility, pick_fastest_branch, get_velocities, set_up_network
from copy import deepcopy
import time
import pickle

grid_dim = 3
source_nodes = [1]
sink_nodes = [10,12]
droplet_count = 4
droplet_type_list = [[1,2,1,1]]
# Sorting problem: the droplet type should pass through which sink node?
target_exit_dict = {1: 0, 2: 1}

# Making a param dict
param_dict = {
    'grid_dim': grid_dim,
    'sim_drops': droplet_count,
    'source_nodes': source_nodes,
    'sink_nodes': sink_nodes,
    'droplet_count': droplet_count,
    'arrange': np.array(droplet_type_list) - 1,
    'rr': 2
}
set_up_network(param_dict)

random.seed(0)
np.random.seed(0)
# ====================================

class Event:
    def __init__(self, droplet_index, event_type, from_branch, to_node):
        self.droplet_index = droplet_index
        self.event_type = event_type
        self.from_branch = from_branch
        self.to_node = to_node
        self.is_done = False
        self.target_branch = from_branch
        self.feasible = True
    
    def SetTargetBranch(self, target_branch):
        self.target_branch = target_branch
        self.is_done = True

    def SetTargets(self, target_node, target_branch):
        self.to_node = target_node
        self.target_branch = target_branch

    def Describe(self):
        print(f"Event of droplet {self.droplet_index}: From branch {self.from_branch} to Node {self.to_node}, Type \"{self.event_type}\"")
        if (self.is_done):
            print(f"Event is done. Moved {self.droplet_index} from branch {self.from_branch} to {self.target_branch}. Is event feasible: {self.feasible}")

# ====================================

class SimulationSystem:
    def __init__(self, grid_dim, source_nodes, sink_nodes, droplet_count, droplet_type_list):
        assert(len(np.asarray(droplet_type_list).flatten()) == droplet_count)
        max_droplet_type = 0
        min_droplet_type = droplet_count
        for source_droplet_type_list in droplet_type_list:
            for droplet_type in source_droplet_type_list:
                if (droplet_type > max_droplet_type):
                    max_droplet_type = droplet_type
                if (droplet_type < min_droplet_type):
                    min_droplet_type = droplet_type
        assert(max_droplet_type <= droplet_count and min_droplet_type > 0)

        self.grid_dim = grid_dim
        self.source_nodes = source_nodes
        self.source_count = len(source_nodes)
        self.sink_nodes = sink_nodes
        self.sink_count = len(sink_nodes)
        self.droplet_type_list = []
        self.droplet_count = droplet_count
        self.nodes = NodeClass.GenerateNetworkNodes(grid_dim)
        self.node_branch_mat = self.nodes[0].node_branch_info()
        self.EstablishSignConvention()

        self.total_branches_count = self.nodes[0].total_branches_count
        self.eendidx = self.total_branches_count + 1

        self.event_list = []
        self.step_count = 0
        self.sink_droplet_types = {}

        self.droplet_positions = []
        self.droplet_target_nodes = {}

        self.explored_branches = []

        for i in range(self.source_count):
            for j in range(len(droplet_type_list[i])):
                self.droplet_positions.append(-self.source_nodes[i])
                self.droplet_type_list.append(droplet_type_list[i][j])

        self.last_event = None

        self.sim_complete = False
        self.results_counted = False

    def EstablishSignConvention(self):
        """
        Establish sign convention for velocities in and out of each node.
        Run during initialisation of simulation environment
        """
        self.sign_convention = {}
        for node in self.nodes:
            node_id = node.node_num
            node_sign_convention = {}
            # For each node, find the connected branches
            connected_branches = node.link_branches
            for branch in connected_branches:
                # Find all nodes connected to this branch.
                nodes_connected_to_this_branch = self.node_branch_mat[branch]
                # Either the current node is the larger of the two indices, or it is not
                if (node_id >= max(nodes_connected_to_this_branch)):
                    # This would mean that fluid will flow into this node from this branch.
                    node_sign_convention[branch] = -1
                else:
                    # This would mean that fluid will flow out of this node from this branch.
                    node_sign_convention[branch] = 1
            self.sign_convention[node_id] = node_sign_convention

    def SimDropletPos2FeasCheckDropletPos(self, sim_droplet_pos):
        """
        Conversion of droplet positions dict into format accepted by the feasibility checker. Back-conversion not possible.
        """
        feascheck_droplet_pos = {}
        for drop in sim_droplet_pos:
            if sim_droplet_pos[drop] < 0:
                feascheck_droplet_pos[drop] = 'entrance'
            elif sim_droplet_pos[drop] >= self.total_branches_count:
                feascheck_droplet_pos[drop] = 'Exit'
            else:
                feascheck_droplet_pos[drop] = sim_droplet_pos[drop]
        
        return feascheck_droplet_pos

    def GetAvailableEvents(self, droplet_positions = None):
        
        """
        Get Available Events given droplet positions
        """

        event_dict = {}

        if (droplet_positions is None):
            droplet_positions = self.droplet_positions

        # Start from the smallest droplet index. It is assumed that smallest droplet index will enter first

        for idx in range(self.droplet_count):
            cur_drop_pos = droplet_positions[idx]
            
            # If droplet has already exit
            if (cur_drop_pos >= self.total_branches_count):
                continue

            # Check if it is an entry event
            elif (cur_drop_pos < 0):
                source_node = -cur_drop_pos
                new_event = Event(idx, 'entry', -source_node, source_node)
                event_dict[idx] = new_event
                # Since this droplet is yet to enter, the following droplets can't enter yet. Therefore break the algorithm
                break

            else:
                # Otherwise, it is tagged as a travel event. We will later confirm whether it is of type exit or travel.
                # Target cannot be chosen yet.
                chosen_target = -1
                new_event = Event(idx, 'travel', cur_drop_pos, chosen_target)
                event_dict[idx]= new_event
        
        return event_dict

    def PickFastestBranch(self, dm_drop, droplet_positions):
        """
        Uses velocity calculations to find the branch the dm_drop will move towards next
        """
        if (len(source_nodes) != 1):
            raise NotImplementedError("Only 1 Source Node is supported.")
        if (droplet_positions[dm_drop] == 'Exit'):
            return None

        dm_drop_position = droplet_positions[dm_drop]
        if (dm_drop_position == 'entrance'):
            branch_nodes = [source_nodes[0], source_nodes[0]]
        else:
            branch_nodes = self.node_branch_mat[dm_drop_position]
        target_branches = {}

        for branch_node in branch_nodes:
            target_branches[branch_node] = self.nodes[branch_node].link_branches
        
        return self.DropletVelocityArgmax(droplet_positions, dm_drop_position, target_branches)

    def DropletVelocityArgmax(self, droplet_positions, dm_drop_position, target_branches):
        """
        Internal function used to find branch with maximum velocity
        """
        target_nodes = list(target_branches.keys())
        vel_dict = get_velocities(droplet_positions)

        # Special case for entrance branch. An entrance branch will have arbitrarily high velocity
        if (dm_drop_position != 'entrance'):
            cur_branch_vel = vel_dict[dm_drop_position]
        else:
            cur_branch_vel = 100
        
        if (cur_branch_vel < 0):
            # Case when the droplet is travelling towards the lower node number of the connecting nodes
            chosen_node = min(target_nodes)
            if (chosen_node in sink_nodes):
                # If it is a sink node that was chosen, we must additionally calculate velocity of the outlet tube
                chosen_node_branches = [branch for branch in target_branches[chosen_node]]
                local_branch_vels = {branch: vel_dict[branch] * self.sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
                # Exit velocity is the of sum of all other velocities
                exit_branch_vel = sum(local_branch_vels.values())
                
                # REMOVED:
                # Now that we have the exit velocity, we can remove the original branch the droplet is in. Because it cannot move back to that branch
                chosen_node_branches.remove(dm_drop_position)
                del local_branch_vels[dm_drop_position]
                # chosen_node_branches = [branch for branch in chosen_node_branches if branch != dm_drop_position]
                # local_branch_vels = {branch: vel_dict[branch] * self.sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
                # ===

                exit_branch_idx = self.total_branches_count + sink_nodes.index(chosen_node)
                local_branch_vels[exit_branch_idx] = exit_branch_vel
                chosen_branch = min(local_branch_vels, key=local_branch_vels.get)

            else:
                # Case when not a sink node
                chosen_node_branches = [branch for branch in target_branches[chosen_node] if branch != dm_drop_position]
                local_branch_vels = {branch: vel_dict[branch] * self.sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
                chosen_branch = min(local_branch_vels, key=local_branch_vels.get)
        else:
            chosen_node = max(target_nodes)
            if (chosen_node in sink_nodes):
                chosen_node_branches = [branch for branch in target_branches[chosen_node]]
                local_branch_vels = {branch: vel_dict[branch] * self.sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
                # Exit velocity is the of sum of all other velocities
                exit_branch_vel = -sum(local_branch_vels.values())
                
                # REMOVED:
                # Now that we have the exit velocity, we can remove the original branch the droplet is in. Because it cannot move back to that branch
                chosen_node_branches.remove(dm_drop_position)
                del local_branch_vels[dm_drop_position]
                # chosen_node_branches = [branch for branch in chosen_node_branches if branch != dm_drop_position]
                # local_branch_vels = {branch: vel_dict[branch] * self.sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
                # ===

                exit_branch_idx = self.total_branches_count + sink_nodes.index(chosen_node)
                local_branch_vels[exit_branch_idx] = exit_branch_vel
                chosen_branch = max(local_branch_vels, key=local_branch_vels.get)
            else:
                chosen_node_branches = [branch for branch in target_branches[chosen_node] if branch != dm_drop_position]
                local_branch_vels = {branch: vel_dict[branch] * self.sign_convention[chosen_node][branch] for branch in vel_dict if branch in chosen_node_branches}
                chosen_branch = max(local_branch_vels, key=local_branch_vels.get)
        return chosen_branch, chosen_node

    def Describe(self):
        print(f"System of grid dimension {self.grid_dim}")
        print(f"Currently has performed {len(self.event_list)} events.")
        print("-----------------------------------")
        print(f"Source Nodes: {source_nodes}, Sink Nodes:{sink_nodes}")
        print("Droplet Positions:")
        for i in range(self.droplet_count):
            print(f"Droplet {i}: At branch {self.droplet_positions[i]}")
        
        if (self.sim_complete):
            print(f"Simulation finished in {self.step_count} steps.")
            print("Purity Score:")
            print(self.GetPurityScores())
            print("Final droplet tally:")
            print(self.sink_droplet_types)
            
    def CountSimResults(self):
        for sink_idx,sink_node in enumerate(sink_nodes):
            droplet_indices = [i for i, x in enumerate(self.droplet_positions) if x == self.eendidx + sink_idx]
            self.sink_droplet_types[sink_node] = list(np.array(self.droplet_type_list)[droplet_indices])
        
        self.results_counted = True

    def GetPurityScores(self):
        if (self.results_counted == False):
            self.CountSimResults()
            self.results_counted = True
    
        total_score = 0
        for sink_node in sink_nodes:
            current_sink_droplets = self.sink_droplet_types[sink_node]
            if (len(current_sink_droplets) <= 0):
                purity_score = 0
            else:
                droplet_counter = Counter(current_sink_droplets)
                max_occurance_count = max(droplet_counter.values())
                purity_score = max_occurance_count / len(current_sink_droplets)
            total_score += purity_score
        return total_score / len(sink_nodes)

    def PlotCurrentState(self):
        coords = []
        for node in self.nodes:
            coords.append(node.coord)

        source_length = 1

        start_coord = self.nodes[self.source_nodes[0]].coord
        pre_source_coord = (start_coord[0] - source_length, start_coord[1])

        def split_x_y_coords(pt1, pt2):
            return [pt1[0], pt2[0]], [pt1[1], pt2[1]]

        def drawline(pt1, pt2, colour = 'g'):
            x_coords, y_coords = split_x_y_coords(pt1,pt2)
            plt.plot(x_coords, y_coords, linewidth = 6, c = colour)

        def drawmidpt(pt1, pt2, colour = 'r'):
            x_coords, y_coords = split_x_y_coords(pt1,pt2)
            mid_x = np.mean(x_coords)
            mid_y = np.mean(y_coords)
            plt.scatter(mid_x, mid_y, marker = 's', s=200, c = colour)

        droplet_cmap = {
            1: 'black',
            2: 'r'
        }

        newfig = plt.figure(figsize = (12.5,7.5))
        plt.scatter(*zip(*coords), marker = 's', s=400)
        plt.axis('off')
        drawline(start_coord, pre_source_coord, 'g')
        
        for sink_node in sink_nodes:
            exit_coord = self.nodes[sink_node].coord
            post_exit_coord = (exit_coord[0] + source_length, exit_coord[1])
            colour = 'g'
            drawline(exit_coord, post_exit_coord, colour)

        node_branch_mat = self.nodes[0].node_branch_info()

        for branch in self.explored_branches:
            nodes_of_branch = node_branch_mat[branch]
            pt1 = self.nodes[nodes_of_branch[0]].coord
            pt2 = self.nodes[nodes_of_branch[1]].coord
            drawline(pt1, pt2, 'b')

        for idx,droplet_branch in enumerate(self.droplet_positions):
            if (droplet_branch < 0 or droplet_branch >= self.total_branches_count):
                continue
            nodes_of_branch = node_branch_mat[droplet_branch]
            pt1 = self.nodes[nodes_of_branch[0]].coord
            pt2 = self.nodes[nodes_of_branch[1]].coord
            droplet_type = self.droplet_type_list[idx]            
            colour = droplet_cmap[droplet_type]
            drawmidpt(pt1, pt2, colour)

        # if current event was an exit event, display the exiting droplet
        if (self.last_event.event_type == 'exit'):
            exit_coord = self.nodes[self.last_event.to_node].coord
            post_exit_coord = (exit_coord[0] + source_length, exit_coord[1])
            colour = droplet_cmap[self.droplet_type_list[self.last_event.droplet_index]]
            drawmidpt(exit_coord, post_exit_coord, colour)


        plt.savefig('node_progress/{}.png'.format(self.step_count))
        plt.close(newfig)

    def GenerateVideo(self):
        # print('Rendering Video:')
        # imgDir = 'node_progress/'
        # img_array = []
        # for i in range(1,self.step_count + 1):
        #     img = cv2.imread(imgDir + '{}.png'.format(i))
        #     height, width, _ = img.shape
        #     size = (width, height)
        #     img_array.append(img)

        # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()
        pass

# ====================================

class QLearningAgent:
    def __init__(self, sim_env: SimulationSystem, droplet_type_list, target_exit_dict, epsilon = 0.1, learning_rate = 0.01, discount_factor = 0.9):
        self.grid_dim = sim_env.grid_dim
        self.source_nodes = sim_env.source_nodes
        self.source_nodes = sim_env.source_nodes
        self.source_count = sim_env.source_count
        self.sink_nodes = sim_env.sink_nodes
        self.sink_count = sim_env.sink_count
        self.droplet_type_list = sim_env.droplet_type_list
        self.droplet_count = sim_env.droplet_count
        self.node_count = len(sim_env.nodes)
        self.nodes = sim_env.nodes
        self.total_branches_count = sim_env.total_branches_count

        self.Q_values = np.zeros((self.droplet_count, self.node_count, 8))
        self.epsilon = epsilon
        self.alpha = learning_rate
        self.gamma = discount_factor

        self.feas_fun_values = []

        self.reward = 100
        self.infeasible_reward = -1000

        self.SetExitTargets(droplet_type_list, target_exit_dict)

    def TDUpdate(self, state, action, reward, next_state):
        old_q_value = self.Q_values[state[0], state[1], action]

        if (next_state[1] >= self.node_count):
            next_q_values = old_q_value * 0

        else:
            next_q_values = self.Q_values[next_state]

        max_q = np.max(next_q_values)

        new_q_value = reward + self.gamma * max_q

        self.Q_values[state[0], state[1], action] = old_q_value + self.alpha * (new_q_value - old_q_value)

    def SetExitTargets(self, droplet_type_list, target_exit_dict):
        """
        Calculates where each droplet index should end up ideally speaking. Run before learning begins
        """
        self.exit_targets = []
        for idx in range(len(droplet_type_list[0])):
            self.exit_targets.append(self.sink_nodes[target_exit_dict[droplet_type_list[0][idx]]])

# ====================================

class DQNAgent(QLearningAgent):
    def __init__(self, sim_sys: SimulationSystem, droplet_type_list, target_exit_dict, epsilon = 0.2, learning_rate = 0.01, discount_factor = 0.9, epsilon_annealing = False, eps_anl_half_life = 10000):
        super().__init__(sim_sys, droplet_type_list, target_exit_dict, epsilon, learning_rate, discount_factor)

        # Ensure that this method cannot be used outside the scope for which it was defined
        if (self.droplet_count != 4 or self.grid_dim != 3 or self.source_count != 1 or self.sink_count != 2):
            raise NotImplementedError("Unsupported input format.")
        
        self.droplet_positions = {}
        self.droplet_int_positions = {}
        self.droplet_positions_history = []
        self.droplet_exit_nodes = {}
        self.droplet_exit_count = 0
        self.droplet_exit_purity = 0

        for idx in range(sim_sys.droplet_count):
            self.droplet_positions[idx] = 'entrance'

        n = self.total_branches_count + 3
        d = self.droplet_count

        self.step_count = 0
        self.should_decay = epsilon_annealing
        if (self.should_decay):
            self.decay_constant = 2 ** (-1 / eps_anl_half_life)
        
        self.entry_flags = 0

        self.DQ_values = np.zeros((n, n, n, n, d))

    def TDUpdate(self, state, action, reward, next_state):
        old_q_value = self.DQ_values[state[0], state[1], state[2], state[3], action]

        next_q_values = self.DQ_values[next_state[0], next_state[1], next_state[2], next_state[3], :]

        max_q = np.max(next_q_values)

        new_q_value = reward + self.gamma * max_q

        self.DQ_values[state[0], state[1], state[2], state[3], action] = old_q_value + self.alpha * (new_q_value - old_q_value)

    def DropletPos2StateSpace(self, droplet_positions):
        """
        State space representation used internally during Q Learning Updates. Not for representing simulation state
        """
        # State space transformation: Branch i is corresponds to state i + 1.
        #                             Entrance is corresponds to state 0
        #                             Exit corresponds to state 29

        drop_positions_integrised = []

        for drop_idx in range(self.droplet_count):
            drop_idx_pos = droplet_positions[drop_idx]

            if (drop_idx_pos == 'entrance'):
                drop_positions_integrised.append(0)
            elif (drop_idx_pos == 'Exit'):
                drop_positions_integrised.append(self.total_branches_count + 1)
            else:
                drop_positions_integrised.append(drop_idx_pos + 1)

        state = np.array(drop_positions_integrised)
        return state

    def Update(self, feascheck_result, dm_drop, dm_drop_exit_id, prev_drop_positions, new_drop_positions):
        # Update method handles the reward-giving and value updation of the Q-table

        # Ensure that this method cannot be used outside the scope for which it was defined

        prev_state = self.DropletPos2StateSpace(prev_drop_positions)
        next_state = self.DropletPos2StateSpace(new_drop_positions)

        if (not(feascheck_result)):
            reward = self.infeasible_reward
        else:
            reward = self.reward * dm_drop_exit_id
        
        self.TDUpdate(prev_state, dm_drop, reward, next_state)

    def Step(self, sim_env, disable_update = False):
        done = False
        self.step_count += 1
        
        prev_drop_positions = self.droplet_positions

        # Pick optimal, yet feasible next event as per algorithm
        next_event = self.PickDmDropEvent(sim_env)
        
        if (not next_event):
            return True, False

        dm_drop, dm_drop_target_node, dm_drop_new_branch = next_event.droplet_index, next_event.to_node, next_event.target_branch
        
        dm_drop_exit_id = 0
        # Set in-class variables to values used in next iteration
        if dm_drop_new_branch >= self.total_branches_count:
            # Droplet has exit the simulation
            self.droplet_exit_count += 1
            self.droplet_exit_nodes[dm_drop] = dm_drop_target_node
            # Did it leave from the correct sink?
            if (dm_drop_target_node == self.exit_targets[dm_drop]):
                dm_drop_exit_id = 1
                self.droplet_exit_purity += 1
            else:
                dm_drop_exit_id = -1
                self.droplet_exit_purity -= 1
            if (self.droplet_exit_count >= sim_env.droplet_count):
                done = True
            dm_drop_new_position = 'Exit'
        else:
            dm_drop_new_position = dm_drop_new_branch
        
        self.droplet_int_positions[dm_drop] = dm_drop_new_branch
        self.droplet_positions = self.droplet_positions.copy()
        self.droplet_positions[dm_drop] = dm_drop_new_position
        self.droplet_positions_history.append(self.droplet_positions)


        new_drop_positions = self.droplet_positions
        if (not(disable_update)):
            self.Update(True, dm_drop, dm_drop_exit_id, prev_drop_positions, new_drop_positions)

        return done, True         

    def PickDmDropEvent(self, sim_env):
        # Find out which droplets are able to move:
        av_event_dict = sim_env.GetAvailableEvents(self.droplet_int_positions)

        # Only the droplets that are available can be picked
        if (len(av_event_dict) <= 0):
            raise ValueError("No available droplets, but agent has been asked to perform action.")
            # return None

        pick_randomly = False
        # Random chance for exploration
        if np.random.rand() < self.epsilon:
            # Exploration
            pick_randomly = True

        # Keep attempting to find a suitable event, until a feasible one is found.
        is_feasible = False

        while not is_feasible:
            if (len(av_event_dict) <= 0):
                f = open("infeasible_log.txt", "a")
                f.write(str(self.droplet_positions_history))
                f.write("\n")
                f.close()
                # raise ValueError("No more remaining feasible events. Check feasibility function")
                return None
        # Calculate target branch of decision making drop
            if pick_randomly:
                dm_drop = np.random.choice(list(av_event_dict.keys()))
            else:
                # Pick the next branch using argmax of q-values
                # Get the current state
                current_state = self.DropletPos2StateSpace(self.droplet_positions)

                # current_q_values = self.DQ_values[current_state[0], current_state[1], current_state[2], current_state[3], :]
                # q_value_dict = {idx: q_value for idx, q_value in enumerate(current_q_values) if idx in av_event_dict}

                # if (len(q_value_dict) == 0):
                #     pick_randomly = True
                #     dm_drop = np.random.choice(list(av_event_dict.keys()))
                # else:
                #     maxq = max(q_value_dict.values())
                #     max_indices = [idx for idx, q_val in q_value_dict.items() if q_val >= maxq]
                #     if (len(max_indices) == 1):
                #         dm_drop = max_indices[0]
                #     else:
                #         dm_drop = np.random.choice(max_indices)
                
                dm_drop = self.ArgmaxQValues(current_state, list(av_event_dict.keys()))
                
            dm_drop_new_branch, dm_drop_target_node = sim_env.PickFastestBranch(dm_drop, self.droplet_positions)

            droplet_positions_history_copy = deepcopy(self.droplet_positions_history)
            droplet_positions_copy = deepcopy(self.droplet_positions)

            if (dm_drop_new_branch >= self.total_branches_count):
                droplet_positions_copy[dm_drop] = 'Exit'
            else:
                droplet_positions_copy[dm_drop] = dm_drop_new_branch

            droplet_positions_history_copy.append(droplet_positions_copy)
            is_feasible, _ = CheckFeasibility(droplet_positions_history_copy, dm_drop)
            if (not is_feasible):
                # High Penalty given to the action to prevent action being taken in future
                self.Update(False, dm_drop, 0, self.droplet_positions, droplet_positions_copy)
            event_type = av_event_dict[dm_drop]
            del av_event_dict[dm_drop]

        new_event = Event(dm_drop, event_type, self.droplet_int_positions[dm_drop], dm_drop_target_node)
        new_event.SetTargetBranch(dm_drop_new_branch)

        return new_event

    def ArgmaxQValues(self, current_state, valid_action_list = None):
        current_q_values = self.DQ_values[current_state[0], current_state[1], current_state[2], current_state[3], :]
        if (valid_action_list is None):
            q_value_dict = {idx: q_value for idx, q_value in enumerate(current_q_values)}
        else:
            q_value_dict = {idx: q_value for idx, q_value in enumerate(current_q_values) if idx in valid_action_list}
        
        maxq = max(q_value_dict.values())
        max_indices = [idx for idx, q_val in q_value_dict.items() if q_val >= maxq]
        if (len(max_indices) == 1):
            dm_drop = max_indices[0]
        else:
            dm_drop = np.random.choice(max_indices)
        return dm_drop

    def ResetState(self, sim_sys):
        self.droplet_positions = {}
        self.droplet_int_positions = {}
        self.droplet_positions_history = []
        self.droplet_exit_nodes = {}
        self.droplet_exit_count = 0
        self.droplet_exit_purity = 0

        # All droplets start at the first (and only) source node
        for idx in range(sim_sys.droplet_count):
            self.droplet_int_positions[idx] = -self.source_nodes[0]
            self.droplet_positions[idx] = 'entrance'
        self.droplet_positions_history.append(self.droplet_positions)
        
        if (self.should_decay):
            self.epsilon *= self.decay_constant
        

# ====================================

def ConvertEventList(event_list):
    # Convert event list into target format
    drop_position={0: "entrance", 1: "entrance", 2: "entrance", 3: "entrance"}
    converted_event_list = []

    if (len(source_nodes) != 1):
        raise NotImplementedError("Feasibilty check failed. Can't support multiple entrances.")

    if (len(event_list) <= 0):
        return True, 0

    # Move drops based on each event
    for event in event_list:
        if (event.event_type == 'end_sim'):
            print(f"Full Sim Feasibility Check Completed.")
            return True
        if (event.event_type == 'exit'):
            drop_position[event.droplet_index] = "Exit"
        else:
            drop_position[event.droplet_index] = event.target_branch
        converted_event_list.append(drop_position.copy())
    return converted_event_list

def CheckFeasibility(droplet_positions_history, dm_drop = -1):
    """
    If dm_drop was specified, can additionally check for edge cases (entrance drop always feasible)
    """
    if (dm_drop != -1):
        if (droplet_positions_history[-2][dm_drop] == 'entrance'):
            return True, 1e-9
    return check_feasibility(droplet_positions_history)

def NewRunTest(seed_list, iter_count = 500, modify = False, new_eps = 1, new_eps_anl = False):
    
    sim_sys = SimulationSystem(grid_dim, source_nodes, sink_nodes, droplet_count, droplet_type_list)

    np.random.seed(0)
    random.seed(0)

    # Runs used
    seeds = seed_list
    run_count = len(seed_list)

    ov_max_fitness = []
    ov_fitness_vals = []
    export_dict = {

    }
    ov_repitition_rate = []
    ov_repeat_count = []

    toe_list = []

    for run_number in seeds:
        np.random.seed(run_number)
        random.seed(run_number)
        st_time = time.time()
        if (not modify):
            rl_agent = DQNAgent(sim_sys, droplet_type_list, target_exit_dict, epsilon= 0.6, epsilon_annealing= True, eps_anl_half_life= 125)
        else:
            rl_agent = DQNAgent(sim_sys, droplet_type_list, target_exit_dict, epsilon= new_eps, epsilon_annealing= new_eps_anl, eps_anl_half_life= 125)
        prev_100_iter_q_table = rl_agent.DQ_values.copy()

        max_fitness = -1
        fitness_vals = []
        avg_fitness_vals = []
        max_fitness_vals = []
        repitition_rate = []
        repeat_count = 0


        for iter_number in range(iter_count):
            # Perform an iteration (generate event sequence)
            rl_agent.ResetState(sim_sys)
            if (iter_number % 100 == 0):
                diff_norm = np.linalg.norm(rl_agent.DQ_values.flatten() - prev_100_iter_q_table.flatten()) / (np.linalg.norm(prev_100_iter_q_table.flatten()) + 1e-9)
                prev_100_iter_q_table = rl_agent.DQ_values.copy()
                print(f"Seed {run_number}: {iter_number} sequences completed.\nNorm difference: {diff_norm * 100:.2f}%, Epsilon value: {rl_agent.epsilon :.3f}\nMaximum Fitness so far: {max_fitness}\n")
            # New instantiation of environment
            sim_sys = SimulationSystem(grid_dim, source_nodes, sink_nodes, droplet_count, droplet_type_list)

            done = False
            while not done:
                done, status = rl_agent.Step(sim_sys)
            
            if not status:
                print(f"Iteration {iter_number} completed unsuccessfully.")
            fitness = rl_agent.droplet_exit_purity / 4
            max_fitness = max(max_fitness, fitness)
            fitness_vals.append(fitness)
            avg_fitness_vals.append(np.mean(fitness_vals))
            max_fitness_vals.append(max_fitness)
            
            current_rep_rate = repeat_count / (iter_number + 1)
            repitition_rate.append(current_rep_rate)

        ed_time = time.time()
        toe_time = ed_time - st_time

        toe_list.append(toe_time)

        print(f"Seed {run_number} completed. Maximum Fitness: {max_fitness}")
        eta_time = (run_count - run_number - 1) * sum(toe_list) / len(toe_list)
        print(f"Time taken for run: {toe_time:.4f} sec.\nETA: {eta_time:.4f} secs")
        ov_max_fitness.append(max_fitness)
        ov_fitness_vals.append(fitness_vals)
        export_dict[run_number] = {
            'avg': avg_fitness_vals,
            'max': max_fitness_vals
        }
        ov_repitition_rate.append(repitition_rate)
        ov_repeat_count.append(repeat_count)
    
    print("\n=========================\n")

    return export_dict