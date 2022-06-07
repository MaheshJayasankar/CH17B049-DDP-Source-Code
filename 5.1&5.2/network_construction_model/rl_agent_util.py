import numpy as np
import matplotlib.pyplot as plt

from numpy.lib.function_base import append

# Written for use in Dual Degree Project, 2021-2022 by J Mahesh, CH17B049, Dept. Chemical Engineering, IIT Madras

class NodeClass:
    """
    This class will help with finding the neighbouring nodes & branches of an input node.
    """
    def __init__(self, nodeNum, grid_dim = 4):
        self.node_num = nodeNum

        col_factor = 2 * grid_dim - 1
        self.col_factor = col_factor
        self.total_node_count = col_factor * (grid_dim - 1) + grid_dim
        self.total_branches_count = (grid_dim - 1) * (6 * grid_dim - 4)

        self.link_nodes = []
        self.link_branches = []
        self.to_direction = {}
        self.branch_node = {}

        # Find node type
        if (nodeNum % col_factor >= grid_dim):
            self.node_type = 'intersect'
            self.link_count = 4
            self.coord = (nodeNum // col_factor + 0.5 , -1 * (0.5 + (nodeNum % col_factor - grid_dim)))
        elif (nodeNum == 0):
            self.node_type = 'topleft'
            self.link_count = 4
            self.coord = (0,0)
        elif (nodeNum == grid_dim - 1):
            self.node_type = 'bottomleft'
            self.link_count = 4
            self.coord = (0, -1 * (grid_dim - 1))
        elif (nodeNum == self.total_node_count - grid_dim):
            self.node_type = 'topright'
            self.link_count = 4
            self.coord = (grid_dim - 1, 0)
        elif (nodeNum == self.total_node_count - 1):
            self.node_type = 'bottomright'
            self.link_count = 4
            self.coord = (grid_dim - 1, -1 * (grid_dim - 1))
        elif (nodeNum < grid_dim):
            self.node_type = 'left'
            self.link_count = 5
            self.coord = (0, -nodeNum)
        elif (nodeNum % col_factor == 0):
            self.node_type = 'top'
            self.link_count = 5
            self.coord = (nodeNum // col_factor, 0)
        elif (nodeNum > self.total_node_count - grid_dim):
            self.node_type = 'right'
            self.link_count = 5
            self.coord = (grid_dim - 1, -1 * (nodeNum % col_factor))
        elif (nodeNum % col_factor == grid_dim - 1):
            self.node_type ='bottom'
            self.link_count = 5
            self.coord = (nodeNum // col_factor, -1 * (grid_dim - 1))
        else:
            self.node_type = 'interior'
            self.link_count = 8
            self.coord = (nodeNum // col_factor, -1 * (nodeNum % col_factor))

        self.grid_dim = grid_dim
    
    def Describe(self):
        """
        For Debugging purpose, display node properties
        """
        print('Node Number {}:'.format(self.node_num))
        print('Node Type: {}'.format(self.node_type))
        print('Linked Nodes: {}'.format(self.link_nodes))
        print('Linked Branches: {}'.format(self.link_branches))
        print('Direction: (branch, node)\n{}\n\n'.format(self.to_direction))

    def GetDirection(self, node_index):
        """
        Gets the Direction (North, South etc.) to travel from Current Node to the Target Node Index
        """
        from_ind = self.node_num
        to_ind = node_index
        grid_dim = self.grid_dim
        if (from_ind - to_ind == grid_dim):
            return('nw')
        elif(from_ind - to_ind == grid_dim - 1):
            return('sw')
        elif(from_ind - to_ind == self.col_factor):
            return('w')
        elif(from_ind - to_ind == 1):
            return('n')
        elif(to_ind - from_ind == 1):
            return('s')
        elif(to_ind - from_ind == grid_dim):
            return('se')
        elif(to_ind - from_ind == grid_dim - 1):
            return('ne')
        elif(to_ind - from_ind == self.col_factor):
            return('e')
    
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
        return(A)

    @staticmethod
    def GenerateNetworkNodes(grid_dim):
        """
        Given an input of grid dimension, return a list of NodeClass objects corresponding to each node
        """
        grid_dim = grid_dim
        testNode = NodeClass(0, grid_dim)
        node_branch_mat = testNode.node_branch_info()
        total_node_count = testNode.total_node_count

        # The nodes variable stores all the nodes of the networks in NodeClass format. To access NodeClass functions for node number 10, use nodes[10]
        nodes = []

        for i in range(total_node_count):
            newNode = NodeClass(i, grid_dim)
            for j in range(len(node_branch_mat)):
                nodes_of_branch = node_branch_mat[j]
                if (i in nodes_of_branch):
                    newNode.link_branches.append(j)
                    for node_index in nodes_of_branch:
                        if node_index != i:
                            newNode.link_nodes.append(node_index)
                            newNode.branch_node[j] = node_index
                            newNode.to_direction[newNode.GetDirection(node_index)] = (j, node_index)
            nodes.append(newNode)
            if (len(newNode.link_branches) == 2):
                pass
            # newNode.Describe()
        
        return nodes

class RlNetworkAgent:
    """
    Class used by the RL Agent.
    Need to specify grid_dim, source_node.
    Epsilon is the Exploration Factor. Epsilon value of 1 means the decisions are completely random.
    Use Radial Traversal will switch Node Generation Technique to alternate (Make All Decisions From A Node Simultaneously)
    """
    def __init__(self, grid_dim, source_node, epsilon = 0, learning_rate = 0.001, baseline_learning_rate = 0.001, seed = 42, use_radial_traversal = False):
    
        self.grid_dim = grid_dim
        testNode = NodeClass(0, grid_dim)
        self.node_branch_mat = testNode.node_branch_info()
        self.total_node_count = testNode.total_node_count
        self.total_branches_count = testNode.total_branches_count

        # The nodes variable stores all the nodes of the networks in NodeClass format. To access NodeClass functions for node number 10, use nodes[10]
        nodes = []

        for i in range(self.total_node_count):
            newNode = NodeClass(i, grid_dim)
            for j in range(len(self.node_branch_mat)):
                nodes_of_branch = self.node_branch_mat[j]
                if (i in nodes_of_branch):
                    newNode.link_branches.append(j)
                    for node_index in nodes_of_branch:
                        if node_index != i:
                            newNode.link_nodes.append(node_index)
                            newNode.branch_node[j] = i
                            newNode.to_direction[newNode.GetDirection(node_index)] = (j, node_index)
            nodes.append(newNode)
            # newNode.Describe()

        self.nodes = nodes

        np.random.seed(seed)

        # Random Initialisation of Weights
        self.weights = np.random.random_sample((8,8)) * 0.2 - 0.15
        # self.weights = np.zeros((8,8))
        self.bias = np.random.random_sample((8,1)) * 0.2 - 0.1

        # Initialize Other Parameters

        self.count_matrix = np.zeros((1,8), dtype = int)

        self.start_node = source_node

        self.path = []
        self.explored_branches = []
        self.states = []

        self.rejected_branches = []

        self.explored_nodes = []

        self.node_exp_order = []
        self.state_exp_order = []
        self.dec_exp_order = []
        self.dir_exp_order = []

        self.decisions_made = {}

        self.new_branch_gen_steps = []
        self.active_states_history = []

        self.dir_dict = {
            'n': 0,
            'ne': 1,
            'e': 2,
            'se': 3,
            's': 4,
            'sw': 5,
            'w': 6,
            'nw': 7
        }

        self.epsilon = epsilon / 2
        self.seed = seed
        self.baseline = 0
        self.iter_count = 0
        self.lr = learning_rate
        self.baseline_lr = baseline_learning_rate

        self.use_radial = use_radial_traversal

    def split_node_maze(self,node_num, path, cur_counts):
        """
        Function which will recursively explore branches of the network, it will only stop when all paths are completely discovered.
        The branches explored are stored in explored_branches. This can be used to make the var_strng parameter required for simulation.
        """
        cur_node = self.nodes[node_num]
        self.explored_nodes.append(node_num)

        # Decision Probabilities calculated using W.C + b formula
        split_prob = (self.weights.dot(cur_counts.T) + self.bias).flatten()

        decision_values = np.zeros(8, dtype=int)
        # Algorithm
        # Iterate over the 8 directions
        append_count = 0
        for direction in self.dir_dict:
            # If the current direction is accesible from the current node (eg. Intersection nodes cannot access North Direction)
            if (direction in cur_node.to_direction):
                tgt_branch, tgt_node = cur_node.to_direction[direction]
                dir_idx = self.dir_dict[direction]
                # If the path was not already explored
                if (tgt_branch not in self.explored_branches and tgt_node not in self.explored_nodes and tgt_branch not in self.rejected_branches):
                    cur_split_prob = split_prob[dir_idx]

                    if (np.random.rand() < self.epsilon):
                        # Exploration Chance. There is a random chance (epsilon% chance) that the decision made will be opposite of the calculated decision value.
                        cur_split_prob *= -1
                    
                    # Split along this direction only if the value of W.C + b > 0 in this direction
                    if (cur_split_prob > 0):
                        self.explored_branches.append(tgt_branch)
                        self.node_exp_order.append(node_num)
                        self.state_exp_order.append(cur_counts)
                        self.dir_exp_order.append(direction)
                        append_count += 1

                        newpath = path[:]
                        new_counts = cur_counts.copy()
                        new_counts[0,dir_idx] += 1
                        newpath.append((tgt_branch, tgt_node))
                        self.split_node_maze(tgt_node, newpath, new_counts)
                        # has_split = True
                        decision_values[dir_idx] = 1
                    
                    # Otherwise, don't split in that direction
                    else:
                        decision_values[dir_idx] = -1
                        self.rejected_branches.append(tgt_branch)
                # Repeat this for all directions

        self.states.append((cur_counts, np.asarray([decision_values])))
        self.decisions_made[node_num] = decision_values
        for kdx in range(append_count):
            self.dec_exp_order.append(decision_values)
        # Since this is a recursive function, it will only end when all possible paths are fully explored.

    def split_node_radial(self):
        """
        Function which will iteratively generate branches in Alternate Traversal Method.
        Results stored in explored_branches
        """
        active_states = []
        active_states.append((self.start_node, np.zeros((1,8), dtype = int)))
        
        # Algorithm:
        # Look for active nodes. From these active nodes, make decisions. These decisions will lead to new active nodes.
        # The algorithm stops when there are no active nodes left at which decisions can be made.
        #
        # Initialize with 1 active node (source node)
        while (len(active_states) > 0):
            new_active_states = []
            new_branches_made = []

            # Iterate over each active node
            for node_num, cur_counts in active_states:
                cur_node = self.nodes[node_num]
                self.explored_nodes.append(cur_node)

                # Calculate WC + b
                split_prob = (self.weights.dot(cur_counts.T) + self.bias).flatten()
                
                decision_values = np.zeros(8, dtype=int)
                self.node_exp_order.append(node_num)
                self.state_exp_order.append(cur_counts)
                
                append_count = 0
                # Iterate for each direction in the node
                for direction in self.dir_dict:
                    # If can travel in direction
                    if (direction in cur_node.to_direction):
                        tgt_branch, tgt_node = cur_node.to_direction[direction]
                        dir_idx = self.dir_dict[direction]
                        # If not already explored
                        if (tgt_branch not in self.explored_branches and tgt_node not in self.explored_nodes and tgt_branch not in self.rejected_branches):
                            cur_split_prob = split_prob[dir_idx]
                            # Random chance to invert decision
                            if (np.random.rand() < self.epsilon):
                                cur_split_prob *= -1

                            # If Decision value > 0
                            if (cur_split_prob > 0):
                                self.explored_branches.append(tgt_branch)
                                append_count += 1
                                new_branches_made.append(tgt_branch)
                                new_counts = cur_counts.copy()
                                new_counts[0,dir_idx] += 1
                                
                                decision_values[dir_idx] = 1
                                new_active_states.append((tgt_node, new_counts))
                            # If Decision value < 0
                            else:
                                decision_values[dir_idx] = -1
                                self.rejected_branches.append(tgt_branch)
                
                self.states.append((cur_counts, np.asarray([decision_values])))
                self.decisions_made[node_num] = decision_values
                for kdx in range(append_count):
                    self.dec_exp_order.append(decision_values)
                # Repeat for all active states
            
            self.active_states_history.append(active_states.copy())
            self.new_branch_gen_steps.append(new_branches_made)
            # For the next iteration, the active nodes are the new nodes found in this iteration.

            active_states = new_active_states
            # Repeat until no more active states left

    def get_var_strng(self, disable_explore = False, seed = None):
        """
        Method will Traverse Network, Generate Branches and Return the Array of branches in the [0 1 1 0 ...] format.
        """
        old_epsilon = self.epsilon
        # In case exploration should be disabled
        if (disable_explore == True):
            self.epsilon = 0
        self.reset_state()
        if (seed is None):
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        # Depending on type of traversal, use different branch generation function
        if (self.use_radial):
            self.split_node_radial()
        else:
            self.split_node_maze(self.start_node, self.path, self.count_matrix)
        var_strng = np.zeros(self.total_branches_count)
        for i in range(self.total_branches_count):
            if i in self.explored_branches:
                var_strng[i] = 1
        if (disable_explore == True):
            self.epsilon = old_epsilon
        return var_strng

    def reset_state(self):
        """
        Remove generated paths, branches from object, used before re-splitting. Does not change the Trained Weights.
        """
        self.count_matrix = np.zeros((1,8), dtype = int)

        self.path = []
        self.explored_branches = []
        self.explored_nodes = []
        self.states = []

        self.rejected_branches = []

        self.node_exp_order = []
        self.state_exp_order = []
        self.dir_exp_order = []

        self.decisions_made = {}
        self.new_branch_gen_steps = []
        self.active_states_history = []

    def update_weights_single(self, fitness):
        """
        Update weights for a single iteration. Using a single fitness value.
        """
        del_fit = fitness - self.baseline
        for state in self.states:
            C = state[0]
            D = state[1]
            if (C.sum() <= 0):
                continue
            delW = del_fit * D.T.dot(C) / (np.linalg.norm(C, ord = 2))
            delB = del_fit * D.T

            self.weights += delW * self.lr
            self.bias += delB * self.lr
        self.baseline += (fitness - self.baseline) * self.baseline_lr

def demo_nodeclass():
    """
    Code used to generate the video of node splitting (for the ppt demo)
    """

    newAgent = RlNetworkAgent(3, 1, epsilon = 0.1, seed = 42, use_radial_traversal=True)
    newAgent.weights -= 0.0
    newAgent.bias += 0.0

    gen_video = True

    var_strng = newAgent.get_var_strng()
    
    print("Total Nodes: ", newAgent.total_node_count)
    print("Total Branches:", newAgent.total_branches_count)
    print(var_strng)
    print('\n')
    # for state in newAgent.states:
    #     print('\n')
    #     print(state[0])
    #     print(state[1])
    #     print('\n')

    coords = []
    for node in newAgent.nodes:
        coords.append(node.coord)

    source_length = 1

    start_coord = newAgent.nodes[newAgent.start_node].coord
    pre_source_coord = (start_coord[0] - source_length, start_coord[1])

    def split_x_y_coords(pt1, pt2):
        return [pt1[0], pt2[0]], [pt1[1], pt2[1]]

    plt.figure(figsize = (10,7.5))
    plt.scatter(*zip(*coords), marker = 's', s=400)
    plt.axis('off')
    plt.plot([start_coord[0], pre_source_coord[0]], [start_coord[1], pre_source_coord[1]], linewidth = 6, c = 'g')
    plot_count = 0
    plt.savefig('node_progress/0.png')
    for branch in newAgent.explored_branches:
        plot_count += 1
        node_idxs = newAgent.node_branch_mat[branch]
        node_coord_1 = newAgent.nodes[node_idxs[0]].coord
        node_coord_2 = newAgent.nodes[node_idxs[1]].coord
        x_coords, y_coords = split_x_y_coords(node_coord_1, node_coord_2)
        plt.plot(x_coords, y_coords, linewidth = 6, c = 'g')

        idx = plot_count - 1
        dec_node = newAgent.node_exp_order[idx + 1]
        dec_state = newAgent.state_exp_order[idx]
        dec_item = 'Following Branch'
        if newAgent.use_radial:
            dec_dir = f'{dec_node}'
            dec_item = 'Highlighted Node'
        else:
            dec_dir = newAgent.dir_exp_order[idx]
        decision_made = newAgent.dec_exp_order[idx]
        dec_coord1, dec_coord2 = newAgent.nodes[dec_node].coord
        highlight_node = plt.scatter(dec_coord1, dec_coord2, marker = 's', s=400, c = 'g')
        ax = plt.gca()
        infotext = plt.text(0.005, 0.995, "Current State: {}\n{}: {}\nDecision Made: {}".format(dec_state, dec_item, dec_dir, decision_made), horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)
        if (gen_video):
            plt.savefig('node_progress/{}.png'.format(plot_count))
        highlight_node.remove()
        infotext.remove()
        plt.plot(x_coords, y_coords, linewidth = 6, c = 'r')
    plt.show()
    if gen_video:
        print('Rendering Video:')
        # imgDir = 'node_progress/'
        # img_array = []
        # for i in range(plot_count + 1):
        #     img = cv2.imread(imgDir + '{}.png'.format(i))
        #     height, width, _ = img.shape
        #     size = (width, height)
        #     img_array.append(img)

        # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()

# demo_nodeclass()