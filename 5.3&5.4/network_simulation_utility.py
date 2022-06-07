# Utility functions for network construction

from feasibility_simulation_code import Network
from original_simulation_code import Network as SimNet
import numpy as np

def CountBranches(grid_dim: int) -> int:
	return (grid_dim - 1) * (6 * grid_dim - 4)

def CreateNetwork(params) -> Network:
	
	# Extract key value pairs
	if "grid_dim" in params:
		grid_dim = params['grid_dim']
	else:
		grid_dim = 3
	if "branch_config" in params:
		branch_config = params["branch_config"]
	else:
		branch_config=np.ones(CountBranches(grid_dim))
	if 'source_nodes' in params:
		source_nodes = params['source_nodes']
	else:
		source_nodes = [1]
	if 'sink_nodes' in params:
		sink_nodes = params['sink_nodes']
	else:
		sink_nodes = [10,12]
	if 'rr_list' in params:
		rr_list = params['rr_list']
	else:
		rr_list = [1, 2]
	# Fix Random Seed for length-variation in Network Model
	if 'rn_list' in params:
		rn1, rn2, rn3 = params['rn_list']
	else:
		rn1, rn2, rn3 = 50, 98, 54
	
	if len(branch_config) != CountBranches(grid_dim):
		raise ValueError(f"Branch Configuration array size {len(branch_config)}"\
		f" does not equal expected size of {CountBranches(grid_dim)} for grid size of {grid_dim}")

	# Fix the Network Model
	network = Network(grid_dim = grid_dim,var_strng=branch_config,
		source_nodes=source_nodes,sink_nodes=sink_nodes,length_factor=0.02,Q_in=10*10**(-6)/3600, 
		P_out=1.01325 *10** 5 ,width_channel=1000 * (10 ** -6),height_channel=1000 * (10 ** -6),
		beta=1.5, rr_list=rr_list,vscty=50 * (10 ** -3),rn1=rn1,rn2=rn2,rn3=rn3)
	
	network.node_branch_info()
	if not IsNetworkFeasible(network):
		return None
	network.node_connector_matrix(network.var_strng)
	network.calc_length_branches()
	network.remove_branches()
	network.calc_length_branches()
	network.incidence_matrix_gen()
	network.branch_connectivity_fn()
	network.calc_length_branches()
	
	return network

def IsNetworkFeasible(network: Network):
	targets = set(network.sink_nodes)
	if len(targets) <= 0:
		return False
	cur_nodes = network.source_nodes
	visited = [False] * len(network.var_strng)
	for cur_node in cur_nodes:
		visited[cur_node] = True
	node_branch_mat = network.node_branch_info()
	while len(cur_nodes) > 0:
		new_nodes = []
		for node in cur_nodes:
			for branch in network.linked_branches(node):
				if network.var_strng[branch] > 0:
					for new_node in node_branch_mat[branch,:]:
						if not visited[new_node]:
							new_nodes.append(new_node)
							visited[new_node] = True
						if new_node in targets:
							targets.remove(new_node)
		cur_nodes = new_nodes
	return len(targets) <= 0

def IsBranchConfigFeasible(grid_dim: int, branch_config: list[int], source_nodes: list[int], sink_nodes: list[int]):
	params = {
		'grid_dim': grid_dim,
		'branch_config': branch_config,
		'source_nodes': source_nodes,
		'sink_nodes': sink_nodes
	}
	network = CreateNetwork(params)
	return network is not None


def SimulateNetwork(drops:list[list[int]], spacing = None, params: dict = {}, ret_sequence = False, ret_raw_seq = False, ret_network = False, gen_images = False):
	if spacing is None:
		spacing= np.array(([[0,	-0.16933,	-0.410262,	-0.8505267]]))
		# Extract key value pairs
	if "grid_dim" in params:
		grid_dim = params['grid_dim']
	else:
		grid_dim = 3
	if "branch_config" in params:
		branch_config = params["branch_config"]
	else:
		branch_config=np.ones(CountBranches(grid_dim))
	if 'source_nodes' in params:
		source_nodes = params['source_nodes']
	else:
		source_nodes = [1]
	if 'sink_nodes' in params:
		sink_nodes = params['sink_nodes']
	else:
		sink_nodes = [10,12]
	if 'rr_list' in params:
		rr_list = params['rr_list']
	else:
		rr_list = [1, 2]
	if 'rn_list' in params:
		rn1, rn2, rn3 = params['rn_list']
	else:
		rn1, rn2, rn3 = 50, 98, 54
	length_factor = params.get('length_factor', 20 * (10 ** -3))
	num_drps = sum([len(space) for space in spacing])
	simnet = SimNet(grid_dim, branch_config, num_drps, source_nodes, sink_nodes, drops, spacing,
		length_factor=length_factor, Q_in=10 * 10**(-6)/3600, P_out=1.01325 * 10 ** 5,
		width_channel=1 * (10 ** -3), height_channel=1 * (10 ** -3), interfacial_tension=45*0.001, rr=2, rr_list = rr_list,
		alpha=0.9, beta=1.5, vscty=50 * (10 ** -3), toterminate=False,
		rn1=rn1, rn2=rn2, rn3=rn3,
		subdue_print = True, output_img_ext = '.png', should_gen_images= gen_images)
	simnet.simulate_network_func()
	fitness = simnet.network_fitness_calc_fn()
	e_q = simnet.event_sequence
	t_s = [0]
	for time in simnet.delt_tmin_details:
		t_s.append(t_s[-1] + time)
	ret_params = [fitness]
	if ret_sequence:
		ret_params.append(e_q)
		ret_params.append(t_s)
	if ret_raw_seq:
		ret_params.append(simnet.raw_event_sequence)
	if ret_network:
		ret_params.append(simnet)
	if len(ret_params) == 1:
		return ret_params[0]
	return ret_params