"""
count_descriptors

Count the number of different descriptors / labels on nodes and edges of all
graphs.

@Author: linlin
@Date: 10.05.23
"""


def compress_labels_in_graphs(graphs):
	"""
	Compress labels in graphs.

	Parameters
	----------
	graphs : list of networkx.Graph
		A list of graphs.

	Returns
	-------
	graphs : list of networkx.Graph
		A list of graphs with compressed labels.
	"""
	# Compress labels:
	for graph in graphs:
		# on nodes:
		for node, attrs in graph.nodes(data=True):
			labels = tuple(attrs.values())
			# Remove all labels in graph node:
			graph.nodes[node].clear()
			graph.nodes[node]['labels'] = labels
		# on edges:
		for u, v, attrs in graph.edges(data=True):
			labels = tuple(attrs.values())
			# Remove all labels in graph edge:
			graph.edges[u, v].clear()
			graph.edges[u, v]['labels'] = labels
	return graphs


def count_each_label_in_graphs(graphs, compress_labels=True):
	"""
	Count the number of each different descriptors / labels on nodes and edges
	of all graphs.

	Parameters
	----------
	graphs : list of networkx.Graph
		A list of graphs.

	Returns
	-------
	counts_nodes : dict
		The number of each different descriptor / label on nodes.
	counts_edges : dict
		The number of each different descriptor / label on edges.

	Notes
	-----
	Only works for symbolic descriptors / labels.
	"""
	# Compress labels:
	if compress_labels:
		graphs = compress_labels_in_graphs(graphs)

	counts_nodes, counts_edges = {}, {}
	for graph in graphs:
		# on nodes:
		for node, attrs in graph.nodes(data=True):
			for k, v in attrs.items():
				if k not in counts_nodes:
					counts_nodes[k] = {}
				if v not in counts_nodes[k]:
					counts_nodes[k][v] = 0
				counts_nodes[k][v] += 1
		# on edges:
		for node1, node2, attrs in graph.edges(data=True):
			for k, v in attrs.items():
				if k not in counts_edges:
					counts_edges[k] = {}
				if v not in counts_edges[k]:
					counts_edges[k][v] = 0
				counts_edges[k][v] += 1
	return counts_nodes, counts_edges


if __name__ == '__main__':
	ds_name = 'brem_togn'  # 'Acyclic', 'Alkane_unlabeled', 'brem_togn'
	descriptor = '1hot'  # 'atom_bond_types', '1hot', '1hot-dis', 'af1hot+3d-dis'
	from ged_cost_learning.dataset.load_dataset import get_data

	ds_kwargs = {
		'level': 'pbe0', 'target': 'dGred',
		'sort_by_targets': True,
		'fn_families': None,
	}
	ds_dir_feat = '../datasets/Redox/'

	if descriptor == 'atom_bond_types':
		graphs, _, _ = get_data(
			ds_name, descriptor='smiles', format_='networkx', **ds_kwargs
		)

	# 2. Get descriptors: one-hot descriptors on nodes and edges.
	elif descriptor in ['1hot', '1hot-dis', 'af1hot+3d-dis']:
		from redox_prediction.dataset.get_featurizer import get_featurizer

		featurizer = get_featurizer(
			descriptor, ds_dir=ds_dir_feat,
			use_atomic_charges='none'
		)
		# The string is used for the GEDLIB module of the graphkit-learn library.
		ds_kwargs['feats_data_type'] = 'str'
		graphs, _, _ = get_data(
			ds_name, descriptor=featurizer, format_='networkx',
			coords_dis=(True if descriptor == 'af1hot+3d-dis' else False),
			**ds_kwargs
		)

	counts_nodes, counts_edges = count_each_label_in_graphs(graphs)
	print(counts_nodes)
	print(counts_edges)

# Acyclic
# {'atom_symbol': {'C': 1243, 'O': 156, 'S': 93}}
# {}

# brem_togn
# {'symbol': {'C': 2449, 'N': 277, 'S': 38, 'O': 180, 'Cl': 19, 'Br': 7, 'F': 1, 'I': 1}}
# {'bond_type_double': {'1.5': 976, '1.0': 1728, '2.0': 559, '3.0': 13}}


# brem_togn (1hot)
# {
# 	'labels': {  # 45 diff labels.
# 		('1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		 '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		 '1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		 '0.0', '0.0'): 551, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 386, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 33, (
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 29, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 247, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 2, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 278, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '1.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 112, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0',
# 		'0.0', '0.0'): 186, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 668, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0'): 26, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 9, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 155, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 61, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 17, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 13, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 11, (
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 19, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 2, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0'): 25, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 8, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 2, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 3, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'-1.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 3, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 3, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 10, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 3, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 4, (
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 8, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 7, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 25, (
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 7, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 29, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 4, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 1, (
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 1, (
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 1, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 1, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 3, (
# 		'0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 1, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 1, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 11, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '1.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0'): 1, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 5
# 	}
# }
# {
# 	'labels': {
# 		('0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
# 		 '0.0', '1.0'): 2972, (
# 		'0.0', '0.0', '0.0', '1.0', '1.0', '1.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 976, (
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 551, (
# 		'1.0', '0.0', '0.0', '0.0', '1.0', '1.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 678, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 395, (
# 		'1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 104, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 96, (
# 		'0.0', '1.0', '0.0', '0.0', '1.0', '1.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 434, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 9, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0',
# 		'0.0', '0.0'): 4, (
# 		'0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 24, (
# 		'0.0', '0.0', '1.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 4, (
# 		'0.0', '1.0', '0.0', '0.0', '1.0', '0.0', '1.0', '0.0', '0.0', '0.0',
# 		'0.0', '0.0'): 1
# 	}
# }
