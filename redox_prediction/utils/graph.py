"""
graph

Utility functions for graph processing.

@Author: linlin
@Date: 11.05.23
"""
from typing import List

import numpy as np

import networkx


def convert_sym_attrs_to_integers(
		graphs, return_attrs=False, to_str=False, remove_old_attrs=True,
		inplace=True
):
	"""Converts symbolic attributes on nodes and edges in graphs into integers.

	Parameters
	----------
	graphs : list of networkx.Graph
		A list of graphs.
	return_attrs : bool, optional (default=False)
		Whether to return the converted attributes of nodes and edges.
	to_str : bool, optional (default=False)
		Whether to convert integer attributes to strings.
	remove_old_attrs : bool, optional (default=True)
		Whether to remove the old attributes.
	inplace : bool, optional (default=True)
		Whether to convert the attributes in the graphs inplace.

	Returns
	-------
	graphs : list of networkx.Graph
		A list of graphs with symbolic attributes converted into integers.
	node_attrs : list
		A list of all node attributes.
	edge_attrs : list
		A list of all edge attributes.

	Authors
	-------
	linlin Jia, Github Copilot (2023.05.12)

	Notes
	-----
	Only works for symbolic descriptors / attributes.
	"""
	# Copy graphs if not inplace:
	if not inplace:
		graphs = graphs.copy()

	# Get all labels:
	node_labels, edge_labels = get_all_tuplized_labels(
		graphs, with_keys=True, to_str=False
	)
	# Sort and convert labels to integers:
	node_labels = sorted(list(node_labels))
	edge_labels = sorted(list(edge_labels))
	print('all node labels: ', node_labels)
	print('all edge labels: ', edge_labels)
	print()
	node_labels = dict(zip(node_labels, range(len(node_labels))))
	edge_labels = dict(zip(edge_labels, range(len(edge_labels))))
	if to_str:
		node_labels = {k: str(v) for k, v in node_labels.items()}
		edge_labels = {k: str(v) for k, v in edge_labels.items()}
	# Get all possible node and edge attributes:
	node_attrs = list(node_labels.values())
	edge_attrs = list(edge_labels.values())

	# Convert labels in graphs, remove old labels if specified:
	for graph in graphs:
		# on nodes:
		if len(node_labels):
			for node, attrs in graph.nodes(data=True):
				attrs = tuplize_labels(attrs, with_keys=True, to_str=False)
				graph.nodes[node]['l'] = node_labels[attrs]
		# on edges:
		if len(edge_labels):
			for node1, node2, attrs in graph.edges(data=True):
				attrs = tuplize_labels(attrs, with_keys=True, to_str=False)
				graph.edges[node1, node2]['l'] = edge_labels[attrs]
		# remove old labels (all labels except 'l'):
		if remove_old_attrs:
			if len(node_labels):
				for node, attrs in graph.nodes(data=True):
					keys = list(attrs.keys())
					for attr in keys:
						if attr != 'l':
							graph.nodes[node].pop(attr)
			if len(edge_labels):
				for node1, node2, attrs in graph.edges(data=True):
					keys = list(attrs.keys())
					for attr in keys:
						if attr != 'l':
							graph.edges[node1, node2].pop(attr)

	# return:
	if return_attrs:
		return graphs, node_attrs, edge_attrs
	else:
		return graphs


def tuplize_labels(labels, with_keys=True, to_str=False):
	"""
	Tuplize labels.

	Parameters
	----------
	labels : dict
		A dict of labels.
	with_keys : bool, optional (default=True)
		Whether to include attribute keys in the tuple.
	to_str : bool, optional (default=False)
		Convert labels to strings.

	Returns
	-------
	labels : tuple
		A tuple of labels.
	"""
	if with_keys:
		labels = tuple(labels.items())
		if to_str:
			labels = tuple(map(str, labels))
	else:
		labels = tuple(labels.values())
		if to_str:
			labels = tuple(map(str, labels))
	return labels


# # if dict `labels` has only one key, return the value:
# if len(labels) == 1:
# 	labels = tuple(labels.values())[0]
# 	if to_str:
# 		labels = str(labels)
# # else, return all values:
# else:
# 	labels = tuple(labels.values())
# 	if to_str:
# 		labels = tuple(map(str, labels))
# return labels


def get_all_tuplized_labels(graphs, with_keys=True, to_str=False):
	"""
	Get all labels in graphs.

	Parameters
	----------
	graphs : list of networkx.Graph
		A list of graphs.
	with_keys : bool, optional (default=True)
		Whether to include attribute keys in the tuple.
	to_str : bool, optional (default=False)
		Convert labels to strings.

	Returns
	-------
	node_labels : set
		All labels on nodes.
	edge_labels : set
		All labels on edges.

	Notes
	-----
	Only works for symbolic descriptors / labels.
	"""
	node_labels, edge_labels = set(), set()
	for graph in graphs:
		# on nodes:
		for node, attrs in graph.nodes(data=True):
			node_labels.add(
				tuplize_labels(attrs, with_keys=with_keys, to_str=to_str)
			)
		# on edges:
		for node1, node2, attrs in graph.edges(data=True):
			# @TODO: check if this is correct for labeled edges.
			edge_labels.add(
				tuplize_labels(attrs, with_keys=with_keys, to_str=to_str)
			)
	# Remove empty label tuple.
	if () in node_labels:
		node_labels.remove(())
	if () in edge_labels:
		edge_labels.remove(())
	return node_labels, edge_labels


def get_label_pairs(labels, sort=True, extended=False):
	"""
	Get all possible pairs of labels from a list of labels.

	Parameters
	----------
	labels : list of hashable
		List of labels.
	sort : bool, optional (default=True)
		Sort labels to ensure that the same pair of labels is always represented.
	extended : bool, optional (default=False)
		Include insertions and deletions pairs.

	Returns
	-------
	label_pairs : list of tuples
		List of all possible pairs of labels.
	"""
	# Sort labels to ensure that the same pair of labels is always represented
	# by the same tuple.
	if sort:
		labels = sorted(labels)

	label_pairs = []

	# Include insertions and deletions pairs.
	if extended:
		for i in range(len(labels)):
			if isinstance(i, str) and i == '':
				raise ValueError(
					'The empty string is not allowed as a label. As it is used '
					'to represent the special label (epsilon) for insertion and deletion.'
				)
			label_pairs.append(('', labels[i]))
		for i in range(len(labels)):
			label_pairs.append((labels[i], ''))

	# Get all possible pairs of labels, two directions.
	for i in range(len(labels)):
		for j in range(i + 1, len(labels)):
			label_pairs.append((labels[i], labels[j]))
			label_pairs.append((labels[j], labels[i]))

	return label_pairs


def reorder_graphs_by_index(
		graphs: List[networkx.Graph],
		idx_key: str = 'id'
) -> List[networkx.Graph]:
	"""
	Reorder graphs by their index.

	Parameters
	----------
	graphs: list[networkx.Graph]
		List of graphs to reorder.

	idx_key: str
		Key of the index in the graph.

	Returns
	-------
	graphs: list[networkx.Graph]
		List of graphs reordered by their index.
	"""
	idx = [G.graph[idx_key] for G in graphs]
	idx = np.argsort(idx)
	return [graphs[i] for i in idx]
