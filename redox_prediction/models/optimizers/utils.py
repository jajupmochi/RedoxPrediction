"""
utils



@Author: linlin
@Date: 17.05.23
"""
import numpy as np


def _reconstruct_costs(cost_vec, sorted_label_pairs):
	"""
	Reconstruct the costs vector from the cost vector.

	parameters
	----------
	cost_vec: list
		The cost vector, including node and edge costs, in that order.
	sorted_label_pairs: tuple(list, list)
		The sorted label pairs. The first list contains the node labels, and
		the second list contains the edge labels.

	returns
	-------
	tuple(list, list)
		The reconstructed costs, in the same format as `sorted_label_pairs`.
	"""


	def _vec_to_dict(cost_vec, labels):
		cost_dict = {}
		for i, label in enumerate(labels):
			cost_dict[label] = cost_vec[i]
		return cost_dict


	# Get node and edge costs.
	idx_sep = len(sorted_label_pairs[0])
	node_costs = cost_vec[:idx_sep]
	edge_costs = cost_vec[idx_sep:]

	# Reconstruct node costs.
	node_costs = _vec_to_dict(node_costs, sorted_label_pairs[0])

	# Reconstruct edge costs.
	edge_costs = _vec_to_dict(edge_costs, sorted_label_pairs[1])

	return node_costs, edge_costs


def _remove_zero_cols(nb_cost_mat, constraints):
	"""
	Remove the columns of nb_cost_mat that are zero and the corresponding
	constraints.

	parameters
	----------
	nb_cost_mat: numpy array
		The matrix of edit costs.
	constraints: list of list
		The list of constraints.

	returns
	-------
	numpy array
		The reduced matrix of number of edit costs matrix.
	list of cvxpy constraints
		The reduced list of constraints.
	list of int
		The indices of the non-zero columns.
	"""
	idx_non_zero = [
		i for i in range(nb_cost_mat.shape[1]) if
		np.sum(nb_cost_mat[:, i]) > 0
	]
	new_nb_cost_mat = nb_cost_mat[:, idx_non_zero]
	new_constraints = [
		[c[i] for i in idx_non_zero] for c in constraints
	]

	# Print the number and indices of removed columns:
	print(
		"Removed {} zero columns from the number of edit costs matrix.".format(
			nb_cost_mat.shape[1] - new_nb_cost_mat.shape[1]
		)
	)
	print(
		"Removed columns: {}".format(
			[i for i in range(nb_cost_mat.shape[1]) if
			np.sum(nb_cost_mat[:, i]) == 0]
		)
	)

	return new_nb_cost_mat, new_constraints, idx_non_zero


def _rejoin_zero_cols(edit_costs, idx_non_zero, total_len):
	"""
	Rejoin the zero columns to the edit costs.

	parameters
	----------
	edit_costs: list of float
		The edit costs to be rejoined.
	idx_non_zero: list of int
		The indices of the non-zero columns.
	total_len: int
		The total length of the rejoined edit costs.

	returns
	-------
	list of float
		The new edit costs with the zero columns.
	"""
	edit_costs_new = [0.0] * total_len
	for i, idx in enumerate(idx_non_zero):
		edit_costs_new[idx] = edit_costs[i]

	return edit_costs_new
