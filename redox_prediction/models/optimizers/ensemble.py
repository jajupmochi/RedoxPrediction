"""
ensemble



@Author: linlin
@Date: 16.05.23
"""

import numpy as np


def optimize_costs_ensemble(
		nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode, remove_zeros,
		reconstruct_costs=True
):
	"""
	This is the function that reduce the algorithm to the original cost fitting
	problem as presented in Jia2021 paper. This function can serve as a
	preliminary test of the code modification.
	"""
	from redox_prediction.models.optim_costs2 import _optimize_costs_solve
	from redox_prediction.models.optimizers.utils import _reconstruct_costs

	# Construct the triangle inequality constraints:
	tri_rule_list = ([] if tria_rule_mode == 'none' else [
		[1.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, -1.0],
	])

	# Compress columns that are substitutions:
	nb_cost_mat = _compress_all_cols(nb_cost_mat, sorted_label_pairs)

	# Solve the optimization problem:
	edit_costs_new, residual = _optimize_costs_solve(
		nb_cost_mat, dis_k_vec, tri_rule_list
	)

	# Reconstruct the edit costs from the compressed edit costs:
	edit_costs_new = _rejoin_all_cols(edit_costs_new, sorted_label_pairs)

	if reconstruct_costs:
		edit_costs_new = _reconstruct_costs(edit_costs_new, sorted_label_pairs)

	# Notice that the returned residual is the distance instead of the squared
	# distance. You may want to revise the codes where this function is invoked.
	return edit_costs_new, residual


def _compress_all_cols(nb_cost_mat, sorted_label_pairs):
	def _compress(sorted_l_pairs, nb_mat):
		# for unlabeled graphs:
		if sorted_l_pairs == [('', ' '), (' ', ''), (' ', ' ')]:
			return nb_mat
		# for labeled graphs:
		else:
			# Get the number of labels by finding the pairs starting with '':
			n_labels = len([i for i in sorted_l_pairs if i[0] == ''])

			# Get the insertion and deletion columns of nb_mat, namely the
			# [0: 2 * n_labels] columns, and compress the substitution columns:
			nb_mat_new = np.array(
				[[np.sum(i[0:n_labels])] + [np.sum(i[n_labels:2 * n_labels:])] + [
					np.sum(i[2 * n_labels:])] for i in nb_mat]
			)
			return nb_mat_new

	# Get the index of the separation between nodes and edges:
	idx_sep = len(sorted_label_pairs[0])

	# for nodes:
	nb_cost_mat_nodes = nb_cost_mat[:, 0:idx_sep]
	nb_cost_mat_nodes = _compress(sorted_label_pairs[0], nb_cost_mat_nodes)

	# for edges:
	nb_cost_mat_edges = nb_cost_mat[:, idx_sep:]
	nb_cost_mat_edges = _compress(sorted_label_pairs[1], nb_cost_mat_edges)

	# Concatenate the two:
	nb_cost_mat_new = np.concatenate(
		[nb_cost_mat_nodes, nb_cost_mat_edges], axis=1
	)

	# Print out some information:
	print('----- After compression: -----')
	# Print the sum of number of columns in nb_cost_mat as the total number of
	# each edit operation over all pairs of graphs:
	print(
		'Total number of edit operations over all pairs of graphs: '
		'[ ' + '  '.join([str(i) for i in np.sum(nb_cost_mat_new, axis=0)]) + ' ]'
	)
	# Print the number of zeros, the total number, and their ratio in nb_cost_mat:
	print(
		'Number of zeros in nb_cost_mat: {} / {} = {:.2f}%'.format(
			np.sum(nb_cost_mat_new == 0), nb_cost_mat_new.size,
			np.sum(nb_cost_mat_new == 0) / nb_cost_mat_new.size * 100
		)
	)

	return nb_cost_mat_new


def _rejoin_all_cols(edit_costs, sorted_label_pairs):
	def _rejoin(sorted_l_pairs, ed_costs):
		# for unlabeled graphs:
		if sorted_l_pairs == [('', ' '), (' ', ''), (' ', ' ')]:
			return ed_costs
		# for labeled graphs:
		else:
			# Get the number of labels by finding the pairs starting with '':
			n_labels = len([i for i in sorted_l_pairs if i[0] == ''])
			# Get the total length of the edit costs:
			length = len(sorted_l_pairs)
			new_costs = [ed_costs[0] for _ in range(n_labels)]
			new_costs += [ed_costs[1] for _ in range(n_labels)]
			new_costs += [ed_costs[2] for _ in range(length - 2 * n_labels)]
			return new_costs

	# Get node and edge costs.
	node_costs = edit_costs[:3]
	edge_costs = edit_costs[3:]

	# for nodes:
	node_costs = _rejoin(sorted_label_pairs[0], node_costs)

	# for edges:
	edge_costs = _rejoin(sorted_label_pairs[1], edge_costs)

	# Concatenate the two:
	edit_costs_new = np.concatenate([node_costs, edge_costs], axis=0)

	return edit_costs_new