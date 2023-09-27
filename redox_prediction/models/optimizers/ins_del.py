"""
ins_del



@Author: linlin
@Date: 17.05.23
"""
import numpy as np


def _optimize_costs_ins_del(
		nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode, remove_zeros,
		reconstruct_costs=True
):
	from redox_prediction.models.optim_costs2 import _optimize_costs_solve
	from redox_prediction.models.optimizers.utils import _reconstruct_costs, \
		_remove_zero_cols, _rejoin_zero_cols
	from redox_prediction.models.optimizers.tri_rule import _construct_tri_rule_vecs

	# Construct the triangle inequality constraints:
	tri_rule_list = _construct_tri_rule_vecs(sorted_label_pairs, tria_rule_mode)

	# Compress columns that are substitutions:
	nb_cost_mat, tri_rule_list, idx_rel = _compress_substitution_cols(
		nb_cost_mat, tri_rule_list, sorted_label_pairs
	)

	# Remove zero columns:
	if remove_zeros:
		new_nb_cost_mat, new_tri_rule_list, idx_non_zero = _remove_zero_cols(
			nb_cost_mat, tri_rule_list
		)
	else:
		new_nb_cost_mat = nb_cost_mat
		new_tri_rule_list = tri_rule_list

	# Solve the optimization problem:
	edit_costs_new, residual = _optimize_costs_solve(
		new_nb_cost_mat, dis_k_vec, new_tri_rule_list
	)

	if remove_zeros:
		edit_costs_new = _rejoin_zero_cols(edit_costs_new, idx_non_zero, nb_cost_mat.shape[1])

	# Reconstruct the edit costs from the compressed edit costs:
	edit_costs_new = _rejoin_substitution_cols(
		edit_costs_new, sorted_label_pairs, idx_rel
	)

	if reconstruct_costs:
		edit_costs_new = _reconstruct_costs(edit_costs_new, sorted_label_pairs)

	# Notice that the returned residual is the distance instead of the squared
	# distance. You may want to revise the codes where this function is invoked.
	return edit_costs_new, residual


def _compress_substitution_cols(nb_cost_mat, tri_rule_list, sorted_label_pairs):
	def _compress(sorted_l_pairs, nb_mat, rule_list):
		# for unlabeled graphs:
		if sorted_l_pairs == [('', ' '), (' ', ''), (' ', ' ')]:
			return nb_mat, rule_list
		# for labeled graphs:
		else:
			# Get the number of labels by finding the pairs starting with '':
			n_labels = len([i for i in sorted_l_pairs if i[0] == ''])

			# Get the insertion and deletion columns of nb_mat, namely the
			# [0: 2 * n_labels] columns, and compress the substitution columns:
			nb_mat_new = np.array(
				[list(i[0:2 * n_labels]) + [np.sum(i[2 * n_labels:])] for i in nb_mat]
			)
			# Same for the rule list:
			rule_list_new = [
				i[0:2 * n_labels] + [i[2 * n_labels]] for i in rule_list
			]
			return nb_mat_new, rule_list_new

	# Get the index of the separation between nodes and edges:
	idx_sep = len(sorted_label_pairs[0])

	# for nodes:
	nb_cost_mat_nodes = nb_cost_mat[:, 0:idx_sep]
	tri_rule_list_nodes = [i[0:idx_sep] for i in tri_rule_list]
	nb_cost_mat_nodes, tri_rule_list_nodes = _compress(
		sorted_label_pairs[0], nb_cost_mat_nodes, tri_rule_list_nodes
	)
	idx_sep_new = nb_cost_mat_nodes.shape[1]

	# for edges:
	nb_cost_mat_edges = nb_cost_mat[:, idx_sep:]
	tri_rule_list_edges = [i[idx_sep:] for i in tri_rule_list]
	nb_cost_mat_edges, tri_rule_list_edges = _compress(
		sorted_label_pairs[1], nb_cost_mat_edges, tri_rule_list_edges
	)

	# Concatenate the two:
	nb_cost_mat_new = np.concatenate(
		[nb_cost_mat_nodes, nb_cost_mat_edges], axis=1
	)
	tri_rule_list_new = [
		i + j for i, j in zip(tri_rule_list_nodes, tri_rule_list_edges)
	]

	# Remove the same rows from the rule list:
	tri_rule_list_new = list(set(tuple(i) for i in tri_rule_list_new))

	return nb_cost_mat_new, tri_rule_list_new, idx_sep_new


def _rejoin_substitution_cols(edit_costs, sorted_label_pairs, idx_sep):
	def _rejoin(sorted_l_pairs, ed_costs):
		# for unlabeled graphs:
		if sorted_l_pairs == [('', ' '), (' ', ''), (' ', ' ')]:
			return ed_costs
		# for labeled graphs:
		else:
			# Get the total length of the edit costs:
			length = len(sorted_l_pairs)
			# Append the substitution costs to the end of the edit costs:
			cs = ed_costs[-1]
			len_to_append = length - len(ed_costs)
			ed_costs = np.append(ed_costs, [cs for i in range(len_to_append)])
			return ed_costs

	# Get node and edge costs.
	node_costs = edit_costs[:idx_sep]
	edge_costs = edit_costs[idx_sep:]

	# for nodes:
	node_costs = _rejoin(sorted_label_pairs[0], node_costs)

	# for edges:
	edge_costs = _rejoin(sorted_label_pairs[1], edge_costs)

	# Concatenate the two:
	edit_costs_new = np.concatenate([node_costs, edge_costs], axis=0)

	return edit_costs_new
