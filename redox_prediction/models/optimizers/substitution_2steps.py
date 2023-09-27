"""
optim_substitution_2steps



@Author: linlin
@Date: 16.05.23
"""
import numpy as np


def optimize_costs_substitution_2steps(
		nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode,
		remove_zeros,
		reconstruct_costs=True
):
	from redox_prediction.models.optim_costs2 import _optimize_costs_solve
	from redox_prediction.models.optimizers.utils import _reconstruct_costs

	# Step 1: optimize edit costs with the original / ensemble optimizer.
	print(
		'===== Step 1: optimize edit costs with the original / ensemble '
		'optimizer: ====='
	)
	from redox_prediction.models.optimizers.ensemble import \
		optimize_costs_ensemble
	edit_costs_ori, residual = optimize_costs_ensemble(
		nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode,
		remove_zeros,
		reconstruct_costs=False
	)
	print('residual after the 1st step optimization:', residual)

	# Step 2: further optimize the substitution cost:
	print('===== Step 2: further optimize the substitution cost: =====')

	# Construct the triangle inequality constraints:
	tri_rule_list = (
		_construct_tri_rule_vecs(
			sorted_label_pairs, tria_rule_mode
		) if tria_rule_mode == 'each' else []
	)

	# Reconstruct nb_cost_mat and dis_k_vec:
	nb_cost_mat_new, dis_k_vec_new, idx_sep_new, non_zero_rows = \
		_retrieve_vecs_for_substitution(
			nb_cost_mat, sorted_label_pairs, edit_costs_ori, dis_k_vec
		)

	# Solve the optimization problem:
	edit_costs_new, residual = _optimize_costs_solve(
		nb_cost_mat_new, dis_k_vec_new, tri_rule_list
	)

	# Add insertion and deletion costs:
	edit_costs_new = _rejoin_all_costs(
		edit_costs_new, edit_costs_ori, idx_sep_new, len(sorted_label_pairs[0])
	)

	if reconstruct_costs:
		edit_costs_new = _reconstruct_costs(edit_costs_new, sorted_label_pairs)

	# Notice that the returned residual is the distance instead of the squared
	# distance. You may want to revise the codes where this function is invoked.
	return edit_costs_new, residual


def _retrieve_vecs_for_substitution(
		nb_cost_mat, sorted_label_pairs, edit_costs_ori, dis_k_vec
):
	def _retrieve(sorted_l_pairs, nb_mat, ori_costs, i_sep):
		# for unlabeled graphs, return empty arrays, which will be ignored in
		# the optimization process:
		if sorted_l_pairs == [('', ' '), (' ', ''), (' ', ' ')]:
			# nb_mat_new = nb_mat[:, 2:]
			# # Compute corresponding distance vector:
			# d_vec_new = []
			# for idx, dis in enumerate(d_vec):
			# 	d_vec_new.append(ori_costs[2] * nb_mat_new[idx, 0])
			# Return an empty array of siwe (nb_mat.shape[0], 0):
			return np.empty((nb_mat.shape[0], 0), dtype=np.int64), [0] * nb_mat.shape[0]
		# for labeled graphs:
		else:
			# @TODO: do this for edge labels.
			# Get the number of labels by finding the pairs starting with '':
			n_labels = len([i for i in sorted_l_pairs if i[0] == ''])

			# Get the substitution columns of nb_mat, namely the [2 * n_labels: ]
			# columns:
			i_s_range = range(2 * n_labels, i_sep)
			nb_mat_new = np.array([i[i_s_range] for i in nb_mat])

			# Compute the corresponding distance vector. Instead of computing
			# directly by ns * cs for each substitution, we minus the multiplication
			# of the other operations from the total distance. Otherwise, the
			# resulting distance vector will have too many zeros and same values.
			# Get the indices of the other operations:
			i_others = [i for i in range(ori_costs.size) if i not in i_s_range]
			# Get the corresponding costs:
			cs_others = ori_costs[i_others]
			# Get the corresponding nb_mat:
			nb_mat_others = nb_mat[:, i_others]
			# Compute the corresponding distance vector:
			d_vec_new = list(np.dot(nb_mat_others, cs_others) - np.array(dis_k_vec))

			return nb_mat_new, d_vec_new


	# Get the index of the separation between nodes and edges:
	idx_sep = len(sorted_label_pairs[0])

	# for nodes:
	nb_cost_mat_nodes = nb_cost_mat[:, 0:idx_sep]
	ori_costs_nodes = edit_costs_ori[:idx_sep]
	nb_cost_mat_nodes, dis_vec_nodes = _retrieve(
		sorted_label_pairs[0], nb_cost_mat, edit_costs_ori, idx_sep
	)
	idx_sep_new = nb_cost_mat_nodes.shape[1]

	# for edges:
	# nb_cost_mat_edges = nb_cost_mat[:, idx_sep:]
	# ori_costs_edges = edit_costs_ori[idx_sep:]
	nb_cost_mat_edges, dis_vec_edges = _retrieve(
		sorted_label_pairs[1], nb_cost_mat, edit_costs_ori, idx_sep
	)

	# Concatenate the two:
	nb_cost_mat_new = np.concatenate(
		[nb_cost_mat_nodes, nb_cost_mat_edges], axis=1
	)
	# Add up the two distance vectors:
	dis_k_vec_new = [a + b for a, b in zip(dis_vec_nodes, dis_vec_edges)]

	# Remove all zero rows:
	# Get the indices of non-zero rows:
	non_zero_rows = np.where(np.sum(nb_cost_mat_new, axis=1) != 0)[0]
	# Get the non-zero rows:
	nb_cost_mat_new = nb_cost_mat_new[non_zero_rows, :]
	# Get the corresponding distance vector:
	dis_k_vec_new = [dis_k_vec_new[i] for i in non_zero_rows]

	# Print out some information:
	print('----- For selected edit operations: -----')
	# Print the sum of number of columns in nb_cost_mat as the total number of
	# each edit operation over all pairs of graphs:
	print(
		'Total number of edit operations over all pairs of graphs: '
		'[ ' + '  '.join(
			[str(i) for i in np.sum(nb_cost_mat_new, axis=0)]
		) + ' ]'
	)
	# Print the number of zeros, the total number, and their ratio in nb_cost_mat:
	print(
		'Number of zeros in nb_cost_mat: {} / {} = {:.2f}%'.format(
			np.sum(nb_cost_mat_new == 0), nb_cost_mat_new.size,
			np.sum(nb_cost_mat_new == 0) / nb_cost_mat_new.size * 100
		)
	)
	# Print the new distance vector:
	print(
		'New distance vector: [ ' + '  '.join(
			[str(i) for i in dis_k_vec_new[0:20]]
		) + (' ]' if len(dis_k_vec_new) <= 20 else '  ...  ]')
	)
	# Print the number of zeros rows and the total number of rows in nb_cost_mat:
	print(
		'Number of zero rows in nb_cost_mat: {} / {} = {:.2f}%'.format(
			len(non_zero_rows), nb_cost_mat_new.shape[0],
			len(non_zero_rows) / nb_cost_mat_new.shape[0] * 100
		)
	)

	return nb_cost_mat_new, dis_k_vec_new, idx_sep_new, non_zero_rows


def _rejoin_all_costs(edit_costs, ori_costs, idx_sep_new, idx_sep_ori):
	def _rejoin(ed_costs, ori_cs):
		# for unlabeled graphs, return the original costs:
		if ed_costs.size == 0:
			return ori_cs
		# for labeled graphs:
		else:
			# Add the insertion and deletion costs at the beginning:
			idx_ins = len(ori_cs) - len(ed_costs)
			new_costs = np.concatenate((ori_cs[:idx_ins], ed_costs), axis=0)
			return new_costs


	# Get node and edge costs.
	node_costs = edit_costs[:idx_sep_new]
	edge_costs = edit_costs[idx_sep_new:]

	# for nodes:
	node_costs = _rejoin(node_costs, ori_costs[:idx_sep_ori])

	# for edges:
	edge_costs = _rejoin(edge_costs, ori_costs[idx_sep_ori:])

	# Concatenate the two:
	edit_costs_new = np.concatenate([node_costs, edge_costs], axis=0)

	return edit_costs_new


# %% functions for constructing the triangle inequality constraints:


def _construct_tri_rule_vecs(sorted_label_pairs, rule_mode):
	"""
	Construct the vectors of triangle inequality constraints for the optimization
	problem.

	parameters
	----------
	sorted_label_pairs: tuple(list, list)
		The sorted label pairs. The first list contains the node labels, and
		the second list contains the edge labels.

	returns
	-------
	list of list of float
		The vectors of triangle inequality constraints.
	"""
	# if no rule:
	if rule_mode == 'none':
		return []
	# if rule for each pair of labels:
	elif rule_mode == 'each':
		return []
		return _construct_tri_rule_vecs_each(sorted_label_pairs)
	# if only two rules for all labels:
	elif rule_mode == 'ensemble':
		return _construct_tri_rule_vecs_ensemble(sorted_label_pairs)
	else:
		raise ValueError("Invalid rule model_type: {}.".format(rule_mode))


def _construct_tri_rule_vecs_each(sorted_label_pairs):
	def _construct(sorted_l_pairs):
		# for unlabeled graphs:
		if sorted_l_pairs == [('', ' '), (' ', ''), (' ', ' ')]:
			return [[1.0, 1.0, -1.0]]
		# for labeled graphs:
		else:
			# Get the number of labels by finding the pairs starting with '':
			n_labels = len([i for i in sorted_l_pairs if i[0] == ''])

			# Construct the vectors for each pair of labels.
			rule_vecs = []
			for i in range(n_labels):
				for j in range(i + 1, n_labels):
					# Get the labels of the current edit operation.
					label1 = sorted_l_pairs[i][1]
					label2 = sorted_l_pairs[j][1]
					# Find the indices of the corresponding operation in the vector:
					_add_rule_vec_for_1_pair(
						label1, label2, rule_vecs, sorted_l_pairs
					)
					_add_rule_vec_for_1_pair(
						label2, label1, rule_vecs, sorted_l_pairs
					)

			return rule_vecs


	def _add_rule_vec_for_1_pair(label1, label2, rule_vecs, sorted_l_pairs):
		idx_i = sorted_l_pairs.index(('', label1))  # insertion
		idx_r = sorted_l_pairs.index((label2, ''))  # removal
		idx_s1 = sorted_l_pairs.index((label1, label2))  # substitution 1
		idx_s2 = sorted_l_pairs.index((label2, label1))  # substitution 2
		# First direction of substitution:
		vec = [0] * len(sorted_l_pairs)
		vec[idx_i] = 1.0
		vec[idx_r] = 1.0
		vec[idx_s1] = -1.0
		rule_vecs.append(vec)
		# Second direction of substitution:
		vec = [0] * len(sorted_l_pairs)
		vec[idx_i] = 1.0
		vec[idx_r] = 1.0
		vec[idx_s2] = -1.0
		rule_vecs.append(vec)


	# Construct the vectors for the node labels:
	node_rule_vecs = _construct(sorted_label_pairs[0])
	# Extend the vectors to the correct length:
	node_rule_vecs = [
		vec + [0.0] * len(sorted_label_pairs[1]) for vec in node_rule_vecs
	]

	# Construct the vectors for the edge labels:
	edge_rule_vecs = _construct(sorted_label_pairs[1])
	# Extend the vectors to the correct length:
	edge_rule_vecs = [
		[0.0] * len(sorted_label_pairs[0]) + vec for vec in edge_rule_vecs
	]

	# Combine the node and edge vectors:
	rule_vecs = node_rule_vecs + edge_rule_vecs

	return rule_vecs


def _construct_tri_rule_vecs_ensemble(sorted_label_pairs, ori_costs):
	"""
	Construct the vectors of triangle inequality constraints for the optimization
	problem. This function is specialized for the substitution_2steps optimizer.
	"""


	def _construct(sorted_l_pairs):
		# for unlabeled graphs:
		if sorted_l_pairs == [('', ' '), (' ', ''), (' ', ' ')]:
			return [[1.0, 1.0, -1.0]]
		# for labeled graphs:
		else:
			# Get the number of labels by finding the pairs starting with '':
			n_labels = len([i for i in sorted_l_pairs if i[0] == ''])

			# Construct the vectors for each pair of labels.
			rule_vec = [0.0] * len(sorted_l_pairs)
			# for insertion:
			for i in range(n_labels):
				rule_vec[i] = 1.0
			# for removal:
			for i in range(n_labels):
				rule_vec[i + n_labels] = 1.0
			# for substitution:
			for i in range(n_labels * 2, len(sorted_l_pairs)):
				rule_vec[i] = -1.0

			return [rule_vec]


	# Construct the vectors for the node labels:
	node_rule_vecs = _construct(sorted_label_pairs[0])
	# Extend the vectors to the correct length:
	node_rule_vecs = [
		vec + [0.0] * len(sorted_label_pairs[1]) for vec in node_rule_vecs
	]

	# Construct the vectors for the edge labels:
	edge_rule_vecs = _construct(sorted_label_pairs[1])
	# Extend the vectors to the correct length:
	edge_rule_vecs = [
		[0.0] * len(sorted_label_pairs[0]) + vec for vec in edge_rule_vecs
	]

	# Combine the node and edge vectors:
	rule_vecs = node_rule_vecs + edge_rule_vecs

	return rule_vecs
