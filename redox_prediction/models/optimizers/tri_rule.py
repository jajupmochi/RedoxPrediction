"""
tri_rule

Functions for constructing the triangle inequality constraints.

@Author: linlin
@Date: 17.05.23
"""


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


def _construct_tri_rule_vecs_ensemble(sorted_label_pairs):
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
