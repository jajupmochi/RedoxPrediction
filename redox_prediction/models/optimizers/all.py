"""
all_costs



@Author: linlin
@Date: 17.05.23
"""


def _optimize_costs_all(
		nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode,
		remove_zeros,
		reconstruct_costs=True
):
	from redox_prediction.models.optim_costs2 import _optimize_costs_solve
	from redox_prediction.models.optimizers.utils import _reconstruct_costs

	# Construct the triangle inequality constraints:
	from redox_prediction.models.optimizers.tri_rule import _construct_tri_rule_vecs
	tri_rule_list = _construct_tri_rule_vecs(sorted_label_pairs, tria_rule_mode)

	# Remove zero columns:
	if remove_zeros:
		from redox_prediction.models.optimizers.utils import _remove_zero_cols
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
		from redox_prediction.models.optimizers.utils import _rejoin_zero_cols
		edit_costs_new = _rejoin_zero_cols(
			edit_costs_new, idx_non_zero, nb_cost_mat.shape[1]
		)

	if reconstruct_costs:
		edit_costs_new = _reconstruct_costs(edit_costs_new, sorted_label_pairs)

	# Notice that the returned residual is the distance instead of the squared
	# distance. You may want to revise the codes where this function is invoked.
	return edit_costs_new, residual


# # @TODO: Test gedlib with unified costs for Acyclic.
# import cvxpy as cp
# x = cp.Variable(6)
# # Sum up the 0 to 1 columns of nb_cost_mat:
# new_nb_cost_mat = np.array(
# 	[[np.sum(i[0:3]), np.sum(i[3:6]), np.sum(i[6:12]), i[12], i[13], i[14]]
# 	 for i in nb_cost_mat]
# )
# cost = cp.sum_squares((new_nb_cost_mat @ x) - dis_k_vec)
# constraints = [
# 	x >= [0.01 for i in range(new_nb_cost_mat.shape[1])],
# 	np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T @ x >= 0.0,
# 	np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T @ x >= 0.0
# ]
# prob = cp.Problem(cp.Minimize(cost), constraints)
# prob.solve()
# edit_costs_new = x.value
# residual = prob.value
#
# edit_costs_new = [edit_costs_new[0]] * 3 + [edit_costs_new[1]] * 3 + [
# 	edit_costs_new[2]] * 6 + list(edit_costs_new[3:])
#
# edit_costs_new = _reconstruct_costs(
# 	edit_costs_new, sorted_label_pairs
# )
#
# return edit_costs_new, residual
