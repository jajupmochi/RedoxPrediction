"""
ged



@Author: linlin
@Date: 27.07.23
"""
import numpy as np
from gklearn.ged.util import pairwise_ged, get_nb_edit_operations


def compute_ged(
		Gi,
		Gj,
		edit_cost,
		edit_cost_fun='CONSTANT',
		method='BIPARTITE',
		repeats=1,
		return_nb_eo=False,
		**kwargs
):
	"""
	Compute GED between two graph according to edit_cost.
	"""
	ged_options = {
		'edit_cost': edit_cost_fun,
		'method': method,
		'edit_cost_constants': edit_cost
	}
	node_labels = kwargs.get('node_labels', [])
	edge_labels = kwargs.get('edge_labels', [])
	node_attrs = kwargs.get('node_attrs', [])
	edge_attrs = kwargs.get('edge_attrs', [])
	dis, pi_forward, pi_backward = pairwise_ged(
		Gi, Gj, ged_options, repeats=repeats
	)

	if return_nb_eo:
		n_eo_tmp = get_nb_edit_operations(
			Gi, Gj, pi_forward, pi_backward, edit_cost=edit_cost_fun,
			node_labels=node_labels, edge_labels=edge_labels,
			node_attrs=node_attrs, edge_attrs=edge_attrs
		)

		#  @TODO: for test only:
		# Assert if the multiplication of the number of edit operations and the
		# edit costs is equal to the computed distance:
		try:
			import numpy as np

			dis_computed = np.matmul(n_eo_tmp, edit_cost)
			assert dis == dis_computed
		except AssertionError:
			# print('AssertionError: dis != dis_computed')
			# print('dis:', dis)
			# print('dis_computed:', dis_computed)
			# print('n_eo_tmp: ', n_eo_tmp)
			# print('sorted_costs: ', sorted_costs)
			# print('Trying `np.isclose()` instead...')
			assert np.isclose(dis, dis_computed)

			return dis, n_eo_tmp

	else:
		return dis


def get_fitted_edit_costs(ds_name, method, descriptor, y_scaling=None):
	# The edit costs are fitted using the method introduced in the ACPR 2023
	# paper. The costs correspond to the best embedding space for each dataset
	# are used.
	if method == 'BIPARTITE':

		if ds_name == 'MUTAG':
			return np.array([2.4, 3.5, 5.8, 2.5, 1.7, 4.1])

		elif ds_name == 'brem_togn_dGred':
			# The following 2 costs are only optimized for y_scaling "std" and
			# used the jia2021 paper. todo: use jia2023 paper and optimize for all.
			if descriptor == '1hot':
				return np.array([4.2, 3.9, 4.9, 0.1, 0.1, 0.2])
			elif descriptor == 'af1hot+3d-dis':
				return np.array([2.1, 1.9, 3.2, 0.1, 0.1, 0.2])
			# The following 2 costs are only optimized used the jia2023 paper
			# for y_scaling "none" and for descriptor "atom_bond_types".
			# todo: optimize for other y_scaling.
			elif descriptor == 'atom_bond_types':
				return np.array([1.7, 2.3, 4.0, 2.9, 2.5, 5.4])
			elif descriptor == 'unlabeled':
				return np.array([1.7, 2.3, 4.0, 2.9, 2.5, 5.4])
			else:
				raise ValueError('Unknown descriptor: {}.'.format(descriptor))

		elif ds_name == 'brem_togn_dGox':
			# The following 2 costs are only optimized for y_scaling "std" and
			# used the jia2021 paper. todo: use jia2023 paper and optimize for all.
			if descriptor == '1hot':
				return np.array([3.5, 3.2, 6.7, 0.1, 0.1, 0.2])
			elif descriptor == 'af1hot+3d-dis':
				return np.array([1.9, 1.8, 3.7, 0.1, 0.1, 0.2])
			# The following 2 costs are only optimized used the jia2023 paper
			# for y_scaling "none" and for descriptor "atom_bond_types".
			# todo: optimize for other y_scaling.
			elif descriptor == 'atom_bond_types':
				return np.array([1.5, 2.0, 3.5, 2.7, 2.5, 5.2])
			elif descriptor == 'unlabeled':
				return np.array([1.5, 2.0, 3.5, 2.7, 2.5, 5.2])
			else:
				raise ValueError('Unknown descriptor: {}.'.format(descriptor))

		else:
			raise ValueError('Unknown dataset name: {}.'.format(ds_name))
	else:
		raise ValueError('Unknown GED method: {}.'.format(method))


