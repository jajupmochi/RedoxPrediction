"""
kernels



@Author: linlin
@Date: 27.05.23
"""
import numpy as np


def _compute_D_graph_kernel(
		G_app, y_app, G_test, y_test, estimator, param_grid,
		param_grid_precomputed, mode, fit_test, **kwargs
):
	# Compute Gram matrix:
	from redox_prediction.models.model_selection.kernel2 import \
		model_selection_for_precomputed_kernel
	model, perf_app, perf_test, gram_app, gram_test, param_out, param_in = model_selection_for_precomputed_kernel(
		G_app, y_app, G_test, y_test,
		estimator,
		param_grid_precomputed,
		param_grid,
		mode,
		fit_test=fit_test,
		parallel=False,
		# n_jobs=1,
		read_gm_from_file=False,
		verbose=False,  # (True if len(G_app) > 1000 else False),
		**kwargs,
	)
	# Compute distances between elements in embedded space:
	if not fit_test:
		dis_mat = np.zeros((len(G_app), len(G_app)))
		for i in range(len(G_app)):
			for j in range(i + 1, len(G_app)):
				dis_mat[i, j] = np.sqrt(
					gram_app[i, i] + gram_app[j, j] - 2 * gram_app[i, j]
				)
				dis_mat[j, i] = dis_mat[i, j]

		return dis_mat
	else:
		return perf_app, perf_test, gram_app, gram_test, model[0], model[1]


def compute_D_shortest_path_kernel(
		G_app, y_app, G_test, y_test,
		y_distance=None,
		mode='reg', unlabeled=False, ed_method='bipartite',
		descriptor='atom_bond_types',
		fit_test=False,
		**kwargs
):
	"""
	Return the distance matrix computed by WL subtree kernel.
	"""
	from gklearn.kernels import ShortestPath

	# Get parameter grid:
	import functools
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	mix_kernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	param_grid_precomputed = {
		'node_kernels': [
			{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mix_kernel}]
	}
	param_grid = (
		{'alpha': np.logspace(-10, 10, num=21, base=10)} if mode == 'reg' else
		{'C': np.logspace(-10, 10, num=21, base=10)}
	)

	estimator = ShortestPath
	return _compute_D_graph_kernel(
		G_app, y_app, G_test, y_test, estimator, param_grid,
		param_grid_precomputed, mode, fit_test, **kwargs
	)


def compute_D_structural_sp_kernel(
		G_app, y_app, G_test, y_test,
		y_distance=None,
		mode='reg', unlabeled=False, ed_method='bipartite',
		descriptor='atom_bond_types',
		fit_test=False,
		**kwargs
):
	"""
	Return the distance matrix computed by structural shortest path kernel.
	"""
	from gklearn.kernels import StructuralSP

	# Get parameter grid:
	import functools
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	mix_kernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = [
		{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mix_kernel}
	]
	param_grid_precomputed = {
		'node_kernels': sub_kernels, 'edge_kernels': sub_kernels,
		'compute_method': ['naive']
	}
	param_grid = (
		{'alpha': np.logspace(-10, 10, num=21, base=10)} if mode == 'reg' else
		{'C': np.logspace(-10, 10, num=21, base=10)}
	)

	estimator = StructuralSP
	return _compute_D_graph_kernel(
		G_app, y_app, G_test, y_test, estimator, param_grid,
		param_grid_precomputed, mode, fit_test, **kwargs
	)


def compute_D_path_kernel(
		G_app, y_app, G_test, y_test,
		y_distance=None,
		mode='reg', unlabeled=False, ed_method='bipartite',
		descriptor='atom_bond_types',
		fit_test=False,
		**kwargs
):
	"""
	Return the distance matrix computed by path kernel.
	"""
	from gklearn.kernels import PathUpToH

	# Get parameter grid:
	param_grid_precomputed = {
		'depth': [1, 2, 3, 4, 5, 6],
		'k_func': ['MinMax', 'tanimoto'],
		'compute_method': ['trie']
	}
	param_grid = (
		{'alpha': np.logspace(-10, 10, num=21, base=10)} if mode == 'reg' else
		{'C': np.logspace(-10, 10, num=21, base=10)}
	)

	estimator = PathUpToH
	return _compute_D_graph_kernel(
		G_app, y_app, G_test, y_test, estimator, param_grid,
		param_grid_precomputed, mode, fit_test, **kwargs
	)


def compute_D_treelet_kernel(
		G_app, y_app, G_test, y_test,
		y_distance=None,
		mode='reg', unlabeled=False, ed_method='bipartite',
		descriptor='atom_bond_types',
		fit_test=False,
		**kwargs
):
	"""
	Return the distance matrix computed by treelet kernel.
	"""
	from gklearn.kernels import Treelet

	# Get parameter grid:
	import functools
	from gklearn.utils.kernels import gaussiankernel, polynomialkernel
	gkernels = [
		functools.partial(gaussiankernel, gamma=1 / ga)
		#            for ga in np.linspace(1, 10, 10)]
		for ga in np.logspace(1, 10, num=10, base=10)
		# debug: change it as needed
	]
	pkernels = [
		functools.partial(polynomialkernel, d=d, c=c) for d in range(1, 5)
		for c in np.logspace(0, 10, num=11, base=10)
	]
	param_grid_precomputed = {'sub_kernel': gkernels + pkernels}
	param_grid = (
		{'alpha': np.logspace(-10, 10, num=21, base=10)} if mode == 'reg' else
		{'C': np.logspace(-10, 10, num=21, base=10)}
	)

	estimator = Treelet
	return _compute_D_graph_kernel(
		G_app, y_app, G_test, y_test, estimator, param_grid,
		param_grid_precomputed, mode, fit_test, **kwargs
	)


def compute_D_wlsubtree_kernel(
		G_app, y_app, G_test, y_test,
		y_distance=None,
		mode='reg', unlabeled=False, ed_method='bipartite',
		descriptor='atom_bond_types',
		fit_test=False,
		**kwargs
):
	"""
	Return the distance matrix computed by WL subtree kernel.
	"""
	from gklearn.kernels import WLSubtree

	# Get parameter grid:
	param_grid_precomputed = {'height': [0, 1, 2, 3, 4, 5, 6]}
	param_grid = (
		{'alpha': np.logspace(-10, 10, num=21, base=10)} if mode == 'reg' else
		{'C': np.logspace(-10, 10, num=21, base=10)}
	)

	estimator = WLSubtree
	return _compute_D_graph_kernel(
		G_app, y_app, G_test, y_test, estimator, param_grid,
		param_grid_precomputed, mode, fit_test, **kwargs
	)


import sklearn


def estimator_to_dict(
		estimator: sklearn.base.BaseEstimator,
) -> dict:
	"""
	Convert a `sklearn` estimator object to a dict.

	Parameters
	----------
	estimator: sklearn.base.BaseEstimator
		The estimator to convert.

	Returns
	-------
	dict
		The converted estimator.
	"""
	dict_est = {
		'__estimator__': True,
		'module': estimator.__module__,
		'name': estimator.__class__.__name__,
		'params': estimator.get_params(),
	}
	return dict_est
