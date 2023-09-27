"""
embed



@Author: linlin
@Date: 19.05.23
"""
import time

import numpy as np

from redox_prediction.utils.logging import AverageMeter


# %% ----- the target space (distance matrix): -----

def compute_D_y(
		G_app, y_app, G_test, y_test,
		y_distance=None,
		mode='reg', unlabeled=False, ed_method='bipartite',
		descriptor='atom_bond_types',
		**kwargs
):
	"""
	Return the distance matrix directly computed by y.
	"""
	N = len(y_app)

	dis_mat = np.zeros((N, N))

	for i in range(N):
		for j in range(i + 1, N):
			dis_mat[i, j] = y_distance(y_app[i], y_app[j])
			dis_mat[j, i] = dis_mat[i, j]

	return dis_mat


# %% ----- the embedding spaces (distance matrix): -----


def compute_D_embed(
		G_app, y_app, G_test, y_test,
		y_distance=None,
		mode='reg', unlabeled=False, ed_method='bipartite',
		descriptor='atom_bond_types',
		embedding_space='y',
		fit_test=False,
		**kwargs
):
	"""
	Evaluate the distance matrix between elements in the embedded space.
	"""
	if embedding_space == 'y':
		# Compute distances between elements in embedded space:
		y_dis_mat = compute_D_y(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
		)
		return y_dis_mat
	# ---- kernel spaces: -----
	elif embedding_space == 'sp_kernel':
		from .kernel import compute_D_shortest_path_kernel
		return compute_D_shortest_path_kernel(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
			**kwargs
		)
	elif embedding_space == 'structural_sp':
		from .kernel import compute_D_structural_sp_kernel
		return compute_D_structural_sp_kernel(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
			**kwargs
		)
	elif embedding_space == 'path_kernel':
		from .kernel import compute_D_path_kernel
		return compute_D_path_kernel(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
			**kwargs
		)
	elif embedding_space == 'treelet_kernel':
		from .kernel import compute_D_treelet_kernel
		return compute_D_treelet_kernel(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
			**kwargs
		)
	elif embedding_space == 'wlsubtree_kernel':
		from .kernel import compute_D_wlsubtree_kernel
		return compute_D_wlsubtree_kernel(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
			**kwargs
		)
	# ---- GNN spaces: -----
	elif embedding_space == 'gcn':
		from .gnn import compute_D_gcn
		return compute_D_gcn(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
			model_name=embedding_space,
			**kwargs
		)
	elif embedding_space == 'gat':
		from .gnn import compute_D_gat
		return compute_D_gat(
			G_app, y_app, G_test, y_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			descriptor=descriptor,
			fit_test=fit_test,
			model_name=embedding_space,
			**kwargs
		)
	else:
		raise ValueError(
			'Unknown embedding space: {}'.format(embedding_space)
		)


# %% ----- the embedding spaces (pairwise metrics): -----


def get_metrics_embed(
		graphs1,
		graphs2,
		embedding=None,
		metric_matrix=np.empty((0, 0)),
		**kwargs
):
	"""
	Compute the pairwise metrics between two sets of graphs.
	"""
	# Initialize the time meters:
	time_meters = {key: AverageMeter() for key in ['total', 'load', 'compute']}
	time_meters['total'].tic()

	metrics = []
	for g1, g2 in zip(graphs1, graphs2):
		start_time = time.time()

		idx1, idx2 = g1.graph['id'], g2.graph['id']  # @TODO: check
		if idx1 > idx2:
			idx1, idx2 = idx2, idx1

		if np.isnan(metric_matrix[idx1, idx2]):
			metric = compute_metric_embed(g1, g2, embedding, **kwargs)
			metric_matrix[idx1][idx2] = metric
			time_meters['compute'].update(time.time() - start_time)

		else:
			metric = metric_matrix[idx1][idx2]
			time_meters['load'].update(time.time() - start_time)

		metrics.append(metric)

	# Update the time meters:
	time_meters['total'].toc()

	return metrics, time_meters


def compute_metric_embed(
		G1, G2,
		embedding,
		**kwargs
):
	if embedding == 'ged:bp_random':
		from .ged import compute_ged
		edit_costs = np.random.rand(6)
		return compute_ged(
			G1, G2, edit_costs, edit_cost_fun='CONSTANT', method='BIPARTITE',
			repeats=1, return_nb_eo=False, **kwargs
		)

	elif embedding == 'ged:bp_expert':
		from .ged import compute_metric_bp_expert
		return compute_metric_bp_expert(G1, G2, **kwargs)
	else:
		raise ValueError(
			'Unknown embedding space: {}'.format(embedding)
		)
