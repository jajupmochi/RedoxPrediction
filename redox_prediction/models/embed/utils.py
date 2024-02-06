"""
utils



@Author: linlin
@Date: 26.07.23
"""
import os

import pickle

from typing import List

import networkx

from redox_prediction.utils.logging import AverageMeter
from redox_prediction.utils.graph import reorder_graphs_by_index


def get_entire_metric_matrix(
		graphs: List[networkx.Graph],
		estimator: object,
		metric_params: dict = None,
		parallel: bool = None,
		n_jobs: int = None,
		chunksize: int = None,
		copy_graphs: bool = True,
		load_metric_from_file: bool = False,
		output_dir: str = None,
		params_idx: int = None,
		reorder_graphs: bool = True,
		verbose: int = 2,
		**kwargs
):
	if load_metric_from_file:
		fn_model = os.path.join(
			output_dir, 'metric_model.params_{}.pkl'.format(
				str(params_idx)
			)
		)
		# Load model from file if it exists:
		if os.path.exists(fn_model) and os.path.getsize(fn_model) > 0:
			print('\nLoading model from file...')
			resu = pickle.load(open(fn_model, 'rb'))
			return resu['model'], resu['run_time'], resu[
				'model'].metric_matrix

	# Reorder graphs if specified:
	if reorder_graphs:
		graphs = reorder_graphs_by_index(graphs, idx_key='id')

	# Compute metric matrix otherwise:
	print('Computing metric matrix...')
	model = estimator(
		node_labels=kwargs['node_labels'],
		edge_labels=kwargs['edge_labels'],
		node_attrs=kwargs['node_attrs'],
		edge_attrs=kwargs['edge_attrs'],
		ds_infos={'directed': False},
		parallel=parallel,
		n_jobs=n_jobs,
		chunksize=None,
		normalize=True,
		copy_graphs=True,  # make sure it is a full deep copy. and faster!
		verbose=verbose,
		**metric_params
	)

	# Train model.
	try:
		# Save metric matrix so that we can load it from file later:
		metric_matrix = model.fit_transform(
			graphs, save_mm_train=True
		)
	except FloatingPointError as e:
		print(
			'Encountered FloatingPointError while fitting the model with '
			'the parameters below:' + '\n' + str(metric_params)
		)
		print(e)
		raise e

	# Save history:
	n_pairs = model.n_pairs
	run_time = AverageMeter()
	run_time.update(model.run_time / n_pairs, n_pairs)

	# Save model and history to file:
	if load_metric_from_file:
		os.makedirs(os.path.dirname(fn_model), exist_ok=True)
		pickle.dump({'model': model, 'run_time': run_time}, open(fn_model, 'wb'))

	# Print out the information:
	if verbose:
		print(
			'Computed metric matrix of size {} in {:.3f} / {:.9f} seconds for '
			'parameters # {}.'.format(
				metric_matrix.shape, model.run_time, model.run_time / n_pairs,
				params_idx
			)
		)

	return model, run_time, metric_matrix


def get_metrics(
		graphs1,
		graphs2,
		embedding='ged:bp_random',
		**kwargs
):
	if embedding == 'ged:bp_random':
		pass

	return metrics
