"""
graph kernel



@Author: linlin
@Date: 18.08.23
"""
import os
import pickle
from typing import List

import networkx

from redox_prediction.utils.logging import AverageMeter
from redox_prediction.utils.graph import reorder_graphs_by_index


def fit_model_kernel(
		graphs: List[networkx.Graph],
		estimator,
		kernel_options: dict = None,
		parallel: bool = None,
		n_jobs: int = None,
		chunksize: int = None,
		copy_graphs: bool = True,
		read_resu_from_file: int = 1,
		output_dir: str = None,
		params_idx: str = None,
		reorder_graphs: bool = False,
		verbose: int = 2,
		**kwargs
):
	if read_resu_from_file >= 1:
		fn_model = os.path.join(
			output_dir, 'metric_model.params_{}.pkl'.format(
				params_idx
			)
		)
		# Load model from file if it exists:
		if os.path.exists(fn_model) and os.path.getsize(fn_model) > 0:
			print('\nLoading model from file...')
			resu = pickle.load(open(fn_model, 'rb'))
			return resu['model'], resu['history'], resu['model'].gram_matrix

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
		**kernel_options
	)

	# Train model.
	try:
		# Save metric matrix so that we can load it from file later:
		matrix = model.fit_transform(graphs, save_gm_train=True)
	except FloatingPointError as e:
		print(
			'Encountered FloatingPointError while fitting the model with '
			'the parameters below:' + '\n' + str(kernel_options)
		)
		print(e)
		raise e

	# Save history:
	n_pairs = len(graphs) * (
				len(graphs) + 1) / 2  # For GEDs it is n * (n - 1) / 2.
	history = {'run_time': AverageMeter()}
	history['run_time'].update(model.run_time / n_pairs, n_pairs)

	# Save model and history to file:
	if read_resu_from_file >= 1:
		os.makedirs(os.path.dirname(fn_model), exist_ok=True)
		pickle.dump({'model': model, 'history': history}, open(fn_model, 'wb'))

	# Print out the information:
	print(
		'Computed metric matrix of size {} in {:.3f} / {:.9f} seconds for parameters'
		' # {}.'.format(
			matrix.shape, model.run_time, model.run_time / n_pairs,
			params_idx
		)
	)

	return model, history, matrix
