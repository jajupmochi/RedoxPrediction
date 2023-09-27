"""
ged



@Author: linlin
@Date: 18.08.23
"""
import os
import pickle
from typing import List

import networkx

from redox_prediction.utils.logging import AverageMeter


def fit_model_ged(
		G_app: List[networkx.Graph],
		ged_options: dict = None,
		parallel: bool = None,
		n_jobs: int = None,
		chunksize: int = None,
		copy_graphs: bool = True,
		read_resu_from_file: int = 1,
		output_dir: str = None,
		split_idx: str = None,
		params_idx: str = None,
		verbose: int = 2,
		**kwargs
):
	if read_resu_from_file >= 2:
		fn_model = os.path.join(
			output_dir, 'metric_model.split_{}.params_{}.pkl'.format(
				split_idx, params_idx
			)
		)
		# Load model from file if it exists:
		if os.path.exists(fn_model) and os.path.getsize(fn_model) > 0:
			print('\nLoading model from file...')
			resu = pickle.load(open(fn_model, 'rb'))
			return resu['model'], resu['history'], resu['model'].dis_matrix

	# Compute metric matrix otherwise:
	print('Computing metric matrix...')
	nl_names = list(G_app[0].nodes[list(G_app[0].nodes)[0]].keys())
	el_names = list(G_app[0].edges[list(G_app[0].edges)[0]].keys())
	from gklearn.ged import GEDModel
	model = GEDModel(
		ed_method=ged_options['method'],
		edit_cost_fun=ged_options['edit_cost_fun'],
		init_edit_cost_constants=ged_options['edit_costs'],
		optim_method=ged_options['optim_method'],
		node_labels=nl_names, edge_labels=el_names,
		parallel=parallel,
		n_jobs=n_jobs,
		chunksize=chunksize,
		copy_graphs=copy_graphs,
		# make sure it is a full deep copy. and faster!
		verbose=False
	)

	# Train model.
	try:
		dis_mat_train = model.fit_transform(
			G_app, save_dm_train=True, repeats=ged_options['repeats'],
		)
	except OSError as exception:
		if 'GLIBC_2.23' in exception.args[0]:
			msg = \
				'This error is very likely due to the low version of GLIBC ' \
				'on your system. ' \
				'The required version of GLIBC is 2.23. This may happen on the ' \
				'CentOS 7 system, where the highest version of GLIBC is 2.17. ' \
				'You may check your CLIBC version by bash command `rpm -q glibc`. ' \
				'The `graphkit-learn` library comes with GLIBC_2.23, which you can ' \
				'install by enable the `--build-gedlib` option: ' \
				'`python3 setup.py install --build-gedlib`. This will compile the C++ ' \
				'module `gedlib`, which requires a C++ compiler and CMake.'
			raise AssertionError(msg) from exception
		else:
			assert False, exception
	except Exception as exception:
		assert False, exception

	# Save history:
	# For graph kernels it is n * (n - 1) / 2:
	n_pairs = len(G_app) * (len(G_app) - 1) / 2
	history = {'run_time': AverageMeter()}
	history['run_time'].update(model.run_time / n_pairs, n_pairs)

	# Save model and history to file:
	if read_resu_from_file >= 2:
		os.makedirs(os.path.dirname(fn_model), exist_ok=True)
		pickle.dump({'model': model, 'history': history}, open(fn_model, 'wb'))

	# Print out the information:
	print(
		'Computed metric matrix of size {} in {:.3f} / {:.9f} seconds for parameters'
		' # {}.'.format(
			dis_mat_train.shape, model.run_time, model.run_time / n_pairs,
			params_idx
		)
	)

	return model, history, dis_mat_train
