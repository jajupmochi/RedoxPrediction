import os
import sys
import pickle

import time

from joblib import Parallel, delayed

import numpy as np

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from redox_prediction.utils.logging import PrintLogger


def split_data(D, y, train_index, test_index):
	D_app = [D[i] for i in train_index]
	D_test = [D[i] for i in test_index]
	y_app = [y[i] for i in train_index]
	y_test = [y[i] for i in test_index]
	return D_app, D_test, y_app, y_test


def evaluate_models(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		model_type='reg', unlabeled=False,
		descriptor='atom_bond_types',
		read_resu_from_file=1,
		# fit_test=False,
		**kwargs
):
	if kwargs.get('model').startswith('nn'):
		from redox_prediction.models.model_selection.gnn import evaluate_gnn
		return evaluate_gnn(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	# graph kernels:
	elif kwargs.get('model').startswith('gk'):
		from redox_prediction.models.model_selection.kernel import \
			evaluate_graph_kernel
		return evaluate_graph_kernel(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	# GEDs:
	elif kwargs.get('model').startswith('ged:'):
		from redox_prediction.models.model_selection.ged import evaluate_ged
		return evaluate_ged(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	elif kwargs.get('model').startswith('vc:'):
		from redox_prediction.models.model_selection.vector_model import evaluate_vector_model
		return evaluate_vector_model(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	else:
		raise ValueError(
			'Unknown model: {0}.'.format(kwargs.get('model'))
		)


def process_split(
		i, app_index, test_index, G_all, y_all, model_type, unlabeled,
		descriptor, read_resu_from_file, kwargs
):
	"""
	Process a single split.
	"""
	logging_file = os.path.join(kwargs.get('output_dir'), 'split_%d.log' % (i + 1))
	sys.stdout = PrintLogger(logging_file)

	print()
	print('----- Split {0}/{1} -----'.format(i + 1, 10))

	start = time.time()

	# Get split data
	G_app, G_test, y_app, y_test = split_data(
		G_all, y_all, app_index, test_index
	)

	# Split evaluation set
	valid_size = 0.1 / (1 - 0.1)
	op_splits = train_test_split(
		G_app, y_app, app_index,
		test_size=valid_size,
		random_state=0,
		shuffle=True,
		stratify=None
	)
	G_train, G_valid, y_train, y_valid, _, _ = op_splits

	cur_results = evaluate_models(
		G_train, np.array(y_train),
		G_valid, np.array(y_valid),
		G_test, np.array(y_test),
		model_type=model_type, unlabeled=unlabeled,
		descriptor=descriptor,
		read_resu_from_file=read_resu_from_file,
		n_classes=len(np.unique(y_all)) if model_type == 'classif' else None,
		**{**kwargs, 'split_idx': i + 1}
	)

	run_time = time.time() - start
	cur_results = cur_results + (run_time,)

	return i, cur_results


def xp_main(
		Gn,
		y_all,
		model_type: str = 'reg',
		unlabeled: bool = False,
		descriptor='atom_bond_types',
		output_file: str = None,
		read_resu_from_file: int = 1,
		parallel: bool = False,
		**kwargs
):
	"""
	Perform a knn regressor on given dataset
	"""
	n_splits = 10
	stratified = False
	if model_type == 'classif':
		stratified = False  # debug

	if stratified:
		rs = StratifiedShuffleSplit(
			n_splits=n_splits, test_size=.1, random_state=0
		)
	else:
		# 		rs = ShuffleSplit(n_splits=10, test_size=.1) #, random_state=0)
		rs = ShuffleSplit(n_splits=n_splits, test_size=.1, random_state=0)

	if stratified:
		split_scheme = rs.split(Gn, y_all)
	else:
		split_scheme = rs.split(Gn)

	# Load existing results if possible.
	if read_resu_from_file and output_file is not None and os.path.isfile(
			output_file
	) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = [None] * n_splits

	if parallel:
		# todo: skip existing results.

		n_jobs = n_splits + 1

		results = Parallel(n_jobs=n_jobs)(
			delayed(process_split)(
				i, app_index, test_index, Gn, y_all, model_type, unlabeled,
				descriptor, read_resu_from_file, kwargs
			) for i, (app_index, test_index) in enumerate(split_scheme)
		)

		# Sort results by split index:
		sorted_results = sorted(results, key=lambda x: x[0])

		logging_file = os.path.join(kwargs['output_dir'], 'output.log')
		sys.stdout = PrintLogger(logging_file)

		# Process the sorted results
		for i, cur_results in sorted_results:
			append_split_perf(results, i, cur_results)
			# perf_util_now(results)
			save_split_to_file(results, read_resu_from_file, output_file, i)

	else:
		for i, (app_index, test_index) in enumerate(split_scheme):
			# if i > 0:  # debug: for debug only.
			# 	break

			# Check if the split already computed.
			if results[i] is not None:
				print()
				print('----- Split {0}/{1} -----'.format(i + 1, 10))
				print('Skip.')
				continue

			i, cur_results = process_split(
				i, app_index, test_index, Gn, y_all, model_type, unlabeled,
				descriptor, read_resu_from_file, kwargs
			)

			append_split_perf(results, i, cur_results)

			# # Show mean and std of performance on app and test set until now:
			perf_util_now(results)

			# Save the split:
			save_split_to_file(results, read_resu_from_file, output_file, i)

	# Print run time:
	total_run_time = np.sum(
		[res['split_total_run_time'] for res in results[:-1]]
	)
	print()
	print('Total running time of all CV trials: %.2f seconds.' % total_run_time)
	# print(
	# 	'Attention: the running time might be inaccurate when some parts of the '
	# 	'code are executed a priori. Please check the saved results for the '
	# 	'actual running time.'
	# )

	if len(results) == 10:
		# Calculate mean and standard deviation over splits:
		from redox_prediction.dataset.stats import calculate_stats, \
			save_results_to_csv
		stats = calculate_stats(results)
		results.append(stats)
		save_results_to_csv(results, kwargs['output_dir'])

	return results


# %%


def append_split_perf(results, split_idx, results_to_save):
	"""
	Save performance for split into results.
	"""
	cur_results = {}
	for i, key in enumerate(
			[
				'model',
				'perf_train', 'perf_valid', 'perf_test',
				'y_pred_train', 'y_pred_valid', 'y_pred_test',
				'best_history', 'history_train', 'history_valid',
				'history_test',
				'params_best',
				'split_total_run_time',
			]
	):
		cur_results[key] = results_to_save[i]
	results[split_idx] = cur_results


def perf_util_now(results):
	results = [res for res in results if res is not None]
	print()
	print(
		'performance until now:\t'
		'Train:\t{:0.3f} $\\pm$ {:0.3f}\t'
		'Valid:\t{:0.3f} $\\pm$ {:0.3f}\t'
		'Test:\t{:0.3f} $\\pm$ {:0.3f}'.format(
			np.mean([results[i]['perf_train'] for i in range(len(results))]),
			np.std([results[i]['perf_train'] for i in range(len(results))]),
			np.mean([results[i]['perf_valid'] for i in range(len(results))]),
			np.std([results[i]['perf_valid'] for i in range(len(results))]),
			np.mean([results[i]['perf_test'] for i in range(len(results))]),
			np.std([results[i]['perf_test'] for i in range(len(results))]),
		)
	)


def save_split_to_file(results, read_resu_from_file, output_file, i):
	# Check if the (possible) existing file was updated by another thread
	# during the computation of this split:
	if read_resu_from_file and output_file is not None and os.path.isfile(
			output_file
	) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results_new = pickle.load(file)['results']
			if results_new[i] is not None:
				print('Skip.')
				return

	# If not, save the results.
	if output_file is not None:
		pickle.dump({'results': results}, open(output_file, 'wb'))
