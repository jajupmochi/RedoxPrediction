import os
import pickle

import numpy as np


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
	if kwargs.get('deep_model') == 'none':
		if kwargs.get('embedding').startswith('gk:'):
			# 1. Train the graph kernel model:
			from redox_prediction.models.model_selection.kernel import \
				evaluate_graph_kernel
			return evaluate_graph_kernel(
				G_app, y_app, G_test, y_test, model_type=model_type,
				descriptor=descriptor, read_resu_from_file=read_resu_from_file,
				**kwargs
			)
		if kwargs.get('embedding').startswith('ged:'):
			# 2. Train the GED model:
			from redox_prediction.models.model_selection.ged import evaluate_ged
			return evaluate_ged(
				G_app, y_app, G_test, y_test, model_type=model_type,
				descriptor=descriptor, read_resu_from_file=read_resu_from_file,
				**kwargs
			)
		if kwargs.get('embedding').startswith('nn:'):
			# 3. Train the NN model:
			from redox_prediction.models.model_selection.gnn import evaluate_gnn
			return evaluate_gnn(
				G_app, y_app, G_test, y_test,
				model_type=model_type,
				descriptor=descriptor,
				read_resu_from_file=read_resu_from_file,
				**kwargs
			)
		else:
			raise ValueError(
				'Unknown embedding method: {0}'.format(
					kwargs.get('embedding')
				)
			)
	else:
		if kwargs.get('infer') == 'pretrain+refine':
			pass


# 1. Train the two-step GNN models for metric evaluation:


def xp_main(
		Gn,
		y_all,
		model_type: str = 'reg',
		unlabeled: bool = False,
		descriptor='atom_bond_types',
		output_file: str = None,
		read_resu_from_file: int = 1,
		**kwargs
):
	"""
	Perform a knn regressor on given dataset
	"""
	# Load existing results if possible.
	if read_resu_from_file and output_file is not None and os.path.isfile(
			output_file
	) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = []

	import time
	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
	from sklearn.model_selection import train_test_split

	start_time = time.time()

	stratified = False
	if model_type == 'classif':
		stratified = False  # debug

	if stratified:
		rs = StratifiedShuffleSplit(n_splits=10, test_size=.1, random_state=0)
	else:
		# 		rs = ShuffleSplit(n_splits=10, test_size=.1) #, random_state=0)
		rs = ShuffleSplit(n_splits=10, test_size=.1, random_state=0)

	if stratified:
		split_scheme = rs.split(Gn, y_all)
	else:
		split_scheme = rs.split(Gn)

	i = 1
	for app_index, test_index in split_scheme:
		# if i > 1:  # debug: for debug only.
		# 	break

		print()
		print('----- Split {0}/{1} -----'.format(i, 10))
		i = i + 1

		# Check if the split already computed.
		if i - 1 <= len(results):
			print('Skip.')
			continue

		# Get split data
		G_app, G_test, y_app, y_test = split_data(
			Gn, y_all, app_index, test_index
		)

		# Split evaluation set.
		valid_size = 0.1 / (1 - 0.1)
		# stratify = (F_app if stratified else None)
		op_splits = train_test_split(
			G_app, y_app, app_index,
			test_size=valid_size,
			random_state=0,  # @debug: to change.
			shuffle=True,
			stratify=None
		)
		G_train, G_valid, y_train, y_valid, train_index, valid_index = op_splits

		cur_results = evaluate_models(
			G_train, np.array(y_train),
			G_valid, np.array(y_valid),
			G_test, np.array(y_test),
			model_type=model_type, unlabeled=unlabeled,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			# fit_test=(optim_method == 'embed'),
			n_classes=(
				len(np.unique(y_all)) if model_type == 'classif' else None),
			**{**kwargs, 'split_idx': i - 1}
		)

		append_split_perf(results, cur_results)

		# Show mean and std of performance on app and test set until now:
		perf_util_now(results)

		# Save the split:
		save_split_to_file(results, read_resu_from_file, output_file, i)

	# if optim_method in ['embed', 'jia2021']:
	# 	# Compute distances between elements in embedded space:
	# 	res_embed = compute_D_embed(
	# 		G_app, np.array(y_app), G_test, np.array(y_test),
	# 		y_distance=y_distance,
	# 		model_type=model_type, unlabeled=unlabeled, ed_method=ed_method,
	# 		descriptor=descriptor,
	# 		embedding_space=embedding_space,
	# 		fit_test=(optim_method == 'embed'),
	# 		n_classes=(
	# 			len(np.unique(y_all)) if model_type == 'classif' else None),
	# 		**kwargs
	# 	)
	# else:
	# 	res_embed = None
	#
	# if optim_method == 'embed':
	# 	perf_app, perf_test, mat_app, mat_test, model = res_embed
	# 	save_perf_for_embed(
	# 		results, perf_app, perf_test,
	# 		y_app, y_test, mat_app, mat_test, model
	# 	)
	# else:
	# 	dis_mat_emded = res_embed
	# 	evaluate_setup(
	# 		G_app, y_app, G_test, y_test, dis_mat_emded, results,
	# 		i, descriptor, optim_method, ed_method, unlabeled, model_type,
	# 		y_distance,
	# 		**kwargs
	# 	)

	# Print run time:
	run_time = time.time() - start_time
	print()
	print('Total running time of all CV trials: %.2f seconds.' % run_time)
	print(
		'Attention: the running time might be inaccurate when some parts of the '
		'code are executed a priori. Please check the saved results for the '
		'actual running time.'
	)

	if len(results) == 10:
		# Calculate mean and standard deviation over splits:
		from redox_prediction.dataset.stats import calculate_stats, \
			save_results_to_csv
		stats = calculate_stats(results)
		results.append(stats)
		save_results_to_csv(results, kwargs['output_dir'])

	return results


# %%


def append_split_perf(results, results_to_save):
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
				'params_best'
			]
	):
		cur_results[key] = results_to_save[i]
	results.append(cur_results)


def perf_util_now(results):
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
			if i - 1 <= len(results_new):
				print('Skip.')
				return

	# If not, save the results.
	if output_file is not None:
		pickle.dump({'results': results}, open(output_file, 'wb'))
