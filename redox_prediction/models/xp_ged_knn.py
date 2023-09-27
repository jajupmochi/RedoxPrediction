#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:35:46 2022

@author: ljia
"""
import os
import sys
sys.path.insert(0, '../../')
import pickle
import numpy as np
import functools
from dataset.load_dataset import get_data
from redox_prediction.models import split_data

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from gklearn.ged import GEDModel
from gklearn.utils import get_iters


path_kw = '/ged_knn/std_y/mae/'


#%% Simple test run


def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test,
				   scale_x=False,
				   scale_y=(True if 'std_y' in path_kw else False), #@todo
				   **kwargs):

	trial_index = kwargs.get('trial_index')
	use_best_params = kwargs.get('use_best_params', False) # for the convenience of testing

	# Normalize targets.
	if scale_y:
		from sklearn.preprocessing import StandardScaler
		y_scaler = StandardScaler().fit(np.reshape(y_train, (-1, 1)))
		y_train = y_scaler.transform(np.reshape(y_train, (-1, 1)))
		y_valid = y_scaler.transform(np.reshape(y_valid, (-1, 1)))
		y_test = y_scaler.transform(np.reshape(y_test, (-1, 1)))
	y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)

	# Train and predict.
	# hyper-params:
	from sklearn.model_selection import ParameterGrid
	if use_best_params:
		hparams = list(ParameterGrid({
			'n_neighbors': [7]
			}))
	else:
		hparams = list(ParameterGrid({
			'n_neighbors': [1, 3, 5, 7, 9, 11]
			}))


	MAE_min = np.inf
	MAE_all = []
	for params in get_iters(hparams, desc='Hyperparam tuning'):
		# Initialize model.
		nl_names = list(X_train[0].nodes[list(X_train[0].nodes)[0]].keys())
		el_names = list(X_train[0].edges[list(X_train[0].edges)[0]].keys())
		ged_model = GEDModel(
			  ed_method=kwargs.get('ed_method'),
			  edit_cost_fun='CONSTANT',
			  init_edit_cost_constants=[3, 3, 1, 3, 3, 1],
			  optim_method=kwargs.get('optim_method'),
#			  optim_options={'y_distance': euclid_d, 'mode': 'reg'},
			  node_labels=nl_names, edge_labels=el_names,
			  parallel=None,
			  n_jobs=None,
			  chunksize=None,
#			  normalize=True,
			  copy_graphs=True, # make sure it is a full deep copy. and faster!
			  verbose=(0 if kwargs.get('ed_method') == 'BIPARTITE' else 2)
			  )
		model = KNeighborsRegressor(
			metric='precomputed', n_neighbors=params['n_neighbors'])

		# train the model by train set
		dis_mat_train = ged_model.fit_transform(X_train, y_train, save_dm_train=True)
		model.fit(dis_mat_train, y_train)

		# Record result if better.
		dis_mat_valid = ged_model.transform(X_valid)
		y_pred_valid = model.predict(dis_mat_valid)
		if scale_y:
			y_tmp = np.ravel(y_scaler.inverse_transform(y_valid.reshape(-1, 1)))
			y_pred_tmp = np.ravel(y_scaler.inverse_transform(y_pred_valid.reshape(-1, 1)))
			MAE_cur = mean_absolute_error(y_tmp, y_pred_tmp)
		else:
			MAE_cur = mean_absolute_error(y_valid, y_pred_valid)
		MAE_all.append([params, MAE_cur])
		print('\n%s: %.4f.' % (str(params), MAE_cur))
		if MAE_cur < MAE_min:
			MAE_min = MAE_cur
			best_res = [MAE_min, params, (ged_model, model)]

	print('\nbest hyperparams: %s' % str(best_res[1]))
	graph_kernel, model = best_res[2][0], best_res[2][1]
	bparams = best_res[1]

	# Predict.
	dis_mat_train = graph_kernel.transform(return_dm_train=True)
	y_pred_train = model.predict(dis_mat_train)
	dis_mat_valid = graph_kernel.transform(X_valid)
	y_pred_valid = model.predict(dis_mat_valid)
	dis_mat_test = graph_kernel.transform(X_test)
	y_pred_test = model.predict(dis_mat_test)
	if scale_y:
		y_train = np.ravel(y_scaler.inverse_transform(y_train.reshape(-1, 1)))
		y_valid = np.ravel(y_scaler.inverse_transform(y_valid.reshape(-1, 1)))
		y_test = np.ravel(y_scaler.inverse_transform(y_test.reshape(-1, 1)))
		y_pred_train = np.ravel(y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)))
		y_pred_valid = np.ravel(y_scaler.inverse_transform(y_pred_valid.reshape(-1, 1)))
		y_pred_test = np.ravel(y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)))

	# Evaluate.
	train_accuracy, valid_accuracy, test_accuracy = {}, {}, {}
	for metric in [mean_absolute_error, r2_score]: #mean_absolute_percentage_error
		train_accuracy[metric.__name__] = metric(y_train, y_pred_train)
		valid_accuracy[metric.__name__] = metric(y_valid, y_pred_valid)
		test_accuracy[metric.__name__] = metric(y_test, y_pred_test)
#	from sklearn.metrics import mean_absolute_error
#	train_accuracy = mean_absolute_error(y_train, y_pred_train)
#	valid_accuracy = mean_absolute_error(y_valid, y_pred_valid)
#	test_accuracy = mean_absolute_error(y_test, y_pred_test)


	### Save predictions.
	fn = '../outputs/' + path_kw + '/y_values.t' + str(trial_index) + '.pkl'
	pickle.dump({'y_train': y_train, 'y_pred_train': y_pred_train,
			  'y_valid': y_valid, 'y_pred_valid': y_pred_valid,
			  'y_test': y_test, 'y_pred_test': y_pred_test,},
			 open(fn, 'wb'))
	fn = '../outputs/' + path_kw + 'mae_all.t' + str(trial_index) + '.pkl'
	pickle.dump(MAE_all, open(fn, 'wb'))


	return train_accuracy, valid_accuracy, test_accuracy, bparams, (graph_kernel, model)


def cross_validate(X, targets, families=None,
				   n_splits=30,
				   stratified=True, # @todo
				   output_file=None,
				   load_exist_results=False, # @todo
				   **kwargs):
	"""Run expriment.
	"""
	# output_file can not be set in the argument directly, as it contains
	# `path_kw`, which is modified in the `__main__`.
	if output_file is None:
		output_file='../outputs/' + path_kw + '/results.pkl'

	### Load existing results if possible.
	if load_exist_results and output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = []
#	results = []

	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, train_test_split
	from gklearn.model_selection import RepeatedKFoldWithValid

#	if mode == 'classif':
#		stratified = True # @todo: to change it as needed.
#	else:
#		stratified = kwargs.get('stratified', True)

	cv = '811' # kwargs.get('cv')
	test_size = (0.1 if cv == '811' else 0.2)

#	import collections
#	if np.ceil(test_size * len(X)) < len(collections.Counter(families)):
#		stratified = False # ValueError: The test_size should be greater or equal to the number of classes.
#		# @todo: at least let each data in test set has different family?

	if stratified:
		# @todo: to guarantee that at least one sample is in train set (, valid set and test set).
		if '/kfold' in path_kw:
			rs = RepeatedKFoldWithValid(n_splits=10, n_repeats=10, stratify=True, random_state=0) # @todo: to change it back.
		else:
			rs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
							   random_state=0) # @todo: to change it back.
		split_scheme = rs.split(X, families)
	else:
		if '/kfold' in path_kw:
			rs = RepeatedKFoldWithValid(n_splits=10, n_repeats=int(n_splits/10), stratify=False, random_state=0) # @todo: to change it back.
		else:
#		rs = ShuffleSplit(n_splits=n_splits, test_size=.1) #, random_state=0)
			rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0) # @todo: to change it back.
		split_scheme = rs.split(X)


	i = 1
	for all_index in split_scheme:
		print()
		print("Split {0}/{1}".format(i, n_splits))
		i = i + 1

		# Check if the split already computed.
		if i - 1 <= len(results):
			print('Skip.')
			continue

		# Get splitted data
		if '/kfold' in path_kw:
			train_index, valid_index, test_index = all_index
			op_splits = split_data(
				(X, targets), train_index, valid_index, test_index)
			X_train, X_valid, X_test, y_train, y_valid, y_test = op_splits
		else:
			app_index, test_index = all_index
			if stratified:
				X_app, X_test, y_app, y_test, F_app, F_tests = split_data(
					(X, targets, families), app_index, test_index)
			else:
				X_app, X_test, y_app, y_test = split_data(
					(X, targets), app_index, test_index)

			# Split evaluation set.
			valid_size = test_size / (1 - test_size)
			stratify = (F_app if stratified else None)
			op_splits = train_test_split(
				X_app, y_app, app_index,
				test_size=valid_size,
				random_state=0, # @todo: to change it back.
				shuffle=True,
				stratify=stratify)
			X_train, X_valid, y_train, y_valid, train_index, valid_index = op_splits


		### Save indices.
		fn = '../outputs/' + path_kw + '/indices_splits.t' + str(i-1) + '.pkl'
		os.makedirs(os.path.dirname(fn), exist_ok=True)
		pickle.dump({'train_index': train_index, 'valid_index': valid_index,
				  'test_index': test_index,},
				 open(fn, 'wb'))

		cur_results = {}
		cur_results['y_train'] = y_train
		cur_results['y_valid'] = y_valid
		cur_results['y_test'] = y_test
		cur_results['train_index'] = train_index
		cur_results['valid_index'] = valid_index
		cur_results['test_index'] = test_index

		kwargs['nb_trial'] = i - 1
#		for setup in distances.keys():
#		print("{0} Mode".format(setup))
		setup_results = {}


		# directory to save all results.
		output_dir = '.'.join(output_file.split('.')[:-1])
		os.makedirs(output_dir, exist_ok=True)
		perf_train, perf_valid, perf_test, bparams, model = evaluate_model(
			X_train, y_train, X_valid, y_valid, X_test, y_test,
			output_dir=output_dir, trial_index=i-1, **kwargs)

#		setup_results['perf_app'] = perf_app
		setup_results['perf_train'] = perf_train
		setup_results['perf_valid'] = perf_valid
		setup_results['perf_test'] = perf_test
		setup_results['best_params'] = bparams
		# Save model.
		fn_model = '../outputs/' + path_kw + '/model.t' + str(i - 1)
		with open(fn_model, 'wb') as f:
			pickle.dump(model, f)
		setup_results['model'] = fn_model


#		### Show model stucture.
#		input_ = tf.keras.Input(shape=(100,), dtype='int32', name='input')
#		model_for_plot = tf.keras.Model(inputs=[input_],
#								   outputs=[model.predict(input_)])
#		keras.utils.plot_model(model_for_plot, '' + path_kw + '_fcn_model.png',
#						  show_shapes=True)

		print()
		print('Performance of current trial:')
		print("train: {0}.".format(perf_train))
		print("valid: {0}.".format(perf_valid))
		print("test: {0}.".format(perf_test))
		cur_results = setup_results
		results.append(cur_results)


		### Save the split.

		# Check if the (possible) existing file was updated by another thread during the computation of this split.
		if load_exist_results and output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
			with open(output_file, 'rb') as file:
				results_new = pickle.load(file)['results']
				if i - 1 <= len(results_new):
					print('Skip.')
					continue

		# If not, save the results.
		if output_file is not None:
			pickle.dump({'results': results}, open(output_file, 'wb'))


		perf_tn = compute_final_perf(results)['mean_absolute_error']
		ave_train, std_train = perf_tn['train'][-2], perf_tn['train'][-1]
		ave_valid, std_valid = perf_tn['valid'][-2], perf_tn['valid'][-1]
		ave_test, std_test = perf_tn['test'][-2], perf_tn['test'][-1]
		print('Average performance (MAE) over trials until now:')
		print('train: %.4f+/-%.4f, valid: %.4f+/-%.4f, test: %.4f+/-%.4f.' % (ave_train, std_train, ave_valid, std_valid, ave_test, std_test))


	### Compute final performance.
	perfs = compute_final_perf(results)
	from pprint import pprint
	pprint(perfs)
	if output_file is not None:
		pickle.dump({'results': results, 'perfs': perfs}, open(output_file, 'wb'))


	return results


def compute_final_perf(results):
	perfs = {}
	metrics = list(results[0]['perf_train'].keys())
	for metric in metrics:
		cur_perfs = {'train': [], 'valid': [], 'test': []}
		for res in results:
			cur_perfs['train'].append(res['perf_train'][metric])
			cur_perfs['valid'].append(res['perf_valid'][metric])
			cur_perfs['test'].append(res['perf_test'][metric])
		for key, val in cur_perfs.items():
			mean = np.mean(val)
			std = np.std(val)
			cur_perfs[key] += [mean, std]
		perfs[metric] = cur_perfs
	return perfs


#%%


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('--ds_name', type=str, help='the name of dataset')
	parser.add_argument('--kfold_test', type=str, default='false', help='whether to use kfold test with a given best set of parameters. The default is "false".')

	# for GED methods only
	parser.add_argument('--ed_method', type=str, default='BIPARTITE', help='the approximation method to compute edit distances. for GED methods only.')
	parser.add_argument('--optim_method', type=str, default='random', help='the method to optimze the edit costs. for GED methods only.')
	parser.add_argument('--descriptor', type=str, default='atom_bond_types', help='the descriptor of the data, including "atom_bond_types" and "1hot".')

	# for poly200r dataset only
	parser.add_argument('--ds_version', type=str, default='latest', help='the version of the dataset. for poly200r only.')

	# for redox dataset only
	parser.add_argument('--tgt_name', type=str, default='dGred', help='the name of the targets. for Redox datasets only.')

	args = parser.parse_args()

	return args


if __name__ == '__main__':

#	test_losses()

	# Set parameters.
	args = parse_args()
	DS_Name_List = (['brem_togn'] if args.ds_name is None else [args.ds_name]) # ['poly200+sugarmono']:
	kfold_test = (args.kfold_test == 'true')
	if kfold_test:
		path_kw += '/kfold_test/'
	#
	idx_slash = path_kw.strip('/').index('/') + 2
	descriptor = args.descriptor
	path_kw = path_kw[:idx_slash] + descriptor + '/' + path_kw[idx_slash:]
	optim_method = args.optim_method
	path_kw = path_kw[:idx_slash] + optim_method + '/' + path_kw[idx_slash:]
	ed_method = args.ed_method
	path_kw = path_kw[:idx_slash] + ed_method + '/' + path_kw[idx_slash:]
	ged_kwargs = {'ed_method': ed_method, 'optim_method': optim_method}

	### Load dataset.
	for ds_name in DS_Name_List:

		if ds_name == 'poly200r':
			ds_version = args.ds_version
			if ds_version != 'latest':
				path_kw = '/' + ds_name + ds_version + path_kw
			ds_kwargs = {'version': ds_version}
		elif ds_name in ['brem_togn', 'bremond', 'tognetti']:
			tgt_name = args.tgt_name
			path_kw = '/' + ds_name + '/' + tgt_name + path_kw
			ds_kwargs = {
				'level': 'pbe0', 'target': tgt_name, 'sort_by_targets': True,
				'fn_families': None,
				}
		else:
			path_kw = '/' + ds_name + path_kw
			ds_kwargs = {}

		if descriptor == 'atom_bond_types':
			### 1. Get descriptors: atom and bond types.
			X, y, families = get_data(ds_name, descriptor='smiles', format_='networkx', **ds_kwargs)

		elif descriptor == '1hot':
			### 2. Get descriptors: one-hot descriptors on nodes and edges.
			from dataset.feat import DCMolGraphFeaturizer
			featurizer = DCMolGraphFeaturizer(
				use_edges=True,
				use_chirality=True,
				use_partial_charge=False,
				use_distance_stats=False,
				use_xyz=False,
				return_int=True)
			X, y, families = get_data(
				ds_name, descriptor=featurizer, format_='networkx', **ds_kwargs)

		cross_validate(X, y, families, use_best_params=kfold_test, **ged_kwargs)