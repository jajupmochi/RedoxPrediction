#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:07:35 2022

@author: ljia
"""
import os
import sys
sys.path.insert(0, '../../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
from dataset.load_dataset import get_data
from redox_prediction.models import split_data

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

DIR_CUR_FILE = os.path.dirname(os.path.abspath(__file__))

path_kw = '/mean/'


#%% Simple test run

### Featurizer.

def get_featurizer():
	from dataset.feat import MolGNNFeaturizer
	af_allowable_sets = {
		'atom_type': ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
	# 	'atom_type': ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"],
		'formal_charge': None, # [-2, -1, 0, 1, 2],
		'hybridization': ['SP', 'SP2', 'SP3'],
	# 	'hybridization': ['S', 'SP', 'SP2', 'SP3'],
		'acceptor_donor': ['Donor', 'Acceptor'],
		'aromatic': [True, False],
		'degree': [0, 1, 2, 3, 4, 5],
		# 'n_valence': [0, 1, 2, 3, 4, 5, 6],
		'total_num_Hs': [0, 1, 2, 3, 4],
		'chirality': ['R', 'S'],
		}
	# bf_allowable_sets = {
	# 	'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
	# 	'same_ring': [True, False],
	# 	'conjugated': [True, False],
	# 	'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'],
	# 	}
	featurizer = MolGNNFeaturizer(
		use_edges=False,
		use_partial_charge=False,
		af_allowable_sets=af_allowable_sets,
#		bf_allowable_sets=bf_allowable_sets
		bf_allowable_sets=None,
		add_Hs=False,
		add_3d_coords=False,
		return_format='numpy',
		)
	return featurizer


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
	y_mean = np.mean(y_train)
	y_pred_train = y_train
	y_pred_valid = np.ones(len(y_valid)) * y_mean
	y_pred_test = np.ones(len(y_test)) * y_mean
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
# 	from sklearn.metrics import mean_absolute_error
# 	train_accuracy = mean_absolute_error(y_train, y_pred_train)
# 	valid_accuracy = mean_absolute_error(y_valid, y_pred_valid)
# 	test_accuracy = mean_absolute_error(y_test, y_pred_test)


	### Save predictions.
	fn = DIR_CUR_FILE + '/../outputs/' + path_kw + '/y_values.t' + str(trial_index) + '.pkl'
	pickle.dump({
		'y_train': y_train, 'y_pred_train': y_pred_train,
		'y_valid': y_valid, 'y_pred_valid': y_pred_valid,
		'y_test': y_test, 'y_pred_test': y_pred_test,},
		open(fn, 'wb'))
	# fn = '../outputs/' + path_kw + 'mae_all.t' + str(trial_index) + '.pkl'
	# pickle.dump(MAE_all, open(fn, 'wb'))

	return train_accuracy, valid_accuracy, test_accuracy, None, None



def cross_validate(
		X, targets, families=None,
		n_splits=30, # @todo
		stratified=True,
		output_file=None,
		load_exist_results=False, # @todo
		**kwargs):
	"""Run experiment.
	"""
	# output_file can not be set in the argument directly, as it contains
	# `path_kw`, which is modified in the `__main__`.
	if output_file is None:
		output_file = DIR_CUR_FILE + '/../outputs/' + path_kw + '/results.pkl'

	### Load existing results if possible.
	if load_exist_results and output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = []
#	results = []

	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, \
		train_test_split
	from gklearn.model_selection import RepeatedKFoldWithValid

# 	if mode == 'classif':
# 		stratified = True # @todo: to change it as needed.
# 	else:
# 		stratified = kwargs.get('stratified', True)

	cv = '811' # kwargs.get('cv')
	test_size = (0.1 if cv == '811' else 0.2)

# 	import collections
# 	if np.ceil(test_size * len(X)) < len(collections.Counter(families)):
# 		stratified = False # ValueError: The test_size should be greater or equal to the number of classes.
# 		# @todo: at least let each data in test set has different family?

	if stratified:
		# @todo: to guarantee that at least one sample is in train set (, valid set and test set).
		if '/kfold' in path_kw:
			rs = RepeatedKFoldWithValid(
				n_splits=10, n_repeats=10, stratify=True, random_state=0) # @todo: to change it back.
		else:
			rs = StratifiedShuffleSplit(
				n_splits=n_splits, test_size=test_size, random_state=0) # @todo: to change it back.
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
				stratify=stratify
				)
			X_train, X_valid, y_train, y_valid, train_index, valid_index = op_splits


		### Save indices.
		fn = DIR_CUR_FILE + '/../outputs/' + path_kw + '/indices_splits.t' + str(i-1) + '.pkl'
		os.makedirs(os.path.dirname(fn), exist_ok=True)
		pickle.dump(
			{'train_index': train_index, 'valid_index': valid_index,
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
		fn_model = DIR_CUR_FILE + '/../outputs/' + path_kw + '/model.t' + str(i - 1)
		pickle.dump(model, open(fn_model, 'wb'))
		setup_results['model'] = fn_model

		print()
		print('Performance of current trial:')
		print("train: {0}.".format(perf_train))
		print("valid: {0}.".format(perf_valid))
		print("test: {0}.".format(perf_test))
		cur_results = setup_results
		results.append(cur_results)


# 		### Show model stucture.
# 		input_ = tf.keras.Input(shape=(100,), dtype='int32', name='input')
# 		model_for_plot = tf.keras.Model(inputs=[input_],
# 								   outputs=[model.predict(input_)])
# 		keras.utils.plot_model(model_for_plot, '' + path_kw + '_fcn_model.png',
# 						  show_shapes=True)


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
	parser.add_argument('--cv_mode', type=str, choices=['kfold_test', 'kfold', 'shuffle'], default='shuffle', help='CV mode, where "kfold_test" means using kfold with a given best set of parameters. The default is "shuffle".')
	parser.add_argument('--stratification', type=str, choices=['none', 'db', 'family'], default='none', help='stratification mode, where "db" refers to the deprecated way to stratify according to the names of datasets. "db" is not removed for the sake of compatibility.')

	parser.add_argument('--ds_version', type=str, default='latest', help='the version of the dataset. for poly200r only.')

	parser.add_argument('--tgt_name', type=str, choices=['dGred', 'dGox'], default='dGred', help='the name of the targets. for Redox datasets only.')

	parser.add_argument('--statistics', type=str, choices=['A/s', 'A/c'], default='A/c', help='the statistics used to pool the node features. for vector-based models only.')

	args = parser.parse_args()

	return args


if __name__ == '__main__':

#	test_losses()

	args = parse_args()
	DS_Name_List = (['brem_togn'] if args.ds_name is None else [args.ds_name]) # ['poly200+sugarmono']:
	cv_mode = args.cv_mode
	path_kw += '/' + cv_mode + '/'
	use_best_params = (args.cv_mode == 'kfold_test')
	stratification = args.stratification
	if stratification != 'none':
		path_kw += '/' + stratification + '/'
	stratified = (stratification != 'none')
	statistics = args.statistics
	path_kw = '/' + statistics + '/' + path_kw

	### Load dataset.
	for ds_name in DS_Name_List:

		if ds_name == 'poly200r':
			ds_version = args.ds_version
			if ds_version != 'latest':
				path_kw = '/' + ds_name + ds_version + path_kw
			else:
				path_kw = '/' + ds_name + path_kw
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

		featurizer = get_featurizer()

		X, y, families = get_data(
			ds_name, descriptor=featurizer, format_='vector', 
			statistics=statistics, **ds_kwargs)

		cross_validate(X, y, families)