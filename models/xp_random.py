#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:51:50 2022

@author: ljia
"""
import os
import sys
sys.path.insert(0, '../')
import pickle
import numpy as np
from dataset.load_dataset import get_data
from models.utils import split_data

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


path_kw = '/random/'


#%% Simple test run


def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test,
				   scale_x=False,
				   scale_y=(True if 'std_y' in path_kw else False), #@todo
				   **kwargs):

	trial_index = kwargs.get('trial_index')

	# Normalize targets.
	if scale_y:
		from sklearn.preprocessing import StandardScaler
		y_scaler = StandardScaler().fit(np.reshape(y_train, (-1, 1)))
		y_train = y_scaler.transform(np.reshape(y_train, (-1, 1)))
		y_valid = y_scaler.transform(np.reshape(y_valid, (-1, 1)))
		y_test = y_scaler.transform(np.reshape(y_test, (-1, 1)))
	y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)

	# Train and predict.
	min_y, max_y = np.min(y_train), np.max(y_train)
	y_pred_train = y_train
	y_pred_valid = np.random.uniform(min_y, max_y, len(y_valid))
	y_pred_test = np.random.uniform(min_y, max_y, len(y_test))
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
	fn = '../outputs/' + path_kw + '/y_values.t' + str(trial_index) + '.pkl'
	pickle.dump({'y_train': y_train, 'y_pred_train': y_pred_train,
			  'y_valid': y_valid, 'y_pred_valid': y_pred_valid,
			  'y_test': y_test, 'y_pred_test': y_pred_test,},
			 open(fn, 'wb'))

	return train_accuracy, valid_accuracy, test_accuracy, None, None


def cross_validate(X, targets, families=None,
				   n_splits=30,
				   stratified=True, # @todo
				   output_file='../outputs/' + path_kw + '/results.pkl',
				   **kwargs):
	"""Run expriment.
	"""
# 	### Load existing results if possible.
# 	if output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
# 		with open(output_file, 'rb') as file:
# 			results = pickle.load(file)['results']
# 	else:
# 		results = []
	results = []

	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split

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
		rs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
							   random_state=0) # @todo: to change it back.
		split_scheme = rs.split(X, families)
	else:
# 		rs = ShuffleSplit(n_splits=n_splits, test_size=.1) #, random_state=0)
		rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0) # @todo: to change it back.
		split_scheme = rs.split(X)


	i = 1
	for app_index, test_index in split_scheme:
		print()
		print("Split {0}/{1}".format(i, n_splits))
		i = i + 1

		# Check if the split already computed.
		if i - 1 <= len(results):
			print('Skip.')
			continue

		cur_results = {}
		# Get splitted data
		if stratified:
			X_app, X_test, y_app, y_test, F_app, F_tests = split_data(
				(X, targets, families), app_index, test_index)
		else:
			X_app, X_test, y_app, y_test = split_data((X, targets),
											   app_index, test_index)

		# Split evaluation set.
		valid_size = test_size / (1 - test_size)
		stratify = (F_app if stratified else None)
		op_splits = train_test_split(X_app, y_app, app_index,
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


		cur_results['y_app'] = y_app
		cur_results['y_test'] = y_test
		cur_results['app_index'] = app_index
		cur_results['train_index'] = train_index
		cur_results['valid_index'] = valid_index
		cur_results['test_index'] = test_index

		kwargs['nb_trial'] = i - 1
# 		for setup in distances.keys():
# 		print("{0} Mode".format(setup))
		setup_results = {}


		# directory to save all results.
		output_dir = '.'.join(output_file.split('.')[:-1])
		os.makedirs(output_dir, exist_ok=True)
		perf_train, perf_valid, perf_test, bparams, model = evaluate_model(
			X_train, y_train, X_valid, y_valid, X_test, y_test,
			output_dir=output_dir, trial_index=i-1, **kwargs)

# 		setup_results['perf_app'] = perf_app
		setup_results['perf_train'] = perf_train
		setup_results['perf_valid'] = perf_valid
		setup_results['perf_test'] = perf_test
		setup_results['best_params'] = bparams
		# Save model.
		fn_model = '../outputs/' + path_kw + '/model.t' + str(i - 1)
		with open(fn_model, 'wb') as f:
			pickle.dump(model, f)
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

# 		# Check if the (possible) existing file was updated by another thread during the computation of this split.
# 		if output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
# 			with open(output_file, 'rb') as file:
# 				results_new = pickle.load(file)['results']
# 				if i - 1 <= len(results_new):
# 					print('Skip.')
# 					continue

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


if __name__ == '__main__':

# 	test_losses()

	### Load dataset.
	for ds_name in ['poly200']: # ['poly200+sugarmono']: ['poly200']

		X, y, families = get_data(ds_name, descriptor='smiles', format_='networkx')
		cross_validate(X, y, families)