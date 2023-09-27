"""
model_selection



@Author: linlin
@Date: 20.05.23
"""

import numpy as np
import matplotlib

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, ParameterGrid

# from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool, Array
from functools import partial
import sys
import os
import time
import datetime
# from os.path import basename, splitext
from gklearn.utils.graphfiles import loadDataset
from gklearn.utils.iters import get_iters


def evaluate_gram_matrix(
		gram_matrix_app_train, y_app_train,
		gram_matrix_app_valid, y_app_valid,
		param_grid=None,
		model='gram', mode='reg'
):
	# if the precomputed matrix is a Gram matrix:
	if model == 'gram':
		if mode == 'reg':
			from sklearn.kernel_ridge import KernelRidge
			from redox_prediction.utils.distances import rmse
			estimator = KernelRidge(kernel='precomputed')
			# scoring = 'neg_root_mean_squared_error'
			perf_eval = rmse
		elif mode == 'classif':
			from sklearn.svm import SVC
			from redox_prediction.utils.distances import accuracy
			estimator = SVC(kernel='precomputed')
			# scoring = 'accuracy'
			perf_eval = accuracy
		else:
			raise ValueError('"model_type" must be either "reg" or "classif".')
	# if the precomputed matrix is a distance matrix:
	elif model == 'distance':
		if mode == 'reg':
			from sklearn.neighbors import KNeighborsRegressor
			from redox_prediction.utils.distances import rmse
			estimator = KNeighborsRegressor(metric='precomputed')
			# scoring = 'neg_root_mean_squared_error'
			perf_eval = rmse
		elif mode == 'classif':
			from sklearn.neighbors import KNeighborsClassifier
			from redox_prediction.utils.distances import accuracy
			estimator = KNeighborsClassifier(metric='precomputed')
			# scoring = 'accuracy'
			perf_eval = accuracy
		else:
			raise ValueError('"model_type" must be either "reg" or "classif".')
	else:
		raise ValueError('"model" must be either "gram" or "distance".')

	cv_results = {}
	cv_results['perf_eval'] = perf_eval
	cv_results['params'] = list(ParameterGrid(param_grid))

	# Do cross-validation:
	for n_split in range(len(gram_matrix_app_train)):
		cv_results['split' + str(n_split) + '_train_scores'] = []
		cv_results['split' + str(n_split) + '_valid_scores'] = []

		for i_p, params in enumerate(cv_results['params']):
			estimator.set_params(**params)
			estimator.fit(gram_matrix_app_train[n_split], y_app_train[n_split])
			y_pred_train = estimator.predict(gram_matrix_app_train[n_split])
			y_pred_valid = estimator.predict(gram_matrix_app_valid[n_split])
			cv_results['split' + str(n_split) + '_train_scores'].append(
				perf_eval(y_pred_train, y_app_train[n_split])
			)
			cv_results['split' + str(n_split) + '_valid_scores'].append(
				perf_eval(y_pred_valid, y_app_valid[n_split])
			)

	# Compute mean and std:
	cv_results['mean_train_scores'] = np.mean(
		np.array(
			[
				cv_results['split' + str(n_split) + '_train_scores']
				for n_split in range(len(gram_matrix_app_train))
			]
		),
		axis=0
	)
	cv_results['std_train_scores'] = np.std(
		np.array(
			[
				cv_results['split' + str(n_split) + '_train_scores']
				for n_split in range(len(gram_matrix_app_train))
			]
		),
		axis=0
	)
	cv_results['mean_valid_scores'] = np.mean(
		np.array(
			[
				cv_results['split' + str(n_split) + '_valid_scores']
				for n_split in range(len(gram_matrix_app_train))
			]
		),
		axis=0
	)
	cv_results['std_valid_scores'] = np.std(
		np.array(
			[
				cv_results['split' + str(n_split) + '_valid_scores']
				for n_split in range(len(gram_matrix_app_train))
			]
		),
		axis=0
	)

	# Get the index of the best model:
	if mode == 'reg':
		best_idx = np.argmin(cv_results['mean_valid_scores'])
	elif mode == 'classif':
		best_idx = np.argmax(cv_results['mean_valid_scores'])
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')
	cv_results['best_idx'] = best_idx

	# Get the best model:
	cv_results['best_params'] = cv_results['params'][best_idx]
	cv_results['best_estimator'] = estimator.set_params(
		**cv_results['best_params']
	)
	cv_results['best_train_score'] = cv_results['mean_train_scores'][best_idx]
	cv_results['best_valid_score'] = cv_results['mean_valid_scores'][best_idx]
	cv_results['best_std_train_score'] = cv_results['std_train_scores'][best_idx]
	cv_results['best_std_valid_score'] = cv_results['std_valid_scores'][best_idx]

	return cv_results


def model_selection_for_precomputed_kernel(
		G_app, y_app, G_test, y_test,
		estimator,
		param_grid_precomputed,
		param_grid,
		model_type,
		fit_test=False,
		parallel=False,
		n_jobs=multiprocessing.cpu_count(),
		read_gm_from_file=False,
		verbose=True,
		**kwargs
):
	# Set cross-validation method:
	if model_type == 'reg':
		kf = KFold(n_splits=5, shuffle=True, random_state=42)
	elif model_type == 'classif':
		kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')

	# Do cross-validation:
	param_list_precomputed = list(ParameterGrid(param_grid_precomputed))

	perf_valid_best = (np.inf if model_type == 'reg' else -np.inf)
	for idx, params_out in get_iters(
			enumerate(param_list_precomputed),  # @TODO: remove the [0:2]
			desc='model selection for the graph kernel',
			file=sys.stdout,
			length=len(param_list_precomputed),
			verbose=True
	):
		if verbose:
			print()
			print(params_out)

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
			**params_out
		)
		try:
			gram_matrix_app = model.fit_transform(G_app, save_gm_train=False)
		except FloatingPointError as e:
			print(
				'Encountered FloatingPointError while fitting the model with '
				'the parameters below:' + '\n' + str(params_out)
			)
			print(e)
			continue
		run_time = model.run_time

		# Splits gram_matrix_app and y_app into train and valid sets using kf:
		gram_matrix_app_train = []
		gram_matrix_app_valid = []
		y_app_train = []
		y_app_valid = []
		for train_index, valid_index in kf.split(gram_matrix_app, y_app):
			gram_matrix_app_train.append(gram_matrix_app[train_index, :][:, train_index])
			gram_matrix_app_valid.append(gram_matrix_app[valid_index, :][:, train_index])
			y_app_train.append(y_app[train_index])
			y_app_valid.append(y_app[valid_index])


		# Evaluate the current gram matrix:
		cv_results = evaluate_gram_matrix(
			gram_matrix_app_train, y_app_train,
			gram_matrix_app_valid, y_app_valid,
			param_grid=param_grid,
			model='gram', mode=model_type
		)

		# Get the best score for the current gram matrix:
		perf_valid = cv_results['best_valid_score']

		# Update the best score:
		if check_if_valid_better(perf_valid, perf_valid_best, model_type):
			perf_valid_best = perf_valid
			gram_matrix_app_best = gram_matrix_app
			cv_results_best = cv_results
			params_in_best = cv_results['best_params']
			params_out_best = params_out
			run_time_best = run_time

	# Refit the best model on the whole dataset:
	model = estimator(
		node_labels=kwargs['node_labels'],
		edge_labels=kwargs['edge_labels'],
		ds_infos={'directed': True},
		parallel=parallel,
		n_jobs=n_jobs,
		chunksize=None,
		normalize=True,
		copy_graphs=True,  # make sure it is a full deep copy. and faster!
		verbose=verbose,
		**params_out_best
	)
	gram_matrix_app = model.fit_transform(G_app, save_gm_train=False)
	if fit_test:
		gram_matrix_test = model.transform(G_test)
	else:
		gram_matrix_test = None
	run_time = model.run_time

	# Get the best prediction estimator from the best cv_results and refit it:
	estimator_best = cv_results_best['best_estimator']
	estimator_best.fit(gram_matrix_app, y_app)

	# Evaluate the best estimator on the test set:
	y_pred_app = estimator_best.predict(gram_matrix_app)
	perf_app = cv_results_best['perf_eval'](y_pred_app, y_app)
	if fit_test:
		y_pred_test = estimator_best.predict(gram_matrix_test)
		perf_test = cv_results_best['perf_eval'](y_pred_test, y_test)
	else:
		perf_test = None

	# Print out the best performance:
	# if verbose:
	print('Best app performance: ', perf_app)
	print('Best test performance: ', perf_test)
	print('Best params Gram: ', params_out_best)
	print('Best params Pred: ', params_in_best)

	# Return the best model:
	return (model, estimator_best), perf_app, perf_test, \
		gram_matrix_app, gram_matrix_test, params_out_best, params_in_best


def check_if_valid_better(perf_valid, perf_valid_best, mode):
	if mode == 'reg':
		return perf_valid < perf_valid_best
	elif mode == 'classif':
		return perf_valid > perf_valid_best
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')



# def evaluate_gram_matrix(
# 		gram_app, y_app, gram_test=None, y_test=None,
# 		param_grid=None, fit_test=False, model='gram', model_type='reg'
# ):
# 	# if the precomputed matrix is a Gram matrix:
# 	if model == 'gram':
# 		if model_type == 'reg':
# 			from sklearn.kernel_ridge import KernelRidge
# 			from redox_prediction.utils.distances import rmse
# 			estimator = KernelRidge(kernel='precomputed')
# 			scoring = 'neg_root_mean_squared_error'
# 			perf_eval = rmse
# 		elif model_type == 'classif':
# 			from sklearn.svm import SVC
# 			from redox_prediction.utils.distances import accuracy
# 			estimator = SVC(kernel='precomputed')
# 			scoring = 'accuracy'
# 			perf_eval = accuracy
# 		else:
# 			raise ValueError('"model_type" must be either "reg" or "classif".')
# 	# if the precomputed matrix is a distance matrix:
# 	elif model == 'distance':
# 		if model_type == 'reg':
# 			from sklearn.neighbors import KNeighborsRegressor
# 			from redox_prediction.utils.distances import rmse
# 			estimator = KNeighborsRegressor(metric='precomputed')
# 			scoring = 'neg_root_mean_squared_error'
# 			perf_eval = rmse
# 		elif model_type == 'classif':
# 			from sklearn.neighbors import KNeighborsClassifier
# 			from redox_prediction.utils.distances import accuracy
# 			estimator = KNeighborsClassifier(metric='precomputed')
# 			scoring = 'accuracy'
# 			perf_eval = accuracy
# 		else:
# 			raise ValueError('"model_type" must be either "reg" or "classif".')
# 	else:
# 		raise ValueError('"model" must be either "gram" or "distance".')
#
# 	from sklearn.model_selection import GridSearchCV
# 	model_pred = GridSearchCV(
# 		estimator, param_grid=param_grid,
# 		scoring=scoring,
# 		cv=5, return_train_score=True, refit=True
# 	)
# 	model_pred.fit(gram_app, y_app)
# 	y_pred_app = model_pred.predict(gram_app)
# 	y_pred_test = (model_pred.predict(gram_test) if fit_test else None)
#
# 	perf_app = perf_eval(y_pred_app, y_app)
# 	perf_test = (perf_eval(y_pred_test, y_test) if fit_test else None)
#
# 	return perf_app, perf_test, model_pred.best_estimator_
#
#
# def model_selection_for_precomputed_kernel(
# 		G_app, y_app, G_test, y_test,
# 		estimator,
# 		param_grid_precomputed,
# 		param_grid,
# 		model_type,
# 		fit_test=False,
# 		NUM_TRIALS=1,
# 		parallel=False,
# 		n_jobs=multiprocessing.cpu_count(),
# 		read_gm_from_file=False,
# 		verbose=True,
# 		**kwargs
# ):
# 	# --- Compute all gram matrices ---
# 	param_list_precomputed = list(ParameterGrid(param_grid_precomputed))
#
# 	# gram_matrices = [
# 	# ]  # a list to store gram matrices for all param_grid_precomputed
# 	# gram_matrix_time = [
# 	# ]  # a list to store time to calculate gram matrices
# 	# param_list_pre_revised = [
# 	# ]  # list to store param grids precomputed ignoring the useless ones
#
# 	perf_app_best = np.inf
# 	for idx, params_out in get_iters(
# 			enumerate(param_list_precomputed),  # @TODO: remove the [0:3]
# 			desc='model selection for the graph kernel',
# 			file=sys.stdout,
# 			length=len(param_list_precomputed),
# 			verbose=True
# 	):
# 		if verbose:
# 			print()
# 			print(params_out)
#
# 		model = estimator(
# 			node_labels=kwargs['node_labels'],
# 			edge_labels=kwargs['edge_labels'],
# 			ds_infos={'directed': True},
# 			parallel=parallel,
# 			n_jobs=n_jobs,
# 			chunksize=None,
# 			normalize=True,
# 			copy_graphs=True,  # make sure it is a full deep copy. and faster!
# 			verbose=verbose,
# 			**params_out
# 		)
# 		gram_matrix_app = model.fit_transform(G_app, save_gm_train=False)
# 		run_time = model.run_time
# 		# gram_matrices.append(gram_matrix)
# 		# gram_matrix_time.append(run_time)
#
# 		if fit_test:
# 			gram_matrix_test = model.transform(G_test)  #, save_gm_test=True)
# 		else:
# 			gram_matrix_test = None
#
# 		# Cross validation:
# 		# @TODO: here might be a bit overfitting, since we are using the train
# 		# + valid set to tune the params_out. But it should not be a big issue
# 		# since the `embed` and the other optim_method use this same strategy,
# 		# so it is a fair comparison.
# 		perf_app, perf_test, model_pred = evaluate_gram_matrix(
# 			gram_matrix_app, y_app[:],
# 			gram_test=gram_matrix_test, y_test=y_test[:],
# 			param_grid=param_grid,
# 			fit_test=fit_test, model='gram', model_type=model_type
# 		)
#
# 		# Check if the current valid performance is better:
# 		if perf_app < perf_app_best:
# 			perf_app_best = perf_app
# 			perf_test_best = perf_test
# 			model_best = model
# 			model_pred_best = model_pred
# 			gram_matrix_app_best = gram_matrix_app
# 			gram_matrix_test_best = gram_matrix_test
# 			params_out_best = params_out
#
# 	params_in_best = {
# 		k: v for k, v in model_pred_best.get_params().items() if k in param_grid
# 	}
# 	# Print out the best performance:
# 	if verbose:
# 		print('Best performance: ', perf_app_best)
# 		print('Best params Gram: ', params_out_best)
# 		print('Best params Pred: ', params_in_best)
# 		print('Best test performance: ', perf_test_best)
#
# 	# Return the best model:
# 	return (model, model_pred_best), perf_app_best, perf_test_best, \
# 		gram_matrix_app_best, gram_matrix_test_best, params_out_best, params_in_best
