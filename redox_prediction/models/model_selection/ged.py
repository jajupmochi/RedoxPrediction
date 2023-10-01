"""
model_selection



@Author: linlin
@Date: 20.05.23
"""
import copy
from typing import List, Union

import networkx
import numpy as np

# matplotlib.use('Agg')
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

import multiprocessing
import sys
import time

import copy
from typing import List, Union

import numpy as np

import networkx

from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

from gklearn.utils.iters import get_iters

from redox_prediction.models.model_selection.utils import get_submatrix_by_index
from redox_prediction.utils.logging import AverageMeter


def predict(
		inputs: Union[object, np.ndarray],
		y_test: np.ndarray,
		model_task: object,
		perf_eval: object,
		G_test: List[networkx.Graph] = None,
		metric_mode: str = 'pred',
		y_scaler: object = None,
		model_type: str = 'reg',
		**kwargs
):
	# Initialize average meters:
	history = {k: AverageMeter(keep_all=False) for k in [
		'pred_time_metric', 'pred_time_task',
	]}

	# Predict the metric matrix:
	if metric_mode == 'model':
		# Get the metric matrix from the model, used for the prediction on the
		# app set:
		matrix_test = inputs.dis_matrix
		run_time = inputs.run_time
		n = len(matrix_test)
		history['pred_time_metric'].update(
			# run_time / (n * (n - 1) / 2) * n, n
			run_time / (n - 1) * 2, n
		)
	elif metric_mode == 'pred':
		# Compute the metric matrix from the test set:
		matrix_test = inputs.transform(G_test, repeat=kwargs['repeats'])
		run_time = inputs.test_run_time
		history['pred_time_metric'].update(run_time / len(y_test), len(y_test))
	elif metric_mode == 'matrix':
		# Use the given metric matrix:
		matrix_test = inputs
		run_time = kwargs['pairwise_run_time'] * matrix_test.shape[1]
		history['pred_time_metric'].update(run_time, len(y_test))
	else:
		raise ValueError(
			'"metric_mode" must be either "model", "pred" or "matrix".'
		)

	start_time = time.time()
	y_pred_test = model_task.predict(matrix_test)
	history['pred_time_task'].update(
		(time.time() - start_time) / len(y_test), len(y_test)
	)

	# Compute the performance:
	if y_scaler is not None:
		if model_type == 'classif':
			# Convert to int before inverse transform:
			y_pred_test = np.ravel(y_pred_test.astype(int))
			y_test = np.ravel(y_test.astype(int))
		else:
			y_pred_test = y_pred_test.reshape(-1, 1)
			y_test = y_test.reshape(-1, 1)
		y_pred_test = y_scaler.inverse_transform(y_pred_test)
		y_test = y_scaler.inverse_transform(y_test)
	perf_test = perf_eval(y_pred_test, y_test)

	return perf_test, y_pred_test, y_test, history


def fit_model(
		matrix_train, y_train,
		params: dict,
		metric: str = 'distance',
		model_type: str = 'reg',
):
	"""
	Fit a model on the given dataset.

	Notes
	-----
	Since the metric model has been trained in advance, we only need to train
	the task-specific model here.
	"""
	# Initialize average meters:
	history_task = {k: AverageMeter(keep_all=False) for k in [
		'fit_time_task'
	]}

	# Initialize the model:
	from redox_prediction.models.evaluation.utils import get_estimator
	model_task, perf_eval = get_estimator(metric, model_type)

	# Set the parameters:
	model_task.set_params(**params)

	# Fit the model:
	start_time = time.time()
	model_task.fit(matrix_train, y_train)
	history_task['fit_time_task'].update(
		(time.time() - start_time) / len(y_train), len(y_train)
	)

	return model_task, history_task, perf_eval


def evaluate_parameters(
		matrix_train, y_train,
		matrix_valid, y_valid,
		params_task,
		model_type,
		verbose=False,
		**kwargs
):
	all_history = _init_all_history()
	perf_valid_list = []

	# Fit the model:
	model_task, history_task, perf_eval = fit_model(
		matrix_train, y_train,
		params_task,
		metric='distance',  # for GEDs it is different.
		model_type=model_type
	)

	# Evaluate the model on the validation set:
	perf_valid, y_pred_valid, y_true_valid, pred_history_valid = predict(
		matrix_valid,
		y_valid,
		model_task,
		perf_eval,
		metric_mode='matrix',
		model_type=model_type,
		**kwargs
	)

	perf_valid_list.append(perf_valid)

	_update_history_1fold(all_history, history_task, pred_history_valid)

	# Average the performance over the inner CV folds:
	perf_valid = np.mean(perf_valid_list)

	# Since this function is quite fast, I will not save the history to file.

	return perf_valid, all_history, perf_eval, model_task


def model_selection_for_ged(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		param_grid,
		param_grid_task,
		model_type,
		parallel=False,
		n_jobs=multiprocessing.cpu_count(),
		read_resu_from_file: int = 1,
		verbose=True,
		**kwargs
):
	# Scale the targets:
	# @TODO: use minmax or log instead?
	if model_type == 'reg':
		from redox_prediction.models.model_selection.utils import get_y_scaler
		y_scaler = get_y_scaler(y_train, kwargs.get('y_scaling'))
		if y_scaler is not None:
			y_train = y_scaler.transform(np.reshape(y_train, (-1, 1)))
			y_valid = y_scaler.transform(np.reshape(y_valid, (-1, 1)))
			y_test = y_scaler.transform(np.reshape(y_test, (-1, 1)))
	elif model_type == 'classif':
		# ensure that the labels are in the range [0, n_classes - 1].
		# This is important for classification with Sigmoid and BCELoss.
		# If the labels are not in this range, the loss might be negative.
		from sklearn.preprocessing import LabelEncoder
		y_scaler = LabelEncoder().fit(y_train)
		y_train = y_scaler.transform(y_train)
		y_valid = y_scaler.transform(y_valid)
		y_test = y_scaler.transform(y_test)
		# Ensure the values are floats:
		y_train = y_train.astype(float)
		y_valid = y_valid.astype(float)
		y_test = y_test.astype(float)  # @TODO: is this necessary?
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')
	kwargs['y_scaler'] = y_scaler
	y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(
		y_test
	)

	# # Set cross-validation method:
	# if model_type == 'reg':
	# 	kf = KFold(n_splits=5, shuffle=True, random_state=42)
	# elif model_type == 'classif':
	# 	kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	# else:
	# 	raise ValueError('"model_type" must be either "reg" or "classif".')

	# Do cross-validation:
	param_list = list(ParameterGrid(param_grid))
	param_list_task = list(ParameterGrid(param_grid_task))

	perf_valid_best = (np.inf if model_type == 'reg' else -np.inf)
	for idx, params in enumerate(param_list):  # debug: remove the [0:2]
		# 		get_iters(
		# 		enumerate(param_list),  # debug: remove the [0:2]
		# 		desc='model selection for the GED model',
		# 		file=sys.stdout,
		# 		length=len(param_list),
		# 		verbose=True
		# )):
		if verbose:
			print()
			print(
				'---- Parameter settings {}/{} -----:'.format(
					idx + 1, len(param_list)
				)
			)
			print(params)

		# Get the metric matrix for entire dataset:\
		# We do this a prior for: 1) Reduce time complexity
		# by computing only one matrix over multiple GNNs.
		from redox_prediction.models.evaluation.ged import fit_model_ged
		model, history, matrix_all = fit_model_ged(
			G_train + G_valid + G_test,
			ged_options=params,
			parallel=parallel,
			n_jobs=n_jobs,
			read_resu_from_file=read_resu_from_file,
			params_idx=str(idx),
			reorder_graphs=True,
			**kwargs
		)
		matrix_train = get_submatrix_by_index(
			matrix_all, G_train, idx_key='id'
		)
		matrix_valid = get_submatrix_by_index(
			matrix_all, G_train, G_valid, idx_key='id'
		)

		for idx_task, params_task in enumerate(param_list_task):
			if verbose:
				print()
				print(
					'---- Task parameter settings {}/{} -----:'.format(
						idx_task + 1, len(param_list_task)
					)
				)
				print(params_task)

			perf_valid, all_history, perf_eval, model_task = evaluate_parameters(
				matrix_train, y_train,
				matrix_valid, y_valid,
				params_task,
				model_type,
				fit_history=history,
				params_idx=str(idx),
				verbose=verbose,
				**{**kwargs, 'pairwise_run_time': history['run_time'].avg}
			)

			# Update the best parameters:
			if check_if_valid_better(perf_valid, perf_valid_best, model_type):
				best_model = model
				best_model_task = model_task
				perf_valid_best = perf_valid
				params_best = copy.deepcopy(params)
				params_best_task = copy.deepcopy(params_task)
				best_history = {
					**copy.deepcopy(all_history),
					'fit_time_metric': history['run_time']
				}

	# # Refit the best model on the whole dataset:
	# print('\n---- Start refitting the best model on the whole valid dataset...')

	# # Fit the task model:
	# matrix_app = get_submatrix_by_index(best_model.gram_matrix, G_app, idx_key='id')
	# model_task, history_task, perf_eval = fit_model(
	# 	matrix_app,
	# 	y_app,
	# 	params_best_task,
	# 	metric=kwargs['loss'],
	# 	model_type=model_type
	# )

	# Predict the train set:
	perf_train, y_pred_train, y_true_train, pred_history_train = predict(
		get_submatrix_by_index(
			best_model.dis_matrix, G_train, G_train, idx_key='id'
		),
		y_train,
		best_model_task,
		perf_eval,
		metric_mode='matrix',
		model_type=model_type,
		**{**kwargs, 'pairwise_run_time': best_history['fit_time_metric'].avg}
	)
	history_train = _init_all_history()
	_update_history_1fold(history_train, None, pred_history_train)

	# Predict the valid set:
	perf_valid, y_pred_valid, y_true_valid, pred_history_valid = predict(
		get_submatrix_by_index(
			best_model.dis_matrix, G_train, G_valid, idx_key='id'
		),
		y_valid,
		best_model_task,
		perf_eval,
		metric_mode='matrix',
		model_type=model_type,
		**{**kwargs, 'pairwise_run_time': best_history['fit_time_metric'].avg}
	)
	history_valid = _init_all_history()
	_update_history_1fold(history_valid, None, pred_history_valid)

	# Predict the test set:
	perf_test, y_pred_test, y_true_test, pred_history_test = predict(
		get_submatrix_by_index(
			best_model.dis_matrix, G_train, G_test, idx_key='id'
		),
		y_test,
		best_model_task,
		perf_eval,
		metric_mode='matrix',
		model_type=model_type,
		**{**kwargs, 'pairwise_run_time': best_history['fit_time_metric'].avg}
	)
	history_test = _init_all_history()
	_update_history_1fold(history_test, None, pred_history_test)

	# Print out the best performance:
	if verbose:
		print('\nPerformance on the best model:')
		print('Best train performance: {:.3f}'.format(perf_train))
		print('Best valid performance: {:.3f}'.format(perf_valid))
		print('Best test performance: {:.3f}'.format(perf_test))
		_print_time_info(
			best_history, history_train, history_valid, history_test
		)
		print('Best params: ', params_best)
		print('Best task params: ', params_best_task)

	# Return the best model:
	return (best_model, best_model_task), \
		perf_train, perf_valid, perf_test, \
		y_pred_train, y_pred_valid, y_pred_test, \
		best_history, history_train, history_valid, history_test, \
		(params_best, params_best_task)


def evaluate_ged(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		model_type='reg',
		descriptor='atom_bond_types',
		**kwargs
):
	from redox_prediction.models.model_selection.utils import \
		get_params_grid_task
	param_grid_task = get_params_grid_task(
		'distance',  # not the same as graph kernels.
		model_type
	)

	if kwargs.get('model') == 'ged:bp_random':
		param_grid = {
			'edit_costs': [np.random.rand(6)],  # todo: this is not a real randomness
			'edit_cost_fun': ['CONSTANT'],
			'method': ['BIPARTITE'],
			'optim_method': ['init'],
			'repeats': [1],
			'return_nb_eo': [False],
		}

	elif kwargs.get('model') == 'ged:bp_fitted':
		from redox_prediction.models.embed.ged import get_fitted_edit_costs
		param_grid = {
			'edit_costs': [get_fitted_edit_costs(kwargs['ds_name'], 'BIPARTITE')],
			'edit_cost_fun': ['CONSTANT'],
			'method': ['BIPARTITE'],
			'optim_method': ['init'],
			'repeats': [1],
			'return_nb_eo': [False],
		}

	else:
		raise ValueError(
			'Unknown model: {}.'.format(kwargs.get('model'))
		)

	return model_selection_for_ged(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		param_grid,
		param_grid_task,
		model_type,
		**{**kwargs, 'descriptor': descriptor}
	)

# %%


def check_if_valid_better(perf_valid, perf_valid_best, model_type):
	if model_type == 'reg':
		return perf_valid < perf_valid_best
	elif model_type == 'classif':
		return perf_valid > perf_valid_best
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')


def _update_history_1fold(
		all_history: dict,
		fit_history: dict = None,
		predict_history: dict = None,
		rm_unused_keys: bool = False
):
	if fit_history is not None:
		for key in fit_history:
			all_history[key].update(fit_history[key])
	if predict_history is not None:
		for key in predict_history:
			all_history[key].update(predict_history[key])
		all_history['pred_time_total'].update(
			all_history['pred_time_metric'].avg + all_history[
				'pred_time_task'].avg,
			all_history['pred_time_metric'].count
		)


def _init_all_history():
	all_history = {key: AverageMeter(keep_all=False) for key in [
		'fit_time_task', 'pred_time_metric', 'pred_time_task', 'pred_time_total'
	]}
	return all_history


def _print_time_info(
		best_history,
		history_train,
		history_valid,
		history_test
):
	print('Training time:')
	print(
		'  Metric model:\ttotal {:.3f}\tper pair {:.9f}'.format(
			best_history['fit_time_metric'].sum,
			best_history['fit_time_metric'].avg
		)
	)
	print(
		'  Task model:\ttotal {:.3f}\tper data {:.9f}'.format(
			best_history['fit_time_task'].sum,
			best_history['fit_time_task'].avg
		)
	)
	print('Prediction time:')
	print(
		'  Train (metric):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_train['pred_time_metric'].sum,
			history_train['pred_time_metric'].avg
		)
	)
	print(
		'  Train (task):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_train['pred_time_task'].sum,
			history_train['pred_time_task'].avg
		)
	)
	print(
		'  Valid (metric):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_valid['pred_time_metric'].sum,
			history_valid['pred_time_metric'].avg
		)
	)
	print(
		'  Valid (task):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_valid['pred_time_task'].sum,
			history_valid['pred_time_task'].avg
		)
	)
	print(
		'  Test (metric):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_test['pred_time_metric'].sum,
			history_test['pred_time_metric'].avg
		)
	)
	print(
		'  Test (task):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_test['pred_time_task'].sum,
			history_test['pred_time_task'].avg
		)
	)
