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
from gklearn.utils.iters import get_iters

from redox_prediction.utils.logging import AverageMeter


def predict(
		inputs: Union[object, np.ndarray],
		y_test: np.ndarray,
		model_task: object,
		perf_eval: object,
		G_test: List[networkx.Graph] = None,
		metric_mode: str = 'pred',
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
		run_time = 0  # TODO: fix this.
		history['pred_time_metric'].update(run_time / len(y_test), len(y_test))
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
		matrix_app,
		y_app,
		params_task,
		kf,
		model_type,
		verbose=False,
		**kwargs
):
	all_history = _init_all_history()
	perf_valid_list = []

	# for each inner CV fold:
	for idx, (train_index, valid_index) in enumerate(
			kf.split(matrix_app, y_app)
	):

		# # For debugging only:  # TODO: comment this.
		# if idx > 1:
		# 	break

		if verbose:
			print('\nTrial: {}/{}'.format(idx + 1, kf.get_n_splits()))

		# split the dataset into train and validation sets:
		matrix_train = matrix_app[train_index, :][:, train_index]
		matrix_valid = matrix_app[valid_index, :][:, train_index]
		y_train = y_app[train_index]
		y_valid = y_app[valid_index]

		# Fit the model:
		model_task, history_task, perf_eval = fit_model(
			matrix_train, y_train,
			params_task,
			metric=kwargs['loss'],
			model_type=model_type
		)

		# Evaluate the model on the validation set:
		perf_valid, y_pred_valid, y_true_valid, pred_history_valid = predict(
			matrix_valid,
			y_valid,
			model_task,
			perf_eval,
			metric_mode='matrix',
			**kwargs
		)

		perf_valid_list.append(perf_valid)

		_update_history_1fold(all_history, history_task, pred_history_valid)

	# Average the performance over the inner CV folds:
	perf_valid = np.mean(perf_valid_list)

	# Since this function is quite fast, I will not save the history to file.

	return perf_valid, all_history, perf_eval


def model_selection_for_ged(
		G_app, y_app, G_test, y_test,
		param_grid,
		param_grid_task,
		model_type,
		parallel=None,
		n_jobs=multiprocessing.cpu_count(),
		read_resu_from_file: int = 1,
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
	param_list = list(ParameterGrid(param_grid))
	param_list_task = list(ParameterGrid(param_grid_task))

	perf_valid_best = (np.inf if model_type == 'reg' else -np.inf)
	for idx, params in get_iters(
			enumerate(param_list),  # @TODO: remove the [0:2]
			desc='model selection for the GED model',
			file=sys.stdout,
			length=len(param_list),
			verbose=True
	):
		print()
		print(
			'---- Parameter settings {}/{} -----:'.format(
				idx + 1, len(param_list)
			)
		)
		print(params)

		# Get the metric matrix:
		from redox_prediction.models.evaluation.ged import fit_model_ged
		model, history, matrix_app = fit_model_ged(
			G_app,
			ged_options=params,
			parallel=parallel,
			n_jobs=n_jobs,
			read_resu_from_file=read_resu_from_file,
			params_idx=str(idx),
			**kwargs
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

			perf_valid, all_history, perf_eval = evaluate_parameters(
				matrix_app, y_app,
				params_task,
				kf,
				model_type,
				fit_history=history,
				params_idx=str(idx),
				verbose=verbose,
				**kwargs
			)

			# Update the best parameters:
			if check_if_valid_better(perf_valid, perf_valid_best, model_type):
				perf_valid_best = perf_valid
				params_best = copy.deepcopy(params)
				params_best_task = copy.deepcopy(params_task)
				best_model = model
				best_history = {
					**copy.deepcopy(all_history), 'fit_time_metric': history['run_time']
				}

	# Refit the best model on the whole dataset:
	print('\n---- Start refitting the best model on the whole valid dataset...')

	# Fit the task model:
	model_task, history_task, perf_eval = fit_model(
		best_model.dis_matrix, y_app,
		params_best_task,
		metric=kwargs['loss'],
		model_type=model_type
	)

	# Predict the app set:
	perf_app, y_pred_app, y_true_app, pred_history_app = predict(
		best_model,
		y_app,
		model_task,
		perf_eval,
		metric_mode='model',
		**kwargs
	)
	history_app = _init_all_history()
	_update_history_1fold(history_app, None, pred_history_app)
	history_app['fit_time_task'] = history_task['fit_time_task']

	# Predict the test set:
	perf_test, y_pred_test, y_true_test, pred_history_test = predict(
		best_model,
		y_test,
		model_task,
		perf_eval,
		G_test=G_test,
		metric_mode='pred',
		**{**kwargs, 'repeats': params_best['repeats']}
	)
	history_test = _init_all_history()
	_update_history_1fold(history_test, None, pred_history_test)

	# Print out the best performance:
	if verbose:
		print('\nPerformance on the refitted model:')
		print('Best app performance: {:.3f}'.format(perf_app))
		print('Best test performance: {:.3f}'.format(perf_test))
		_print_time_info(best_history, history_app, history_test)
		print('Best params: ', params_best)

	# Return the best model:
	return (best_model, model_task), \
		perf_app, perf_test, y_pred_app, y_pred_test, \
		best_history, history_app, history_test, \
		params_best


def evaluate_ged(
		G_app, y_app, G_test, y_test,
		model_type='reg',
		descriptor='atom_bond_types',
		**kwargs
):
	from redox_prediction.models.model_selection.utils import get_params_grid_task
	param_grid_task = get_params_grid_task(
		kwargs['loss'], model_type
	)

	if kwargs.get('embedding') == 'ged:bp_random':
		param_grid = {
			'edit_costs': np.random.rand(6),
			'edit_cost_fun': ['CONSTANT'],
			'method': ['BIPARTITE'],
			'optim_method': ['init'],
			'repeats': [1],
			'return_nb_eo': [False],
		}

	elif kwargs.get('embedding') == 'ged:bp_fitted':
		from redox_prediction.models.embed.ged import get_fitted_edit_costs
		param_grid = {
			'edit_costs': get_fitted_edit_costs(kwargs['dataset'], 'BIPARTITE'),
			'edit_cost_fun': ['CONSTANT'],
			'method': ['BIPARTITE'],
			'optim_method': ['init'],
			'repeats': [1],
			'return_nb_eo': [False],
		}

	else:
		raise ValueError(
			'Unknown embedding method: {}.'.format(kwargs.get('embedding'))
		)

	return model_selection_for_ged(
		G_app, y_app, G_test, y_test,
		param_grid,
		param_grid_task,
		model_type,
		**kwargs
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
		history_app,
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
			history_app['fit_time_task'].sum,
			history_app['fit_time_task'].avg
		)
	)
	print('Prediction time:')
	print(
		'  App (metric):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_app['pred_time_metric'].sum,
			history_app['pred_time_metric'].avg
		)
	)
	print(
		'  App (task):\ttotal {:.3f}\tper data {:.9f}'.format(
			history_app['pred_time_task'].sum,
			history_app['pred_time_task'].avg
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
