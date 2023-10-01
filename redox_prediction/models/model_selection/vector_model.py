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
		# inputs: Union[object, np.ndarray],
		x_test: np.ndarray,
		y_test: np.ndarray,
		model: object,
		perf_eval: object,
		# G_test: List[networkx.Graph] = None,
		# metric_mode: str = 'pred',
		y_scaler: object = None,
		model_type: str = 'reg',
		**kwargs
):
	# Initialize average meters:
	history = {k: AverageMeter(keep_all=False) for k in [
		'pred_time',
	]}

	# # Predict the metric matrix:
	# if metric_mode == 'model':
	# 	# Get the metric matrix from the model, used for the prediction on the
	# 	# app set:
	# 	matrix_test = inputs.dis_matrix
	# 	run_time = inputs.run_time
	# 	n = len(matrix_test)
	# 	history['pred_time_metric'].update(
	# 		# run_time / (n * (n - 1) / 2) * n, n
	# 		run_time / (n - 1) * 2, n
	# 	)
	# elif metric_mode == 'pred':
	# 	# Compute the metric matrix from the test set:
	# 	matrix_test = inputs.transform(G_test, repeat=kwargs['repeats'])
	# 	run_time = inputs.test_run_time
	# 	history['pred_time_metric'].update(run_time / len(y_test), len(y_test))
	# elif metric_mode == 'matrix':
	# 	# Use the given metric matrix:
	# 	matrix_test = inputs
	# 	run_time = kwargs['pairwise_run_time'] * matrix_test.shape[1]
	# 	history['pred_time_metric'].update(run_time, len(y_test))
	# else:
	# 	raise ValueError(
	# 		'"metric_mode" must be either "model", "pred" or "matrix".'
	# 	)

	start_time = time.time()
	y_pred_test = model.predict(x_test)
	history['pred_time'].update(
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
		X_train, y_train,
		estimator: callable,
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
	history = {k: AverageMeter(keep_all=False) for k in [
		'fit_time'
	]}

	model = estimator(**params)

	# Get the performance evaluator:
	if model_type == 'reg':
		# todo: This should be changed as needed.
		from sklearn.metrics import mean_absolute_error
		perf_eval = mean_absolute_error
	else:
		from redox_prediction.utils.distances import accuracy
		perf_eval = accuracy

	# Fit the model:
	start_time = time.time()
	model.fit(X_train, y_train)
	history['fit_time'].update(
		(time.time() - start_time) / len(y_train), len(y_train)
	)

	return model, history, perf_eval


def evaluate_parameters(
		G_train, y_train,
		G_valid, y_valid,
		params: dict,
		estimator: callable,
		model_type,
		verbose=False,
		**kwargs
):
	all_history = _init_all_history()
	perf_valid_list = []

	# Fit the model:
	model, history, perf_eval = fit_model(
		G_train, y_train,
		estimator,
		params,
		metric='distance',  # for GEDs it is different.
		model_type=model_type
	)

	if 'model' in kwargs:
		# Replace "model" in kwargs with "model_name":
		kwargs = kwargs.copy()
		kwargs['model_name'] = kwargs['model']
		del kwargs['model']

	# Evaluate the model on the validation set:
	perf_valid, y_pred_valid, y_true_valid, pred_history_valid = predict(
		G_valid,
		y_valid,
		model,
		perf_eval,
		model_type=model_type,
		**kwargs
	)

	perf_valid_list.append(perf_valid)

	_update_history_1fold(all_history, history, pred_history_valid)

	# Average the performance over the inner CV folds:
	perf_valid = np.mean(perf_valid_list)

	# Since this function is quite fast, I will not save the history to file.

	return perf_valid, all_history, perf_eval, model


def model_selection_for_vector_model(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		estimator,
		param_grid,
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
	# param_list_task = list(ParameterGrid(param_grid_task))

	perf_valid_best = (np.inf if model_type == 'reg' else -np.inf)
	for idx, params in enumerate(param_list):  # debug: remove the [0:2]
		if verbose:
			print()
			print(
				'---- Parameter settings {}/{} -----:'.format(
					idx + 1, len(param_list)
				)
			)
			print(params)

		perf_valid, all_history, perf_eval, model = evaluate_parameters(
			G_train, y_train,
			G_valid, y_valid,
			params,
			estimator,
			model_type,
			# fit_history=history,
			params_idx=str(idx),
			verbose=verbose,
			**kwargs,
		)

		# Update the best parameters:
		if check_if_valid_better(perf_valid, perf_valid_best, model_type):
			best_model = model
			perf_valid_best = perf_valid
			params_best = copy.deepcopy(params)
			best_history = {
				**copy.deepcopy(all_history),
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
		G_train,
		y_train,
		best_model,
		perf_eval,
		y_scaler=y_scaler,
		model_type=model_type,
	)
	history_train = _init_all_history(fit=False)
	_update_history_1fold(history_train, None, pred_history_train)

	# Predict the valid set:
	perf_valid, y_pred_valid, y_true_valid, pred_history_valid = predict(
		G_valid,
		y_valid,
		best_model,
		perf_eval,
		y_scaler=y_scaler,
		model_type=model_type,
	)
	history_valid = _init_all_history(fit=False)
	_update_history_1fold(history_valid, None, pred_history_valid)

	# Predict the test set:
	perf_test, y_pred_test, y_true_test, pred_history_test = predict(
		G_test,
		y_test,
		best_model,
		perf_eval,
		y_scaler=y_scaler,
		model_type=model_type,
	)
	history_test = _init_all_history(fit=False)
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

	# Return the best model:
	return best_model, \
		perf_train, perf_valid, perf_test, \
		y_pred_train, y_pred_valid, y_pred_test, \
		best_history, history_train, history_valid, history_test, \
		params_best


def evaluate_vector_model(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		model_type='reg',
		descriptor='atom_bond_types',
		**kwargs
):
	# todo: these hyperparameters may be tuned further.

	if kwargs.get('model').startswith('vc:lr'):
		param_grid = {}

		from sklearn.linear_model import LinearRegression
		estimator = LinearRegression

	elif kwargs.get('model').startswith('vc:gpr'):
		import sklearn.gaussian_process as gp
		param_grid = {
			'alpha': np.linspace(1, 10, 10),
			'kernel': [
				gp.kernels.Matern(nu=1.5),
				gp.kernels.Matern(nu=2.5),
				gp.kernels.RationalQuadratic(),
				# gp.kernels.Exponentiation('none', 2),
				gp.kernels.Exponentiation(gp.kernels.ConstantKernel(), 2),
				gp.kernels.Exponentiation(gp.kernels.DotProduct(), 2),
				gp.kernels.Exponentiation(gp.kernels.RationalQuadratic(), 2),
				gp.kernels.DotProduct(),
				gp.kernels.ConstantKernel(), #kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5)) * gp.kernels.RBF(1.0, (1e-5, 1e5))
				gp.kernels.RBF(),

			],
		}

		estimator = gp.GaussianProcessRegressor

	elif kwargs.get('model').startswith('vc:krr'):
		param_grid = {
			'alpha': np.logspace(-6, 0, 7),
			'kernel': ['rbf'],
			'gamma': np.logspace(-5, 0, 6),
			# 'alpha': np.linspace(1, 10, 10),
			# 'kernel': ['rbf', 'laplacian', 'polynomial', 'sigmoid'],
			# 'gamma': np.linspace(0.01, 1, 10),
		}

		from sklearn.kernel_ridge import KernelRidge
		estimator = KernelRidge

	elif kwargs.get('model').startswith('vc:svr'):
		param_grid = {
			'C': np.logspace(0, 6, 7),
			'gamma': np.logspace(-4, 0, 5),
		}

		from sklearn.svm import SVR
		estimator = SVR

	elif kwargs.get('model').startswith('vc:rf'):
		param_grid = {
			'n_estimators': [100, 150, 200],
			'max_depth': [3, 4, 5, None],
			'max_features': [None, 'sqrt', 'log2'],
		}

		from sklearn.ensemble import RandomForestRegressor
		estimator = RandomForestRegressor

	elif kwargs.get('model').startswith('vc:xgb'):
		param_grid = {
			'learning_rate': [0.01, 0.1, 0.2, 0.4],
			'max_depth': [3, 4, 5, 6],
			'min_child_weight': [1, 5, 10],
			'n_estimators': [100, 150, 200],
		}

		from xgboost import XGBRegressor
		estimator = XGBRegressor

	elif kwargs.get('model').startswith('vc:knn'):
		param_grid = {
			'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
			'metric': ['euclidean', 'manhattan'],
			'weights': ['uniform', 'distance'],
			# 'p': [1, 2],
		}

		from sklearn.neighbors import KNeighborsRegressor
		estimator = KNeighborsRegressor

	else:
		raise ValueError(
			'Unknown model: {}.'.format(kwargs.get('model'))
		)

	return model_selection_for_vector_model(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		estimator,
		param_grid,
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


def _init_all_history(fit=True, pred=True):
	keys = []
	if fit:
		keys += ['fit_time']
	if pred:
		keys += ['pred_time']
	all_history = {key: AverageMeter(keep_all=False) for key in keys}
	return all_history


def _print_time_info(
		best_history,
		history_train,
		history_valid,
		history_test
):
	print('Training time:')
	print(
		'  Model: total {:.3f}\tper pair {:.9f}'.format(
			best_history['fit_time'].sum,
			best_history['fit_time'].avg
		)
	)
	print('Prediction time:')
	print(
		'  Train:\ttotal {:.3f}\tper data {:.9f}'.format(
			history_train['pred_time'].sum,
			history_train['pred_time'].avg
		)
	)
	print(
		'  Valid:\ttotal {:.3f}\tper data {:.9f}'.format(
			history_valid['pred_time'].sum,
			history_valid['pred_time'].avg
		)
	)
	print(
		'  Test:\ttotal {:.3f}\tper data {:.9f}'.format(
			history_test['pred_time'].sum,
			history_test['pred_time'].avg
		)
	)
