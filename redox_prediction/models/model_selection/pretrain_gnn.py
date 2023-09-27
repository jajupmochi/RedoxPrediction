"""
model_selection



@Author: linlin
@Date: 20.05.23
"""
import copy
import multiprocessing
import sys
import os
import pickle

from typing import Iterable, Tuple, Union

import numpy as np

# matplotlib.use('Agg')
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid
from gklearn.utils.iters import get_iters

import torch
from torch_geometric.loader import DataLoader

from redox_prediction.dataset.nn.pairwise import PairwiseDataset
from redox_prediction.dataset.nn.utils import transform_dataset_task
from redox_prediction.utils.logging import AverageMeter


def predict(
		data_loader: DataLoader,
		model: Tuple[torch.nn.Module, Union[torch.nn.Module]],
		metric: str,
		infer_mode: str,
		model_type: str = 'reg',
		y_scaler: object = None,
		n_classes: int = None,
		device: torch.device = None,
		**kwargs
):
	if infer_mode == 'pretrain+refine':
		from redox_prediction.models.evaluation.refinement import predict_refine
		return predict_refine(
			data_loader, model[1], metric, device, model_type, y_scaler,
			n_classes, **kwargs
		)
	else:
		raise ValueError('Invalid inference model_type "{}".'.format(infer_mode))


def fit_model_task(
		train_dataset: Union[DataLoader, Iterable],
		valid_dataset: Union[DataLoader, Iterable],
		infer_mode: str = 'pretrain+refine',
		model_type: str = 'reg',
		model: torch.nn.Module = None,
		**kwargs
):
	"""
	Train a model for a specific task.

	Parameters
	----------
	train_dataset : Iterable
		Training dataset.
	valid_dataset : Iterable
		Validation dataset.
	infer_mode : str, optional
		Mode of inference. The default is 'pretrain+refine'.
	model_type : str, optional
		Type of the model. The default is 'reg'.
	model : torch.nn.Module, optional
		The pre-trained model to be refined, only used when infer_mode is
		'pretrain+refine'. The default is None.
	**kwargs : TYPE
		Other keyword arguments.
	"""
	# Refine the pre-trained model for the task:
	if infer_mode == 'pretrain+refine':
		from redox_prediction.models.evaluation.refinement import fit_model_refine
		return fit_model_refine(
			model,
			train_dataset,
			valid_dataset,
			# n_epochs=n_epochs,
			# device=device,
			model_type=model_type,
			**kwargs
		)
	else:
		raise ValueError('Invalid inference model_type "{}".'.format(infer_mode))


def fit_model(
		train_loader: DataLoader,
		ds_train_task: Union[DataLoader, Iterable],
		estimator: torch.nn.Module,
		params: dict,
		model_type: str,
		n_epochs: int,
		n_epochs_task: int,
		device: torch.device,
		valid_loader: DataLoader = None,
		ds_valid_task: Union[DataLoader, Iterable] = None,
		return_embeddings: bool = False,
		plot_loss: bool = False,
		print_interval: int = 1,  # @TODO: to change as needed.
		verbose: bool = False,
		**kwargs
):
	# Initialize the model:
	model = estimator(
		in_feats=train_loader.dataset.num_node_features,
		edge_dim=train_loader.dataset.num_edge_features,
		normalize=True,
		bias=True,
		feat_drop=0.,
		kernel_drop=0.,
		predictor_batchnorm=False,
		# @TODO #(True if model_type == 'classif' else False),
		n_classes=kwargs.get('n_classes'),
		mode=('regression' if model_type == 'reg' else 'classification'),
		device=device,
		**params
	).to(device)
	print(model)

	# Get parameters for training:
	# TODO
	# Train metric evaluation model:
	from redox_prediction.models.evaluation.paiwise import fit_model_pairwise
	load_state_dict_from = os.path.join(
		kwargs.get('output_dir'),
		'split_{}.params_{}.fold_{}.metric_model_states.pt'.format(
			kwargs.get('split_idx'), kwargs.get('params_idx'), kwargs.get('fold_idx')
		)
	)
	os.makedirs(os.path.dirname(load_state_dict_from), exist_ok=True)
	model, history, best_n_epochs = fit_model_pairwise(
		model,
		train_loader,
		valid_loader,
		n_epochs=n_epochs,
		device=device,
		**{**kwargs, 'load_state_dict_from': load_state_dict_from}
	)

	# Train task-specific model:
	infer_mode = kwargs.get('infer')
	load_state_dict_from = load_state_dict_from.replace(
		'metric_model_', 'task_model_'
	)
	model_task, history_task, best_n_epochs_task = fit_model_task(
		ds_train_task,
		ds_valid_task,
		infer_mode=infer_mode,
		model_type=model_type,
		model=(model if infer_mode == 'pretrain+refine' else None),
		**{
			**kwargs, 'n_epochs': n_epochs_task, 'device': device,
			'load_state_dict_from': load_state_dict_from
		}
	)

	return (model, model_task), (history, history_task), (
		best_n_epochs, best_n_epochs_task
	)


def evaluate_parameters(
		dataset_app,
		y_app,
		params,
		kf,
		estimator,
		model_type,
		device,
		n_epochs=10,  # TODO: 500
		verbose=True,
		**kwargs
):
	all_history = _init_all_history()
	perf_valid_list, best_n_epochs_metric, best_n_epochs_task = [], [], []

	# for each inner CV fold:
	for idx, (train_index, valid_index) in enumerate(kf.split(dataset_app, y_app)):

		# # For debugging only:  # TODO: comment this.
		# if idx > 1:
		# 	break

		print('\nTrial: {}/{}'.format(idx + 1, kf.get_n_splits()))

		# split the dataset into train and validation sets:
		dataset_train = dataset_app[train_index]
		train_loader = DataLoader(
			PairwiseDataset(dataset_train), batch_size=params['batch_size'],
			shuffle=False
		)
		dataset_valid = dataset_app[valid_index]
		valid_loader = DataLoader(
			PairwiseDataset(dataset_valid), batch_size=params['batch_size'],
			shuffle=False
		)
		ds_train_task = transform_dataset_task(
			dataset_train, kwargs['infer'], **params, **kwargs)
		ds_valid_task = transform_dataset_task(
			dataset_valid, kwargs['infer'], **params, **kwargs)

		# Train the model:
		model, history, best_n_epochs = fit_model(
			train_loader,
			ds_train_task,
			estimator,
			params,
			model_type,
			n_epochs,
			n_epochs,
			device,
			valid_loader=valid_loader,
			ds_valid_task=ds_valid_task,
			verbose=verbose,
			plot_loss=True,
			**{**kwargs, 'fold_idx': idx}
		)
		best_n_epochs_metric.append(best_n_epochs[0])
		best_n_epochs_task.append(best_n_epochs[1])

		# Predict the validation set:
		metric = ('rmse' if model_type == 'reg' else 'accuracy')
		perf_valid, y_pred_valid, y_true_valid, predict_history = predict(
			ds_valid_task,
			model,
			metric,
			infer_mode=kwargs['infer'],
			model_type=model_type,
			y_scaler=kwargs['y_scaler'],
			n_classes=kwargs.get('n_classes'),
			device=device,
			# **kwargs
		)
		perf_valid_list.append(perf_valid)

		_update_history_1fold(all_history, history, predict_history)

	# Average the performance over the inner CV folds:
	perf_valid = np.mean(perf_valid_list)

	# Show the mean and standard deviation of the best number of epochs:
	ave_best_n_epochs_metric = np.mean(best_n_epochs_metric)
	ave_best_n_epochs_task = np.mean(best_n_epochs_task)
	if verbose:
		print()
		print('best_n_epochs_metric: {} +/- {}'.format(
			ave_best_n_epochs_metric, np.std(best_n_epochs_metric)))
		print('best_n_epochs_task: {} +/- {}'.format(
			ave_best_n_epochs_task, np.std(best_n_epochs_task)))

	return perf_valid, all_history, (
		int(ave_best_n_epochs_metric), int(ave_best_n_epochs_task)
	)


def model_selection_for_gnn(
		G_app, y_app, G_test, y_test,
		estimator,
		param_grid,
		model_type,
		# fit_test=False,
		n_epochs=500,
		parallel=False,
		n_jobs=multiprocessing.cpu_count(),
		read_gm_from_file=False,
		verbose=True,
		**kwargs
):
	# Scale the targets:
	# @TODO: it is better to fit scaler by training set rather than app set.
	# @TODO: use minmax or log instead?
	if model_type == 'reg':
		from sklearn.preprocessing import StandardScaler
		y_scaler = StandardScaler().fit(np.reshape(y_app, (-1, 1)))
		y_app = y_scaler.transform(np.reshape(y_app, (-1, 1)))
		y_test = y_scaler.transform(np.reshape(y_test, (-1, 1)))
	elif model_type == 'classif':
		# ensure that the labels are in the range [0, n_classes - 1].
		# This is important for classification with Sigmoid and BCELoss.
		# If the labels are not in this range, the loss might be negative.
		from sklearn.preprocessing import LabelEncoder
		y_scaler = LabelEncoder().fit(y_app)
		y_app = y_scaler.transform(y_app)
		y_test = y_scaler.transform(y_test)
		# Ensure the values are floats:
		y_app = y_app.astype(float)
		y_test = y_test.astype(float)  # @TODO: is this necessary?
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')
	kwargs['y_scaler'] = y_scaler
	y_app, y_test = np.ravel(y_app), np.ravel(y_test)

	# Convert NetworkX graphs to PyTorch-Geometric compatible dataset:
	from redox_prediction.dataset.nn.nx import NetworkXGraphDataset
	dataset = NetworkXGraphDataset(
		G_app + G_test, np.concatenate((y_app, y_test)),
		node_label_names=kwargs.get('node_labels'),
		edge_label_names=kwargs.get('edge_labels'),
		node_attr_names=kwargs.get('node_attrs'),
		edge_attr_names=kwargs.get('edge_attrs'),
		keep_nx_graphs=True,
	)
	dataset_app = dataset[:len(G_app)]
	dataset_test = dataset[len(G_app):]

	# @TODO: change it back.
	device = torch.device(
		'cuda' if torch.cuda.is_available() else 'cpu'
	)  # torch.device('cpu') #
	if verbose:
		print('device:', device)

	# Set cross-validation method:
	if model_type == 'reg':
		kf = KFold(n_splits=5, shuffle=True, random_state=42)
	elif model_type == 'classif':
		kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')

	# Do cross-validation:
	param_list = list(ParameterGrid(param_grid))

	perf_valid_best = (np.inf if model_type == 'reg' else -np.inf)
	for idx, params in get_iters(
			enumerate(param_list),  # debug: remove the [0:2]
			desc='model selection for the GNN model',
			file=sys.stdout,
			length=len(param_list),
			verbose=True
	):
		if verbose:
			print()
			print('---- Parameter settings {}/{} -----:'.format(
				idx + 1, len(param_list)))
			print(params)

		# Initialize the metric matrix:
		metric_matrix = _init_metric_matrix(len(dataset), idx, **kwargs)

		perf_valid, all_history, best_n_epochs = evaluate_parameters(
			dataset_app, y_app, params, kf, estimator, model_type, device,
			n_epochs=n_epochs,
			verbose=verbose,
			params_idx=str(idx),
			metric_matrix=metric_matrix,
			**kwargs
		)

		# Update the best parameters:
		if check_if_valid_better(perf_valid, perf_valid_best, model_type):
			perf_valid_best = perf_valid
			params_best = copy.deepcopy(params)
			metric_matrix_best = metric_matrix
			best_best_n_epochs = best_n_epochs
			best_history = copy.deepcopy(all_history)

	# Refit the best model on the whole dataset:
	print('\n---- Start refitting the best model on the whole valid dataset...')
	metric = ('rmse' if model_type == 'reg' else 'accuracy')
	app_loader = DataLoader(
		PairwiseDataset(dataset_app), batch_size=params_best['batch_size'],
		shuffle=False
	)
	ds_app_task = transform_dataset_task(
		dataset_app, kwargs['infer'], **params_best, **kwargs
	)
	model, history, best_n_epochs = fit_model(
		app_loader,
		ds_app_task,
		estimator,
		params_best,
		model_type,
		best_best_n_epochs[0],
		best_best_n_epochs[1],
		device,
		valid_loader=None,
		verbose=verbose,
		plot_loss=True,
		**{**kwargs, 'params_idx': 'refit', 'metric_matrix': metric_matrix_best}
	)
	perf_app, y_pred_app, y_true_app, pred_history_app = predict(
		ds_app_task,
		model,
		metric,
		infer_mode=kwargs['infer'],
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		n_classes=kwargs.get('n_classes'),
		device=device,
	)
	history_app = _init_all_history()
	_update_history_1fold(history_app, history, pred_history_app)

	# Predict the test set:
	ds_test_task = transform_dataset_task(
		dataset_test, kwargs['infer'], **params_best, **kwargs
	)
	perf_test, y_pred_test, y_true_test, pred_history_test = predict(
		ds_test_task,
		model,
		metric,
		infer_mode=kwargs['infer'],
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		n_classes=kwargs.get('n_classes'),
		device=device,
	)
	history_test = _init_all_history()
	_update_history_1fold(
		history_test, None, pred_history_test, rm_unused_keys=True
	)

	# Print out the best performance:
	if verbose:
		print('\nPerformance on the refitted model:')
		print('Best app performance: {:.3f}'.format(perf_app))
		print('Best test performance: {:.3f}'.format(perf_test))
		_print_time_info(history_app, history_test)
		print('Best params: ', params_best)

	# Return the best model:
	return model, perf_app, perf_test, y_pred_app, y_pred_test, best_history, \
		history_app, history_test, params_best


def evaluate_gnn(
		G_app, y_app, G_test, y_test,
		model_type='reg', unlabeled=False,
		descriptor='atom_bond_types',
		# fit_test=False,
		**kwargs
):
	# @todo ['softmax', 'log_softmax'],
	clf_activation = (
		'sigmoid' if kwargs.get('n_classes') == 2 else 'log_softmax')
	# Get parameter grid:
	param_grid = {
		'hidden_feats': [64],  # [128],  # [64, 128],
		'n_heads': [4],  # [4, 8],
		'concat_heads': [True],  # [True, False],
		'message_steps': [3],  # [5],  # [2, 3, 4],
		'attention_drop': [0.5],  # [0., 0.5],
		'agg_activation': ['tanh', 'relu'],
		'readout': ['mean'],
		'predictor_hidden_feats': [64],  # [128],  # [64, 128],
		'predictor_n_hidden_layers': [1],
		'predictor_activation': ['relu'],
		'predictor_clf_activation': [clf_activation],
		'batch_size': [64],
	}

	from redox_prediction.models.nn.gat import GAT
	estimator = GAT
	return model_selection_for_gnn(
		G_app, y_app, G_test, y_test, estimator, param_grid,
		model_type,
		# fit_test,
		**kwargs
	)


#%%


def check_if_valid_better(perf_valid, perf_valid_best, model_type):
	if model_type == 'reg':
		return perf_valid < perf_valid_best
	elif model_type == 'classif':
		return perf_valid > perf_valid_best
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')


def _init_metric_matrix(n_graphs, idx, **kwargs):
	# Load the metric matrix from file if it exists:
	fn_mat = os.path.join(
		kwargs.get('output_dir'), 'metric_matrix.params_{}.pkl'.format(idx)
	)
	if os.path.exists(fn_mat) and os.path.getsize(fn_mat) > 0:
		metric_matrix = pickle.load(open(fn_mat, 'rb'))
	else:
		# Initialize the metric matrix with nan values:
		metric_matrix = np.full((n_graphs, n_graphs), np.nan)

	return metric_matrix


def _init_all_history():
	all_history = {'fit_metric': {}, 'fit_task': {}, 'predict': {}}
	# Update the fitting history for the metric model:
	all_history['fit_metric'] = {k: AverageMeter(keep_all=False) for k in [
		'batch_time_train', 'batch_time_valid',
		'data_time_train', 'data_time_valid',
		'metric_time_compute_train', 'metric_time_compute_valid',
		'metric_time_total_train', 'metric_time_total_valid',
		'metric_time_load_train', 'metric_time_load_valid',
		'epoch_time'
	]}
	# Update the fitting history for the task model:
	all_history['fit_task'] = {k: AverageMeter(keep_all=False) for k in [
		'batch_time_train', 'batch_time_valid',
		'data_time_train', 'data_time_valid',
		'epoch_time'
	]}
	# Update the prediction history:
	all_history['predict'] = {k: AverageMeter(keep_all=False) for k in [
		'batch_time', 'data_time'
	]}

	return all_history


def _update_history_1fold(
		all_history: dict,
		fit_history: Tuple[dict, dict] = None,
		predict_history: dict = None,
		rm_unused_keys: bool = False
):
	def _update_1history(key, history):
		if history is None:
			if rm_unused_keys:
				del all_history[key]
			return

		for k, v in all_history[key].items():
			if k in history:
				meter = history[k]
				v.update(meter)
			else:
				# Remove the keys that are not in the current fold:
				if rm_unused_keys:
					del all_history[key][k]


	# Update the fitting history for the metric model:
	_update_1history('fit_metric', (None if fit_history is None else fit_history[0]))
	# Update the fitting history for the task model:
	_update_1history('fit_task', (None if fit_history is None else fit_history[1]))
	# Update the prediction history:
	_update_1history('predict', predict_history)


def _print_time_info(history_app, history_test):
	print('Training time:')
	print('  Metric model:')
	print('    Batch:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_metric']['batch_time_train'].sum,
		history_app['fit_metric']['batch_time_train'].avg
	))
	print('    Data:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_metric']['data_time_train'].sum,
		history_app['fit_metric']['data_time_train'].avg
	))
	print('    Metric Total:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_metric']['metric_time_total_train'].sum,
		history_app['fit_metric']['metric_time_total_train'].avg
	))
	print('    Metric Compute:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_metric']['metric_time_compute_train'].sum,
		history_app['fit_metric']['metric_time_compute_train'].avg
	))
	print('    Metric Load:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_metric']['metric_time_load_train'].sum,
		history_app['fit_metric']['metric_time_load_train'].avg
	))
	print('    Epoch:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_metric']['epoch_time'].sum,
		history_app['fit_metric']['epoch_time'].avg
	))
	print('  Task model:')
	print('    Batch:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_task']['batch_time_train'].sum,
		history_app['fit_task']['batch_time_train'].avg
	))
	print('    Data:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_task']['data_time_train'].sum,
		history_app['fit_task']['data_time_train'].avg
	))
	print('    Epoch:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['fit_task']['epoch_time'].sum,
		history_app['fit_task']['epoch_time'].avg
	))
	print('Prediction time:')
	print('  App:')
	print('    Batch:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['predict']['batch_time'].sum,
		history_app['predict']['batch_time'].avg
	))
	print('    Data:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['predict']['data_time'].sum,
		history_app['predict']['data_time'].avg
	))
	print('  Test:')
	print('    Batch:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_test['predict']['batch_time'].sum,
		history_test['predict']['batch_time'].avg
	))
	print('    Data:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_test['predict']['data_time'].sum,
		history_test['predict']['data_time'].avg
	))
