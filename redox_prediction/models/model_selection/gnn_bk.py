"""
model_selection



@Author: linlin
@Date: 20.05.23
"""
import copy
import multiprocessing
import os
import pickle
import time

import numpy as np

# matplotlib.use('Agg')
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

import torch
from torch_geometric.loader import DataLoader

from redox_prediction.models.evaluation.gnn import predict_gnn
from redox_prediction.utils.logging import AverageMeter


def fit_model(
		train_loader: DataLoader,
		estimator: torch.nn.Module,
		params: dict,
		model_type: str,
		n_epochs: int,
		device: torch.device,
		valid_loader: DataLoader = None,
		epochs_per_eval: int = None,
		plot_loss: bool = False,
		print_interval: int = 1,  # @debug: to change as needed.
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

	# Train metric evaluation model:
	load_state_dict_from = os.path.join(
		kwargs.get('output_dir'),
		'split_{}.params_{}.fold_{}.model_states.pt'.format(
			kwargs.get('split_idx'), kwargs.get('params_idx'), kwargs.get('fold_idx')
		)
	)
	os.makedirs(os.path.dirname(load_state_dict_from), exist_ok=True)
	# Train task-specific model:
	from redox_prediction.models.evaluation.gnn import fit_model_gnn
	model, history, valid_losses, valid_metrics = fit_model_gnn(
		model,
		train_loader,
		valid_loader,
		model_type=model_type,
		**{
			**kwargs,
			'n_epochs': n_epochs,
			'learning_rate': params['learning_rate'],
			'epochs_per_eval': epochs_per_eval,
			'device': device,
			'load_state_dict_from': load_state_dict_from
		}
	)

	return model, history, valid_losses, valid_metrics


def evaluate_parameters(
		dataset_app,
		y_app,
		params,
		kf,
		estimator,
		model_type,
		device,
		n_epochs=500,  # TODO: 500
		epochs_per_eval: int = 10,
		verbose=True,
		**kwargs
):
	all_history = _init_all_history()
	perf_valid_list, valid_loss_list = [], []

	# for each inner CV fold:
	for idx, (train_index, valid_index) in enumerate(kf.split(dataset_app, y_app)):

		# # For debugging only:  # TODO: comment this.
		# if idx > 1:
		# 	break

		print('\nTrial: {}/{}'.format(idx + 1, kf.get_n_splits()))

		# split the dataset into train and validation sets:
		dataset_train = dataset_app[train_index]
		train_loader = DataLoader(
			dataset_train, batch_size=params['batch_size'], shuffle=False
		)
		dataset_valid = dataset_app[valid_index]
		valid_loader = DataLoader(
			dataset_valid, batch_size=params['batch_size'], shuffle=False
		)

		# Train the model:
		start_time = time.time()
		model, history, valid_losses, valid_metrics = fit_model(
			train_loader,
			estimator,
			params,
			model_type,
			n_epochs,
			device,
			valid_loader=valid_loader,
			epochs_per_eval=epochs_per_eval,
			verbose=verbose,
			plot_loss=True,
			**{**kwargs, 'fold_idx': idx}
		)
		if verbose:
			print('Total time for training in this trial: {:.3f} s.'.format(
				time.time() - start_time)
			)
		perf_valid_list.append(valid_metrics)
		valid_loss_list.append(valid_losses)

		_update_history_1fold(all_history, history, None)

	# Average the metric over the inner CV folds for each epoch:
	perf_valid_list = np.array(perf_valid_list)
	perf_valid_list = np.mean(perf_valid_list, axis=0)

	# Get the index of the best metric and the corresponding number of epochs:
	if model_type == 'reg':
		best_idx = np.argmin(perf_valid_list)
	elif model_type == 'classif':
		best_idx = np.argmax(perf_valid_list)
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')
	perf_valid = perf_valid_list[best_idx]
	best_n_epochs = (best_idx + 1) * epochs_per_eval

	# Show the best performance and the corresponding number of epochs:
	if verbose:
		print()
		print('Best valid performance: {:.3f}'.format(perf_valid))
		print('Best n_epochs: {}'.format(best_n_epochs))

	return perf_valid, all_history, best_n_epochs


def model_selection_for_gnn(
		G_app, y_app, G_test, y_test,
		estimator,
		param_grid,
		model_type,
		n_epochs=1000,  # TODO: 500
		parallel=False,
		n_jobs=multiprocessing.cpu_count(),
		read_resu_from_file: int = 1,
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
		keep_nx_graphs=False,
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
	# for idx, params in get_iters(
	# 		enumerate(param_list),  # @TODO: remove the [0:2]
	# 		desc='model selection for the GNN model',
	# 		file=sys.stdout,
	# 		length=len(param_list),
	# 		verbose=True
	# ):
	for idx, params in enumerate(param_list):  # @TODO: remove the [0:2]
		if verbose:
			print()
			print('---- Parameter settings {}/{} -----:'.format(
				idx + 1, len(param_list)))
			print(params)

		perf_valid, all_history, best_n_epochs = evaluate_parameters(
			dataset_app, y_app, params, kf, estimator, model_type, device,
			n_epochs=n_epochs,
			verbose=verbose,
			params_idx=str(idx),
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)

		# Update the best parameters:
		if check_if_valid_better(perf_valid, perf_valid_best, model_type):
			perf_valid_best = perf_valid
			params_best = copy.deepcopy(params)
			best_best_n_epochs = best_n_epochs
			best_history = copy.deepcopy(all_history)

	# Refit the best model on the whole dataset:
	print('\n---- Start refitting the best model on the whole valid dataset...')
	metric = ('rmse' if model_type == 'reg' else 'accuracy')
	app_loader = DataLoader(
		dataset_app, batch_size=params_best['batch_size'],
		shuffle=False
	)
	model, history, _, _ = fit_model(
		app_loader,
		estimator,
		params_best,
		model_type,
		best_best_n_epochs,
		device,
		valid_loader=None,
		verbose=verbose,
		plot_loss=True,
		**{**kwargs, 'params_idx': 'refit', 'read_resu_from_file': read_resu_from_file}
	)
	perf_app, y_pred_app, y_true_app, pred_history_app = predict_gnn(
		app_loader,
		model,
		metric,
		device,
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		n_classes=kwargs.get('n_classes'),
	)
	history_app = _init_all_history(valid=False)
	_update_history_1fold(history_app, history, pred_history_app)

	# Predict the test set:
	test_loader = DataLoader(
		dataset_test, batch_size=params_best['batch_size'], shuffle=False
	)
	perf_test, y_pred_test, y_true_test, pred_history_test = predict_gnn(
		test_loader,
		model,
		metric,
		device,
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		n_classes=kwargs.get('n_classes'),
	)
	history_test = _init_all_history(train=False, valid=False)
	_update_history_1fold(
		history_test, None, pred_history_test, rm_unused_keys=True
	)

	# Print out the best performance:
	if verbose:
		print('\nPerformance on the refitted model:')
		print('Best app performance: {:.3f}'.format(perf_app))
		print('Best test performance: {:.3f}'.format(perf_test))
		print('Best number of epochs: {}'.format(best_best_n_epochs))
		_print_time_info(history_app, history_test)
		print('Best params: ', params_best)

	# Return the best model:
	return model, perf_app, perf_test, y_pred_app, y_pred_test, best_history, \
		history_app, history_test, {**params_best, 'n_epochs': best_best_n_epochs}


def evaluate_gnn(
		G_app, y_app, G_test, y_test,
		model_type='reg',
		descriptor='atom_bond_types',
		**kwargs
):
	# @todo ['softmax', 'log_softmax'],
	clf_activation = (
		'sigmoid' if kwargs.get('n_classes') == 2 else 'log_softmax')

	if kwargs.get('embedding') == 'nn:gcn':
		# Get parameter grid:
		param_grid = {
			'hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'agg_activation': ['relu', 'tanh'],
			'readout': ['mean'],
			'predictor_hidden_feats': [128],  # [32, 64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': [clf_activation],
			'batch_size': [32],
		}

		from redox_prediction.models.nn.gcn import GCN
		estimator = GCN

	elif kwargs.get('embedding') == 'nn:gat':
		# Get parameter grid:
		param_grid = {
			'hidden_feats': [32, 64],  # [128],  # [64, 128],
			'n_heads': [4, 8],  # [4, 8],
			'concat_heads': [True],  # [True, False],
			'message_steps': [2, 3, 4],  # [5],  # [2, 3, 4],
			'attention_drop': [0, 0.5],  # [0., 0.5],
			'agg_activation': ['relu', 'tanh'],
			'readout': ['mean'],
			'predictor_hidden_feats': [64, 128],  # [128],  # [64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': [clf_activation],
			'batch_size': [32, 64],
		}

		from redox_prediction.models.nn.gat import GAT
		estimator = GAT

	return model_selection_for_gnn(
		G_app, y_app, G_test, y_test,
		estimator,
		param_grid,
		model_type,
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


def _init_all_history(train=True, valid=True, pred=True):
	all_history = {}
	if train:
		all_history.update({k: AverageMeter(keep_all=False) for k in [
			'batch_time_train', 'data_time_train', 'epoch_time_fit'
		]})
	if valid:
		all_history.update({k: AverageMeter(keep_all=False) for k in [
			'batch_time_valid', 'data_time_valid', 'epoch_time_fit'
		]})
	if pred:
		all_history.update({k: AverageMeter(keep_all=False) for k in [
			'batch_time_pred', 'data_time_pred'
		]})
	return all_history


def _update_history_1fold(
		all_history: dict,
		fit_history: dict = None,
		pred_history: dict = None,
		rm_unused_keys: bool = False
):
	if fit_history is not None:
		for key, val in fit_history.items():
			if key in all_history:
				all_history[key].update(val)
	if pred_history is not None:
		for key, val in pred_history.items():
			if key + '_pred' in all_history:
				all_history[key + '_pred'].update(val)


def _print_time_info(history_app, history_test):
	print('Training time:')
	print('  Batch App:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['batch_time_train'].sum,
		history_app['batch_time_train'].avg
	))
	print('  Data App:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['data_time_train'].sum,
		history_app['data_time_train'].avg
	))
	print('  Epoch Fit:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['epoch_time_fit'].sum,
		history_app['epoch_time_fit'].avg
	))
	print('Prediction time:')
	print('  Batch App:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['batch_time_pred'].sum,
		history_app['batch_time_pred'].avg
	))
	print('  Data App:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_app['data_time_pred'].sum,
		history_app['data_time_pred'].avg
	))
	print('  Batch Test:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_test['batch_time_pred'].sum,
		history_test['batch_time_pred'].avg
	))
	print('  Data Test:\ttotal {:.3f}\tavg {:.9f}'.format(
		history_test['data_time_pred'].sum,
		history_test['data_time_pred'].avg
	))
