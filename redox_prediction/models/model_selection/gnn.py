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
		max_epochs: int,
		device: torch.device,
		valid_loader: DataLoader = None,
		epochs_per_eval: int = None,
		plot_loss: bool = False,
		print_interval: int = 1,  # debug: to change as needed.
		verbose: bool = False,
		**kwargs
):
	# Initialize the model:
	edge_dim = train_loader.dataset.num_edge_features
	edge_dim = (None if edge_dim == 0 else edge_dim)
	model = estimator(
		in_feats=train_loader.dataset.num_node_features,
		edge_dim=edge_dim,
		normalize=True,
		bias=True,
		feat_drop=0.,
		kernel_drop=0.,
		predictor_batchnorm=False,
		# @TODO #(True if model_type == 'classif' else False),
		n_classes=kwargs.get('n_classes'),
		mode=('regression' if model_type == 'reg' else 'classification'),
		**params
	).to(device)
	print(model)

	# Train metric evaluation model:
	load_state_dict_from = os.path.join(
		kwargs.get('output_dir'),
		'split_{}.params_{}.model_states.pt'.format(
			kwargs.get('split_idx'), kwargs.get('params_idx'),
		)
	)
	os.makedirs(os.path.dirname(load_state_dict_from), exist_ok=True)
	# Train task-specific model:
	# Replace 'model' key with 'model_name'
	kwargs = kwargs.copy()
	if 'model' in kwargs:
		kwargs['model_name'] = kwargs.pop('model')
	from redox_prediction.models.evaluation.gnn import fit_model_gnn
	model, history, valid_losses, valid_metrics = fit_model_gnn(
		model,
		train_loader,
		valid_loader,
		model_type=model_type,
		**{
			**kwargs,
			'max_epochs': max_epochs,
			'learning_rate': params['lr'],
			'predictor_clf_activation': params['predictor_clf_activation'],
			'epochs_per_eval': epochs_per_eval,
			'device': device,
			'load_state_dict_from': load_state_dict_from
		}
	)

	return model, history, valid_losses, valid_metrics


def evaluate_parameters(
		dataset_train,
		dataset_valid,
		params,
		estimator,
		model_type,
		device,
		max_epochs=800,
		if_tune_n_epochs: bool = False,
		epochs_per_eval: int = 10,
		verbose=True,
		**kwargs
):
	all_history = _init_all_history()

	# Construct data loader:
	train_loader = DataLoader(
		dataset_train, batch_size=params['batch_size'], shuffle=False
	)
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
		max_epochs,
		device,
		valid_loader=valid_loader,
		epochs_per_eval=epochs_per_eval,
		verbose=verbose,
		plot_loss=True,
		**kwargs
	)
	if verbose:
		print(
			'Total time for training in this trial: {:.3f} s.'.format(
				time.time() - start_time
			)
		)
	perf_valid_list = np.array(valid_metrics)
	valid_loss_list = np.array(valid_losses)

	_update_history_1fold(all_history, history, None)

	# if if_tune_n_epochs:
	# 	# Average the metric over the inner CV folds for each epoch:
	# 	perf_valid_list = np.array(perf_valid_list)
	# 	perf_valid_list = np.mean(perf_valid_list, axis=0)
	#
	# Get the last index of the best metric and the corresponding number of epochs:
	if model_type == 'reg':
		best_idx = np.where(perf_valid_list == np.min(perf_valid_list))[0][-1]
	elif model_type == 'classif':
		best_idx = np.where(perf_valid_list == np.max(perf_valid_list))[0][-1]
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')
	perf_valid = perf_valid_list[best_idx]
	best_n_epochs = (best_idx + 1) * epochs_per_eval
	# else:
	# perf_valid = np.mean([v[-1] for v in perf_valid_list])
	# best_n_epochs = max_epochs

	# Show the best performance and the corresponding number of epochs:
	if verbose:
		print()
		print('Best valid performance: {:.3f}'.format(perf_valid))
		# if if_tune_n_epochs:  # todo
		print('Best n_epochs: {}'.format(best_n_epochs))
		# else:
		# 	print('Best n_epochs is set to max_epochs: {}'.format(max_epochs))

	return perf_valid, all_history, best_n_epochs, model


def model_selection_for_gnn(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		estimator,
		param_grid,
		model_type,
		max_epochs=800,
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
	y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)

	# Convert NetworkX graphs to PyTorch-Geometric compatible dataset:
	from redox_prediction.dataset.nn.nx import NetworkXGraphDataset
	dataset = NetworkXGraphDataset(
		G_train + G_valid + G_test, np.concatenate((y_train, y_valid, y_test)),
		node_label_names=kwargs.get('node_labels'),
		edge_label_names=kwargs.get('edge_labels'),
		node_attr_names=kwargs.get('node_attrs'),
		edge_attr_names=kwargs.get('edge_attrs'),
		keep_nx_graphs=False,
		to_one_hot=(False if '1hot' in kwargs.get('descriptor') else True),
	)
	dataset_train = dataset[:len(G_train)]
	dataset_valid = dataset[len(G_train):len(G_train) + len(G_valid)]
	dataset_test = dataset[len(G_train) + len(G_valid):]

	# @debug: change it back.
	device = torch.device(
		'cuda' if torch.cuda.is_available() else 'cpu'
	)  # torch.device('cpu') #
	if verbose:
		print('device:', device)

	# # Set cross-validation method:
	# if model_type == 'reg':
	# 	kf = KFold(n_splits=5, shuffle=True, random_state=42)
	# elif model_type == 'classif':
	# 	kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	# else:
	# 	raise ValueError('"model_type" must be either "reg" or "classif".')

	# Do cross-validation:
	param_list = list(ParameterGrid(param_grid))

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

		perf_valid, all_history, best_n_epochs, model = evaluate_parameters(
			dataset_train,
			dataset_valid,
			params,
			estimator,
			model_type,
			device,
			max_epochs=max_epochs,
			verbose=verbose,
			params_idx=str(idx),
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)

		# Update the best parameters:
		if check_if_valid_better(perf_valid, perf_valid_best, model_type):
			best_model = model
			perf_valid_best = perf_valid
			params_best = copy.deepcopy(params)
			best_best_n_epochs = best_n_epochs
			best_history = copy.deepcopy(all_history)

	# params_best = copy.deepcopy(params)
	# best_best_n_epochs = 970
	# break

	# # Refit the best model on the whole dataset:
	# print('\n---- Start refitting the best model on the whole valid dataset...')
	# metric = ('rmse' if model_type == 'reg' else 'accuracy')
	# app_loader = DataLoader(
	# 	dataset_app, batch_size=params_best['batch_size'],
	# 	shuffle=False
	# )
	# model, history, _, _ = fit_model(
	# 	app_loader,
	# 	estimator,
	# 	params_best,
	# 	model_type,
	# 	best_best_n_epochs,
	# 	device,
	# 	valid_loader=None,
	# 	verbose=verbose,
	# 	plot_loss=True,
	# 	**{
	# 		**kwargs, 'params_idx': 'refit',
	# 		'read_resu_from_file': read_resu_from_file
	# 	}
	# )
	# perf_app, y_pred_app, y_true_app, pred_history_app = predict_gnn(
	# 	app_loader,
	# 	model,
	# 	metric,
	# 	device,
	# 	model_type=model_type,
	# 	y_scaler=kwargs['y_scaler'],
	# 	predictor_clf_activation=params_best['predictor_clf_activation'],
	# )
	# history_app = _init_all_history(valid=False)
	# _update_history_1fold(history_app, history, pred_history_app)

	metric = ('mae' if model_type == 'reg' else 'accuracy')
	# Predict the train set:
	train_loader = DataLoader(
		dataset_train, batch_size=params_best['batch_size'], shuffle=False
	)
	perf_train, y_pred_train, y_true_train, pred_history_train = predict_gnn(
		train_loader,
		best_model,
		metric,
		device,
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		predictor_clf_activation=params_best['predictor_clf_activation'],
	)
	history_train = _init_all_history(train=False, valid=False)
	_update_history_1fold(
		history_train, None, pred_history_train, rm_unused_keys=True
	)

	# Predict the valid set:
	valid_loader = DataLoader(
		dataset_valid, batch_size=params_best['batch_size'], shuffle=False
	)
	perf_valid, y_pred_valid, y_true_valid, pred_history_valid = predict_gnn(
		valid_loader,
		best_model,
		metric,
		device,
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		predictor_clf_activation=params_best['predictor_clf_activation'],
	)
	history_valid = _init_all_history(train=False, valid=False)
	_update_history_1fold(
		history_valid, None, pred_history_valid, rm_unused_keys=True
	)

	# Predict the test set:
	test_loader = DataLoader(
		dataset_test, batch_size=params_best['batch_size'], shuffle=False
	)
	perf_test, y_pred_test, y_true_test, pred_history_test = predict_gnn(
		test_loader,
		best_model,
		metric,
		device,
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		predictor_clf_activation=params_best['predictor_clf_activation'],
	)
	history_test = _init_all_history(train=False, valid=False)
	_update_history_1fold(
		history_test, None, pred_history_test, rm_unused_keys=True
	)

	# Print out the best performance:
	if verbose:
		print('\nPerformance on the best model:')
		print('Best train performance: {:.3f}'.format(perf_train))
		print('Best valid performance: {:.3f}'.format(perf_valid))
		print('Best test performance: {:.3f}'.format(perf_test))
		print('Best number of epochs: {}'.format(best_best_n_epochs))
		_print_time_info(best_history, history_train, history_valid, history_test)
		print('Best params: ', params_best)

	# Return the best model:
	return model, perf_train, perf_valid, perf_test, \
		y_pred_train, y_pred_valid, y_pred_test, \
		best_history, history_train, history_valid, history_test, \
		{**params_best, 'n_epochs': best_best_n_epochs}


def evaluate_gnn(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		model_type='reg',
		descriptor='atom_bond_types',
		**kwargs
):
	# @todo ['softmax', 'log_softmax'],
	# clf_activation = (
	# 	'sigmoid' if kwargs.get('n_classes') == 2 else 'log_softmax')

	if kwargs.get('model') == 'nn:mpnn':
		# Get parameter grid:
		param_grid = {
			'lr': [10 ** -3, 10 ** -4],
			'hidden_feats': [32, 64],
			'edge_hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'aggr': ['add'],
			'root_weight': [True],
			# 'agg_activation': ['relu'],
			'readout': ['set2set'],
			'predictor_hidden_feats': [128, 512, 1024],  # [32, 64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': ['log_softmax'],
			'processing_steps': [3, 6, 9],
			'batch_size': [32, 64],
		}
		max_epochs = 2000

		from redox_prediction.models.nn.mpnn import MPNN
		estimator = MPNN

	elif kwargs.get('model') == 'nn:gcn':
		# Get parameter grid:
		param_grid = {
			'lr': [10 ** -3, 10 ** -4],
			'hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'agg_activation': ['relu'],
			'readout': ['mean'],
			'predictor_hidden_feats': [128, 512, 1024],  # [32, 64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
		}
		max_epochs = 2000  # debug

		from redox_prediction.models.nn.gcn import GCN
		estimator = GCN

	elif kwargs.get('model') == 'nn:dgcnn':
		# Get parameter grid:
		from redox_prediction.models.nn.dgcnn import get_sort_pooling_k
		# When ks consists of multiple 10s, keep only one of them:
		ks = sorted((set([
			get_sort_pooling_k(G_train + G_valid + G_test, perc, 10) for perc in [
				0.6, 0.9]
		])))
		param_grid = {
			'lr': [10 ** -3, 10 ** -4],
			'hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'k': ks,  # [10, 20],
			'agg_activation': ['tanh'],
			'readout': ['sort_pooling'],
			'predictor_hidden_feats': [128, 512, 1024],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_dropout': [0.5],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
			'dim_target': [len(set(np.concatenate((y_train, y_valid, y_test))))],
		}
		max_epochs = 2000

		from redox_prediction.models.nn.dgcnn import DGCNN
		estimator = DGCNN

	elif kwargs.get('model') == 'nn:gat':
		# Get parameter grid:
		param_grid = {
			'lr': [10 ** -3, 10 ** -4],
			'hidden_feats': [32, 64],  # [128],  # [64, 128],
			'n_heads': [4, 8],  # [4, 8],
			'concat_heads': [True],  # [True, False],
			'message_steps': [2, 3, 4],  # [5],  # [2, 3, 4],
			'attention_drop': [0, 0.5],  # [0., 0.5],
			# 'agg_activation': ['relu'],
			'readout': ['mean'],
			'predictor_hidden_feats': [128, 512],  # [128],  # [64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
		}
		max_epochs = 2000

		from redox_prediction.models.nn.gat import GAT
		estimator = GAT

	elif kwargs.get('model') == 'nn:gin':
		# Get parameter grid:
		param_grid = {
			'lr': [10 ** -3, 10 ** -4],
			'hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'train_eps': [True],
			# 'agg_activation': ['relu'],
			'aggregation': ['sum'],  # 'mean
			'readout': ['concat'],
			'predictor_hidden_feats': [128, 512, 1024],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_drop': [0.5, 0.],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
		}
		max_epochs = 2000

		from redox_prediction.models.nn.gin import GIN
		estimator = GIN

	elif kwargs.get('model') == 'nn:unimp':
		# Get parameter grid:
		param_grid = {
			'lr': [10 ** -3, 10 ** -4],
			'hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'n_heads': [4, 8],
			'concat_heads': [True],
			'beta': [False],  # True
			'dropout': [0.],  # 0.5
			# 'agg_activation': ['relu'],
			'readout': ['mean'],
			'predictor_hidden_feats': [128, 512, 1024],  # [32, 64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
		}
		max_epochs = 2000

		from redox_prediction.models.nn.unimp import UniMP
		estimator = UniMP

	else:
		raise ValueError('Unknown model: {}.'.format(kwargs.get('model')))

	return model_selection_for_gnn(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		estimator,
		param_grid,
		model_type,
		max_epochs=max_epochs,
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
		all_history.update(
			{k: AverageMeter(keep_all=False) for k in [
				'batch_time_train', 'data_time_train', 'epoch_time_fit'
			]}
		)
	if valid:
		all_history.update(
			{k: AverageMeter(keep_all=False) for k in [
				'batch_time_valid', 'data_time_valid', 'epoch_time_fit'
			]}
		)
	if pred:
		all_history.update(
			{k: AverageMeter(keep_all=False) for k in [
				'batch_time_pred', 'data_time_pred'
			]}
		)
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


def _print_time_info(best_history, history_train, history_valid, history_test):
	print('Training time:')
	print(
		'  Batch Train:\ttotal {:.3f}\tavg {:.9f}'.format(
			best_history['batch_time_train'].sum,
			best_history['batch_time_train'].avg
		)
	)
	print(
		'  Data Train:\ttotal {:.3f}\tavg {:.9f}'.format(
			best_history['data_time_train'].sum,
			best_history['data_time_train'].avg
		)
	)
	print(
		'  Epoch Fit:\ttotal {:.3f}\tavg {:.9f}'.format(
			best_history['epoch_time_fit'].sum,
			best_history['epoch_time_fit'].avg
		)
	)
	print('Prediction time:')
	print(
		'  Batch Train:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_train['batch_time_pred'].sum,
			history_train['batch_time_pred'].avg
		)
	)
	print(
		'  Data Train:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_train['data_time_pred'].sum,
			history_train['data_time_pred'].avg
		)
	)
	print(
		'  Batch Valid:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_valid['batch_time_pred'].sum,
			history_valid['batch_time_pred'].avg
		)
	)
	print(
		'  Data Valid:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_valid['data_time_pred'].sum,
			history_valid['data_time_pred'].avg
		)
	)
	print(
		'  Batch Test:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_test['batch_time_pred'].sum,
			history_test['batch_time_pred'].avg
		)
	)
	print(
		'  Data Test:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_test['data_time_pred'].sum,
			history_test['data_time_pred'].avg
		)
	)
