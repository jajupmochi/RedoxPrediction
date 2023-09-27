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

from typing import Tuple

import numpy as np

# matplotlib.use('Agg')
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

import torch
from torch_geometric.loader import DataLoader

from redox_prediction.dataset.nn.pairwise import PairwiseDataset
from redox_prediction.models.embed.utils import get_entire_metric_matrix
from redox_prediction.models.model_selection.utils import get_submatrix_by_index
from redox_prediction.utils.logging import AverageMeter


def fit_model(
		train_loader: DataLoader,
		estimator: torch.nn.Module,
		params: dict,
		model_type: str,
		max_epochs: int,
		device: torch.device,
		valid_loader: DataLoader = None,
		valid_loader_task: DataLoader = None,
		epochs_per_eval: int = None,
		plot_loss: bool = False,
		print_interval: int = 1,  # debug: to change as needed.
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
			kwargs.get('split_idx'),
			kwargs.get('params_idx'),
			kwargs.get('fold_idx')
		)
	)
	os.makedirs(os.path.dirname(load_state_dict_from), exist_ok=True)
	# Train task-specific model:
	from redox_prediction.models.evaluation.mtgnn import fit_model_mtgnn
	model, history, valid_losses, valid_metrics = fit_model_mtgnn(
		model,
		train_loader,
		valid_loader,
		valid_loader_task=valid_loader_task,
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
		dataset_app,
		y_app,
		metric_matrix,
		params,
		kf,
		estimator,
		model_type,
		device,
		max_epochs=800,  # debug: 500
		if_tune_n_epochs: bool = False,
		epochs_per_eval: int = 10,
		verbose=True,
		**kwargs
):
	all_history = _init_all_history()
	perf_valid_list, valid_loss_list = [], []

	# for each inner CV fold:
	for idx, (train_index, valid_index) in enumerate(
			kf.split(dataset_app, y_app)
	):

		# # For debugging only:  # debug: comment this.
		# if idx > 1:
		# 	break

		print('\nTrial: {}/{}'.format(idx + 1, kf.get_n_splits()))

		# split the dataset into train and validation sets:
		dataset_train = dataset_app[train_index]
		metric_matrix_train = metric_matrix[train_index, :][:, train_index]
		train_loader = DataLoader(
			PairwiseDataset(dataset_train, metric_matrix=metric_matrix_train),
			batch_size=params['batch_size'], shuffle=False
		)
		dataset_valid = dataset_app[valid_index]
		metric_matrix_valid = metric_matrix[valid_index, :][:, valid_index]
		valid_loader = DataLoader(
			PairwiseDataset(dataset_valid, metric_matrix=metric_matrix_valid),
			batch_size=params['batch_size'], shuffle=False
		)
		valid_loader_task = DataLoader(
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
			valid_loader_task=valid_loader_task,
			epochs_per_eval=epochs_per_eval,
			verbose=verbose,
			plot_loss=True,
			**{**kwargs, 'fold_idx': idx}
		)
		if verbose:
			print(
				'Total time for training in this trial: {:.3f} s.'.format(
					time.time() - start_time
				)
			)
		perf_valid_list.append(valid_metrics)
		valid_loss_list.append(valid_losses)

		_update_history_1fold(all_history, history, None)

	if if_tune_n_epochs:
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
	else:
		perf_valid = np.mean([v[-1] for v in perf_valid_list])
		best_n_epochs = max_epochs

	# Show the best performance and the corresponding number of epochs:
	if verbose:
		print()
		print('Best valid performance: {:.3f}'.format(perf_valid))
		if if_tune_n_epochs:
			print('Best n_epochs: {}'.format(best_n_epochs))
		else:
			print('Best n_epochs is set to max_epochs: {}'.format(max_epochs))

	return perf_valid, all_history, best_n_epochs


def model_selection_for_mtgnn(
		G_app, y_app, G_test, y_test,
		estimator,
		estimator_metric,
		param_grid,
		param_grid_metric,
		model_type,
		max_epochs=800,  # debug: 500
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
	param_list_metric = list(ParameterGrid(param_grid_metric))

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

		# Grid-search for hparams for metric:
		for idx_metric, params_metric in enumerate(param_list_metric):
			if verbose:
				print()
				print(
					'---- Metric parameter settings {}/{} -----:'.format(
						idx_metric + 1, len(param_list_metric)
					)
				)
				print(params_metric)

			# Get the metric matrix for the given hparams:
			# We do this a prior for two purposes: 1) Reduce time complexity
			# by computing only one matrix over multiple GNNs; 2) the `gklearn`
			# implementation of the WLsubtree kernel is currently not correctly
			# when transforming for unseen target graphs.
			try:
				model_metric, run_time_metric, metric_matrix = get_entire_metric_matrix(
					G_app + G_test,
					estimator_metric,
					metric_params=params_metric,
					load_metric_from_file=(read_resu_from_file >= 1),
					params_idx=idx_metric,
					reorder_graphs=True,
					verbose=(1 if verbose else 0),
					**kwargs
					# output_dir = kwargs.get('output_dir'),
				)
				metric_matrix_app = get_submatrix_by_index(
					metric_matrix, G_app, idx_key='id'
				)
			except FloatingPointError:
				continue

			perf_valid, all_history, best_n_epochs = evaluate_parameters(
				dataset_app, y_app, metric_matrix_app,
				params,
				kf,
				estimator,
				model_type,
				device,
				max_epochs=max_epochs,
				verbose=verbose,
				params_idx=str(idx),
				read_resu_from_file=read_resu_from_file,
				**{**kwargs, 'metric_pairwise_run_time': run_time_metric.avg}
			)

			# Update the best parameters:
			if check_if_valid_better(perf_valid, perf_valid_best, model_type):
				perf_valid_best = perf_valid
				params_best = copy.deepcopy(params)
				params_best_metric = copy.deepcopy(params_metric)
				best_best_n_epochs = best_n_epochs
				best_model_metric = model_metric
				best_history = {
					**copy.deepcopy(all_history), 'fit_time_metric': run_time_metric
				}

	# params_best = copy.deepcopy(params)
	# best_best_n_epochs = 970
	# break

	# Refit the best model on the whole dataset:
	print('\n---- Start refitting the best model on the whole valid dataset...')
	metric = ('rmse' if model_type == 'reg' else 'accuracy')
	app_loader = DataLoader(
		dataset_app, batch_size=params_best['batch_size'],
		shuffle=False
	)
	metric_matrix_app = get_submatrix_by_index(
		best_model_metric.gram_matrix, G_app, idx_key='id'  # gram_matrix is not correct for geds
	)
	model, history, _, _ = fit_model(
		app_loader,
		estimator,
		params_best,
		model_type,
		best_best_n_epochs,
		device,
		valid_loader=None,
		metric_matrix=metric_matrix_app,
		verbose=verbose,
		plot_loss=True,
		**{
			**kwargs, 'params_idx': 'refit',
			'read_resu_from_file': read_resu_from_file,
			'metric_pairwise_run_time': best_history['fit_time_metric'].avg
		}
	)
	perf_app, y_pred_app, y_true_app, pred_history_app = predict_gnn(
		app_loader,
		model,
		metric,
		device,
		model_type=model_type,
		y_scaler=kwargs['y_scaler'],
		predictor_clf_activation=params_best['predictor_clf_activation'],
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
		predictor_clf_activation=params_best['predictor_clf_activation'],
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
		history_app, history_test, {
		**params_best, 'n_epochs': best_best_n_epochs
	}


def evaluate_mtgnn(
		G_app, y_app, G_test, y_test,
		model_type='reg',
		descriptor='atom_bond_types',
		**kwargs
):
	# @todo ['softmax', 'log_softmax'],
	# clf_activation = (
	# 	'sigmoid' if kwargs.get('n_classes') == 2 else 'log_softmax')

	if kwargs.get('deep_model') == 'gcn':
		# Get parameter grid:
		param_grid = {
			'lr': [10 ** -4, 10 ** -5],
			'hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'agg_activation': ['relu'],
			'readout': ['mean'],
			'predictor_hidden_feats': [64, 128],  # [32, 64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
		}
		max_epochs = 1000

		from redox_prediction.models.nn.gcn import GCN
		estimator = GCN

	elif kwargs.get('deep_model') == 'dgcnn':
		# Get parameter grid:
		from redox_prediction.models.nn.dgcnn import get_sort_pooling_k
		# When ks consists of multiple 10s, keep only one of them:
		ks = sorted((set([
			get_sort_pooling_k(G_app + G_test, perc, 10) for perc in [
				0.6, 0.9]
		])))
		param_grid = {
			'lr': [10 ** -4, 10 ** -5],
			'hidden_feats': [32, 64],
			'message_steps': [2, 3, 4],
			'k': ks,  # [10, 20],
			'agg_activation': ['tanh'],
			'readout': ['sort_pooling'],
			'predictor_hidden_feats': [128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_dropout': [0.5],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
			'dim_target': [len(set(np.concatenate((y_app, y_test))))],
		}
		max_epochs = 1000

		from redox_prediction.models.nn.dgcnn import DGCNN
		estimator = DGCNN

	elif kwargs.get('deep_model') == 'gat':
		# Get parameter grid:
		param_grid = {
			'lr': [10 ** -4, 10 ** -5],
			'hidden_feats': [32, 64],  # [128],  # [64, 128],
			'n_heads': [4, 8],  # [4, 8],
			'concat_heads': [True],  # [True, False],
			'message_steps': [2, 3, 4],  # [5],  # [2, 3, 4],
			'attention_drop': [0, 0.5],  # [0., 0.5],
			'agg_activation': ['relu'],
			'readout': ['mean'],
			'predictor_hidden_feats': [64, 128],  # [128],  # [64, 128],
			'predictor_n_hidden_layers': [1],
			'predictor_activation': ['relu'],
			'predictor_clf_activation': ['log_softmax'],
			'batch_size': [32, 64],
		}
		max_epochs = 1000

		from redox_prediction.models.nn.gat import GAT
		estimator = GAT

	estimator_metric, param_grid_metric = get_estimator_metric(
		kwargs.get('embedding')
	)

	return model_selection_for_mtgnn(
		G_app, y_app, G_test, y_test,
		estimator,
		estimator_metric,
		param_grid,
		param_grid_metric,
		model_type,
		max_epochs=max_epochs,
		**kwargs
	)


def get_estimator_metric(
		embedding: str
) -> Tuple[object, dict]:
	"""
	Get the estimator and the parameter grid for the metric evaluation model.

	Parameters
	----------
	embedding: str
		Embedding method.

	Returns
	-------
	estimator_metric: object
		Estimator for the metric evaluation model. A `gklearn.kernel.GraphKernel`
		or `gklearn.ged.GEDModel` object.

	param_grid_metric: dict
		Parameter grid for the metric evaluation model.
	"""
	# for the graph kernel:
	if embedding == 'gk:treelet':
		# Get parameter grid:
		import functools
		from gklearn.utils.kernels import gaussiankernel, polynomialkernel
		gkernels = [
			functools.partial(gaussiankernel, gamma=1 / ga)
			#            for ga in np.linspace(1, 10, 10)]
			for ga in np.logspace(1, 10, num=10, base=10)
			# @TODO: change it as needed
		]
		pkernels = [
			functools.partial(polynomialkernel, d=d, c=c) for d in range(1, 5)
			for c in np.logspace(0, 10, num=11, base=10)
		]
		param_grid = {'sub_kernel': gkernels + pkernels}

		from gklearn.kernels import Treelet
		estimator = Treelet

	elif embedding == 'gk:wlsubtree':
		# Get parameter grid:
		param_grid = {'height': [0, 1, 2, 3, 4, 5, 6]}

		from gklearn.kernels import WLSubtree
		estimator = WLSubtree

	elif embedding == 'gk:path':
		# Get parameter grid:
		param_grid = {
			'depth': [1, 2, 3, 4, 5, 6],
			'k_func': ['MinMax', 'tanimoto'],
			'compute_method': ['trie']
		}

		from gklearn.kernels import PathUpToH
		estimator = PathUpToH

	elif embedding == 'gk:sp':
		# Get parameter grid:
		import functools
		from gklearn.utils.kernels import deltakernel, gaussiankernel, \
			kernelproduct
		mix_kernel = functools.partial(
			kernelproduct, deltakernel, gaussiankernel
		)
		param_grid = {
			'node_kernels': [
				{
					'symb': deltakernel, 'nsymb': gaussiankernel,
					'mix': mix_kernel
				}]
		}

		from gklearn.kernels import ShortestPath
		estimator = ShortestPath

	elif embedding == 'gk:structural_sp':
		import functools
		from gklearn.utils.kernels import deltakernel, gaussiankernel, \
			kernelproduct
		mix_kernel = functools.partial(
			kernelproduct, deltakernel, gaussiankernel
		)
		sub_kernels = [
			{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mix_kernel}
		]
		param_grid = {
			'node_kernels': sub_kernels, 'edge_kernels': sub_kernels,
			'compute_method': ['naive']
		}

		from gklearn.kernels import StructuralSP
		estimator = StructuralSP

	else:
		raise ValueError(
			'Unknown embedding method: {}.'.format(embedding)
		)

	return estimator, param_grid


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


def _print_time_info(history_app, history_test):
	print('Training time:')
	print(
		'  Batch App:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_app['batch_time_train'].sum,
			history_app['batch_time_train'].avg
		)
	)
	print(
		'  Data App:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_app['data_time_train'].sum,
			history_app['data_time_train'].avg
		)
	)
	print(
		'  Epoch Fit:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_app['epoch_time_fit'].sum,
			history_app['epoch_time_fit'].avg
		)
	)
	print('Prediction time:')
	print(
		'  Batch App:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_app['batch_time_pred'].sum,
			history_app['batch_time_pred'].avg
		)
	)
	print(
		'  Data App:\ttotal {:.3f}\tavg {:.9f}'.format(
			history_app['data_time_pred'].sum,
			history_app['data_time_pred'].avg
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
