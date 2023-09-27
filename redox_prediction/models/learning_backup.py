#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:17:07 2022

@author: ljia
"""
import os
import sys
import pickle
import numpy as np
# import wandb
# wandb.login()
# from deepchem.models import WandbLogger
import deepchem as dc
sys.path.insert(0, '../../')
# from deepchem.models import ValidationCallback


def plot_perf_vs_epoch(all_scores, fig_name, epoch_interval=10):
	import matplotlib.pyplot as plt
	import seaborn as sns
	colors = sns.color_palette('husl')[0:]
# 		sns.axes_style('darkgrid')
	sns.set_theme()
	fig = plt.figure()
	ax = fig.add_subplot(111)    # The big subplot for common labels

	for idx, (key, val) in enumerate(all_scores.items()):
		epochs = list(range(epoch_interval, (len(val) + 1) * epoch_interval,
					  epoch_interval))
		ax.plot(epochs, val, '-', c=colors[idx], label=key)


	ax.set_xlabel('epochs')
	ax.set_ylabel('MAE')
	ax.set_title('')
	fig.subplots_adjust(bottom=0.3)
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
# 	plt.savefig(fig_name + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fig_name + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fig_name + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()


# def get_fit_params(**kwargs):
# 	fit_params = {}

# 	# loss
# 	if 'loss' in kwargs:
# 		if kwargs['loss'] == 'l1':
# 			fit_params['loss'] = dc.models.losses.L1Loss()
# 		elif kwargs['loss'] == 'l2':
# 			fit_params['loss'] = dc.models.losses.L2Loss()
# 		else:
# 			raise ValueError('The keyword "loss": "%s" can not be recognized, '
# 					'possible candidates include "l1", "l2".' % kwargs['loss'])

# 	# Get other available parameters.
# 	key_params = ['learning_rate']
# 	for key in key_params:
# 		if key in kwargs:
# 			fit_params[key] = kwargs[key]

# 	return fit_params


def get_model_params(model_name, **kwargs):
	model_params = {}

	# graph_conv_layers / graph_attention_layers
	if 'n_graph_layers' in kwargs and 'w_graph_channels' in kwargs:
		if model_name.lower() in ['gcnmodelext', 'graphconvmodelext']:
			model_params['graph_conv_layers'] = [kwargs['w_graph_channels']] * kwargs['n_graph_layers']
		elif model_name.lower() == 'gatmodelext':
			model_params['graph_attention_layers'] = [kwargs['w_graph_channels']] * kwargs['n_graph_layers']


	# loss
	if 'loss' in kwargs:
		if kwargs['loss'] == 'l1':
			model_params['loss'] = dc.models.losses.L1Loss()
		elif kwargs['loss'] == 'l2':
			model_params['loss'] = dc.models.losses.L2Loss()
		else:
			raise ValueError('The keyword "loss": "%s" can not be recognized, '
					'possible candidates include "l1", "l2".' % kwargs['loss'])

	# Get other available parameters.
	key_params = ['dropout', 'batch_size', 'learning_rate']
	if model_name.lower() == 'graphconvmodelext':
		key_params += ['dense_layer_size']
	elif model_name.lower() == 'gatmodelext':
		key_params += ['n_attention_heads', 'agg_modes', 'residual',
				  'predictor_hidden_feats', 'predictor_dropout', 'self_loop']
	elif model_name.lower() == 'gcnmodelext':
		key_params += ['residual', 'batchnorm',
				  'predictor_hidden_feats', 'predictor_dropout', 'self_loop']

	for key in key_params:
		if key in kwargs:
			model_params[key] = kwargs[key]

	return model_params


def get_model(model_name, **kwargs):

	# Get parameters.
	n_layers = kwargs.get('n_graph_layers', 2)
	backend = ('tf' if model_name == 'graphconvmodelext' else 'pytorch')
	activation_fns = get_activation_fns(kwargs.get('activation_fn', 'relu'),
									  n_layers=n_layers, backend=backend)
	graph_pools = get_graph_pools(kwargs.get('graph_pool', 'max'),
							   n_layers=n_layers)
	model_params = get_model_params(model_name, **kwargs)
	if 'number_atom_features' in kwargs:
		model_params['number_atom_features'] = kwargs['number_atom_features']

	# Create the model.
	if model_name.lower() == 'graphconvmodelext':
		from redox_prediction.models import GraphConvModelExt
		model = GraphConvModelExt(mode='regression', n_tasks=1,
							activation_fns=activation_fns,
							graph_pools=graph_pools,
							**model_params)

	elif model_name.lower() == 'gatmodelext':
		from redox_prediction.models import GATModelExt
		model = GATModelExt(mode='regression', n_tasks=1,
					  activation=activation_fns,
					  graph_pools=graph_pools,
					  **model_params)

	elif model_name.lower() == 'gcnmodelext':
		from redox_prediction.models import GCNModelExt
		model = GCNModelExt(mode='regression', n_tasks=1,
					  activation=activation_fns,
					  graph_pools=graph_pools,
					  **model_params)

# 	elif model_name.lower() == 'gcnmodel':
# 		model = dc.models.GCNModel(mode='regression', n_tasks=1,
# 							 activation=activation_fns[0],
# 							 **model_params)
# 							 wandb=logger)
# 	elif model_name.lower() == 'gatmodel':
# 		model = dc.models.GATModel(mode='regression', n_tasks=1,
# 							 activation=activation_fns[0],
# 							 **model_params)
# 							 model_dir='model_test/', log_frequency=10)
# 							 wandb=logger)
# 		model = dc.models.GATModel(mode='regression', n_tasks=len(chembl_tasks),
# 							 learning_rate=0.001)

# 	elif model_name.lower() == 'graphconvmodel':
# 		model = dc.models.GraphConvModel(mode='regression', n_tasks=1
	else:
		raise ValueError('Model "%s" can not be recognized.' % model_name)

	return model


def get_activation_fns(func_name, n_layers=2, backend='tf'):
	if backend == 'tf':
		import tensorflow as tf

		if func_name.lower() == 'relu':
			return [tf.keras.activations.relu for _ in range(n_layers)]
		elif func_name.lower() == 'elu':
			return [tf.keras.activations.elu for _ in range(n_layers)]
		elif func_name.lower() == 'leaky_relu':
			return [tf.nn.leaky_relu for _ in range(n_layers)]
		elif func_name.lower() == 'selu':
			return [tf.keras.activations.selu for _ in range(n_layers)]
		elif func_name.lower() == 'gelu':
			return [tf.keras.activations.gelu for _ in range(n_layers)]
		elif func_name.lower() == 'linear':
			return [tf.keras.activations.linear for _ in range(n_layers)]
		elif func_name.lower() == 'exponetial':
			return [tf.keras.activations.exponetial for _ in range(n_layers)]
		elif func_name.lower() == 'tanh':
			return [tf.keras.activations.tanh for _ in range(n_layers)]
		elif func_name.lower() == 'softmax':
			return [tf.keras.activations.softmax for _ in range(n_layers)]
		elif func_name.lower() == 'sigmoid':
			return [tf.keras.activations.sigmoid for _ in range(n_layers)]
	# 	elif func_name.lower() == 'normalize':
	# 		return [tf.nn.relu for _ in range(n_layers)]
		else:
			raise ValueError('The given activation function "%s" can not be '
					'recognized.' % func_name)

	elif backend == 'pytorch':
		import torch.nn.functional as F

		if func_name.lower() == 'relu':
			return [F.relu for _ in range(n_layers)]
		elif func_name.lower() == 'elu':
			return [F.elu for _ in range(n_layers)]
		elif func_name.lower() == 'leaky_relu':
			return [F.leaky_relu for _ in range(n_layers)]
		elif func_name.lower() == 'selu':
			return [F.selu for _ in range(n_layers)]
		elif func_name.lower() == 'gelu':
			return [F.gelu for _ in range(n_layers)]
		elif func_name.lower() == 'linear':
			return [F.linear for _ in range(n_layers)]
		elif func_name.lower() == 'exponetial':
			raise NotImplementedError('Sorry dear, the activation "exponetial" '
							 'under the backend "pytorch" has not been '
							 'implemented yet.')
		elif func_name.lower() == 'tanh':
			return [F.tanh for _ in range(n_layers)]
		elif func_name.lower() == 'softmax':
			return [F.softmax for _ in range(n_layers)]
		elif func_name.lower() == 'sigmoid':
			return [F.sigmoid for _ in range(n_layers)]
	# 	elif func_name.lower() == 'normalize':
	# 		return [tf.nn.relu for _ in range(n_layers)]
		else:
			raise ValueError('The given activation function "%s" can not be '
					'recognized.' % func_name)

	else:
		raise ValueError('Backend "%s" can not be recognized, possible candidates '
				   'include "tf" and "pytorch".' % backend)


def get_graph_pools(func_name, n_layers=2, backend='tf'):
	import tensorflow as tf

	if func_name.lower() == 'max': # Use the default setting.
		return None
	else:
		raise ValueError('The given graph pool method can not be recognized.')


def split_data(D, y, F, train_index, test_index):
	D_app = [D[i] for i in train_index]
	D_test = [D[i] for i in test_index]
	y_app = [y[i] for i in train_index]
	y_test = [y[i] for i in test_index]
	F_app = [F[i] for i in train_index]
	F_test = [F[i] for i in test_index]
	return D_app, D_test, y_app, y_test, F_app, F_test


def select_model(all_scores):
	best_score = np.inf
	for v in all_scores:
		if v['scores'][1] < best_score:
			best_score = v['scores'][1]
			train_score, valid_score, test_score, loss, best_epoch = v['scores']
			best_params = {**v['params'], **{'epoch': best_epoch}}

	return train_score, valid_score, test_score, loss, best_params


def check_param_grid(param_grid, cv_scheme='random'):
	from sklearn.model_selection import ParameterGrid

	if cv_scheme == 'grid':
		param_grid_list = list(ParameterGrid(param_grid))
		nb_max_params = len(param_grid_list)
	elif cv_scheme == 'random':
		grid_tmp = ParameterGrid(param_grid)
		nb_params = len(grid_tmp)
		i_rand = np.random.randint(0, nb_params, size=nb_params)
		param_grid_list = [grid_tmp[i] for i in i_rand]
		nb_max_params = 100 # @todo: change size as needed.
	else:
		raise ValueError('The cv_scheme "%s" can not be recognized, possible '
				   'candidates include "random" and "grid".' % cv_scheme)

	return param_grid_list, nb_max_params


def train_valid_test_model(train_dataset, valid_dataset, test_dataset,
					  model_name, metric=None,
					  refit=False,
					  hyperparams={},
					  save_scores=True, plot_scores=True,
					  verbose=1,
					  **kwargs):

	# Get parameters.
	transformers = kwargs.get('transformers', [])
	kw_metric = kwargs.get('kw_metric', 'mean-mae_score')
	max_epochs = kwargs.get('max_epochs', 100)
	epoch_interval = kwargs.get('epoch_interval', 10)
	number_atom_features = train_dataset.X[0].node_features.shape[1]

	# Get model.
	model = get_model(model_name, number_atom_features=number_atom_features,
				    **hyperparams, **kwargs)

	### Iterate over epochs.
	all_scores = {'train_scores':  [], 'valid_scores': [], 'test_scores': [],
			   'loss': []}
	nb_i_no_better, max_nb_i_no_better = 0, 30 # for early stop.
	best_valid_mean = np.inf
	for epoch in range(epoch_interval, max_epochs + epoch_interval, epoch_interval):

		# Fit model.
# 		fit_params = get_fit_params(**hyperparams)
		loss = model.fit(train_dataset, nb_epoch=epoch_interval) #, **fit_params)

		# Evaluate our fitted model on training, validation and test sets.
		train_score = model.evaluate(train_dataset, [metric], transformers)[kw_metric]
		valid_score = model.evaluate(valid_dataset, [metric], transformers)[kw_metric]
		test_score = model.evaluate(test_dataset, [metric], transformers)[kw_metric]

		all_scores['train_scores'].append(train_score)
		all_scores['valid_scores'].append(valid_score)
		all_scores['test_scores'].append(test_score)
		all_scores['loss'].append(loss)

		# Show output for each epoch.
		if verbose and epoch % 100 == 0:
			print('--- epoch %d, loss: %f, valid score: %f, train score: %f, '
			'test score: %f ---' % (epoch, loss, valid_score, train_score, test_score))
	# 	assert valid_scores['mean-rms_score'] < 10.00


		# Early stop.
		if nb_i_no_better >= max_nb_i_no_better:
			break

		nb_i_no_better += 1

		cur_valid_mean = np.mean(all_scores['valid_scores'][-7:])
		if cur_valid_mean <= best_valid_mean:
			best_valid_mean = cur_valid_mean
			nb_i_no_better = 0


	if save_scores:
		### Save all scores to file.
		output_dir = kwargs.get('output_dir', '')
		os.makedirs(output_dir, exist_ok=True)
		trial_index = kwargs.get('trial_index', 0)
		fn_params = '.'.join(k + '_' + str(v) for k, v in hyperparams.items())
		fn_scores = os.path.join(output_dir, 'all_scores.trial'
						   + str(trial_index) + '.' + fn_params + '.pkl')
		with open(fn_scores, 'wb') as f:
			pickle.dump(all_scores, f)


	if plot_scores:
		### Plot scores v.s. epochs.
		output_dir = kwargs.get('output_dir', '')
		os.makedirs(output_dir, exist_ok=True)
		trial_index = kwargs.get('trial_index', 0)
		fn_params = '.'.join(k + '_' + str(v) for k, v in hyperparams.items())
		fig_name = os.path.join(output_dir, 'scores_vs_epochs.trial'
						  + str(trial_index) + '.' + fn_params)
		plot_perf_vs_epoch(all_scores, fig_name,
					  epoch_interval=epoch_interval)


	### Get best scores and model.
	best_epoch = np.argmin(all_scores['valid_scores'])
	train_score = all_scores['train_scores'][best_epoch]
	valid_score = all_scores['valid_scores'][best_epoch]
	test_score = all_scores['test_scores'][best_epoch]
	loss = all_scores['loss'][best_epoch]
	if verbose:
		print('### The best results are: epoch %d, loss: %f, valid score: %f, '
		   'train score: %f, test score: %f. ###' % (best_epoch, loss,
											   valid_score, train_score, test_score))


	return train_score, valid_score, test_score, loss, best_epoch, all_scores


def cross_validation(train_dataset, valid_dataset, test_dataset,
					  model_name, param_grid={},
					  cv_scheme='random',
					  metric=None,
					  return_train_score=True, refit=False,
					  save_scores=True, plot_scores=True,
					  verbose=1,
					  **kwargs):

	# Get hyperparameter grid list.
	param_grid_list, nb_max_params = check_param_grid(param_grid, cv_scheme=cv_scheme)

	### hyperparameters selection
	all_scores = [] # scores of all hyperparameter settings
	cnt_nb_p = 0 # number of params tried
	# for each hyperparameter setting
	for params in param_grid_list:

		cnt_nb_p += 1
		if cnt_nb_p > nb_max_params:
			break

		if verbose:
			print('----------- current hyperparameters: ------------')
			print(params)

		try:
			cur_scores = train_valid_test_model(train_dataset,
									  valid_dataset, test_dataset,
									  model_name,
									  metric=metric,
									  refit=refit,
									  hyperparams=params,
									  save_scores=save_scores,
									  plot_scores=plot_scores,
									  verbose=verbose,
									  **kwargs)
		except RuntimeError as e:
			if verbose:
				print('UnboundLocalError: ' + repr(e) + '\n')
			cur_scores = None
			cnt_nb_p -= 1


		if cur_scores is not None:
			all_scores.append({'scores': [i for i in cur_scores[0:-1]],
						'params': params})


	best_results = select_model(all_scores)

	if verbose:
		print('best parameters:')
		print(best_results[4])

	return best_results


def evaluate_GNN(train_dataset, valid_dataset, test_dataset, mode='reg',
				 nb_epoch=100, output_dir=None, trial_index=1, **kwargs):
	### Get hyperparameters.
	model_name = kwargs.get('model', 'GATModelExt')
	feature_scaling = kwargs.get('feature_scaling', 'standard_y')
	metric_name = kwargs.get('metric', 'RMSE')
	activation_fn = kwargs.get('activation_fn', 'relu')
	graph_pool = kwargs.get('graph_pool', 'max')

	# Transform.
	if feature_scaling.lower() == 'none':
		transformers = []
	if feature_scaling.lower() == 'standard_y':
		transformers = [dc.trans.NormalizationTransformer(transform_y=True,
													 dataset=train_dataset)]
	elif feature_scaling.lower() == 'minmax_y':
		transformers = [dc.trans.MinMaxTransformer(transform_y=True,
											 dataset=train_dataset)]
	if feature_scaling != 'none':
		train_dataset = transformers[0].transform(train_dataset)
		valid_dataset = transformers[0].transform(valid_dataset)
		test_dataset = transformers[0].transform(test_dataset)

	# metric, averaged across tasks
	if metric_name.lower() == 'rmse':
		metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
		kw_metric = 'mean-rms_score'
	elif metric_name.lower() == 'mae':
		metric = dc.metrics.Metric(dc.metrics.mae_score, np.mean)
		kw_metric = 'mean-mae_score'
	elif metric_name.lower() == 'r2':
		metric = dc.metrics.Metric(dc.metrics.r2_score, np.mean)
		kw_metric = 'mean-r2_score'
	else:
		raise ValueError('Metric "%s" can not be recognized.' % metric_name)


	### Do CV.
	param_grid = set_param_grid(model_name)

	results = cross_validation(train_dataset, valid_dataset, test_dataset,
						  model_name, param_grid=param_grid,
						  cv_scheme = 'random',
						  metric=metric,
						  return_train_score=True, refit=False,
						  save_scores=True, plot_scores=True,
						  verbose=1,
						  max_epochs=nb_epoch,
						  epoch_interval=10,
						  transformers=transformers,
						  activation_fn=activation_fn,
						  graph_pool=graph_pool,
						  kw_metric=kw_metric,
						  output_dir=output_dir,
						  trial_index=trial_index)
	train_score, valid_score, test_score, loss, best_params = results


# 	### For comparison only.
# 	# Fit model.
# 	loss = model.fit(train_dataset, nb_epoch=1000)
# 	# Evaluate our fitted model on training, validationand test sets.
# 	train_score = model.evaluate(train_dataset, [metric], transformers)[kw_metric]
# 	valid_score = model.evaluate(valid_dataset, [metric], transformers)[kw_metric]
# 	test_score = model.evaluate(test_dataset, [metric], transformers)[kw_metric]
# 	# Show output for each epoch.
# 	print('--- epoch %d, loss: %f, valid score: %f, train score: %f, '
# 	'test score: %f ---' % (1000, loss, valid_score, train_score, test_score))


	### Visualize model.
# 	print(model)
# 	print(model.summary())

# 	# 2. -------------------------------------------
# 	from torchviz import make_dot
# 	output = model.predict(test_dataset, [transformer])
# 	make_dot(output)

# 	# 1. -------------------------------------------
# 	from tensorflow.keras.utils import plot_model
# 	plot_model(model)


	model = best_params # @todo: change it back.

	return train_score, valid_score, test_score, model


def set_param_grid(model_name):
	param_grid = { #@todo: to change back
# 		'n_graph_layers': [1, 2, 3, 4, 5],# [2, 3, 4], # No.2
# 		'w_graph_channels': [32, 64, 128], # No.1
# 		'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], # [0.0, 0.25, 0.5], # No.5
# 		'batch_size': [8, 16, 32, 64, 128], # [16], # [8, 16, 32, 64, 128], # No.7
# 		'learning_rate': [0.1, 0.01, 0.001, 0.0001], # [0.001]# No.4
# 		'loss': ['l1', 'l2'], # No.3
		'n_graph_layers': [1],
		'w_graph_channels': [128],
		'dropout': [0.1],
		'batch_size': [64],
		'learning_rate': [0.01],
		'loss': ['l1'],
# 		'dense_layer_size': [128],
		}

	if model_name.lower() == 'graphconvmodelext':
		pg_ext = {
			'dense_layer_size': [128, 256, 512], # [128], #, # No.6
			}
		param_grid.update(pg_ext)

	elif model_name.lower() == 'gatmodelext':
		pg_ext = {
 			'n_attention_heads': [4, 8, 16],
 			'agg_modes': [None, 'flatten', 'mean'],
 			'residual': [True, False],
 			'predictor_hidden_feats': [128, 256, 512],
 			'predictor_dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
 			'self_loop': [True, False],
# 			'n_attention_heads': [16],
# 			'agg_modes': ['mean'],
# 			'residual': [True],
# 			'predictor_hidden_feats': [256],
# 			'predictor_dropout': [0.0],
# 			'self_loop': [False],
			}
		param_grid.update(pg_ext)

	elif model_name.lower() == 'gcnmodelext':
		pg_ext = {
# 			'residual': [True, False],  #@todo: to change back
# 			'batchnorm': [False, True],
#  			'predictor_hidden_feats': [128, 256, 512],
#  			'predictor_dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
#  			'self_loop': [True, False],
 			'residual': [True],
			'batchnorm': [False],
 			'predictor_hidden_feats': [256],
 			'predictor_dropout': [0.5],
 			'self_loop': [True],

			}
		param_grid.update(pg_ext)

	return param_grid


def xp_GCN(smiles, y_all, families,
		   mode='reg', nb_epoch=100, output_file=None, **kwargs):
	'''
	Perform a GCN regressor on given dataset.
	'''

	### Load existing results if possible.
	if output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = []


	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, train_test_split

# 	stratified = False
	if mode == 'classif':
		stratified = True # @todo: to change it as needed.
	else:
		stratified = kwargs.get('stratified', True)

	cv = kwargs.get('cv')
	test_size = (0.1 if cv == '811' else 0.2)

	import collections
	if np.ceil(test_size * len(smiles)) < len(collections.Counter(families)):
		stratified = False # ValueError: The test_size should be greater or equal to the number of classes.
		# @todo: at least let each data in test set has different family?

	if stratified:
		# @todo: to guarantee that at least one sample is in train set (, valid set and test set).
		rs = StratifiedShuffleSplit(n_splits=5, test_size=test_size,
							   random_state=0)
		split_scheme = rs.split(smiles, families)
	else:
# 		rs = ShuffleSplit(n_splits=10, test_size=.1) #, random_state=0)
		rs = ShuffleSplit(n_splits=5, test_size=test_size, random_state=0) # @todo: to change it back.
		split_scheme = rs.split(smiles)


	i = 1
	for app_index, test_index in split_scheme:
		print()
		print("Split {0}/{1}".format(i, 5))
		i = i + 1

		# Check if the split already computed.
		if i - 1 <= len(results):
			print('Skip.')
			continue

		cur_results = {}
		# Get splitted data
		G_app, G_test, y_app, y_test, F_app, F_test = split_data(smiles,
											   y_all, families,
											   app_index, test_index)

		# Split evaluation set.
		valid_size = test_size / (1 - test_size)
		stratify = (F_app if stratified else None)
		G_train, G_valid, y_train, y_valid = train_test_split(G_app, y_app,
														test_size=valid_size,
														random_state=0,
														shuffle=True,
														stratify=stratify)

		cur_results['y_app'] = y_app
		cur_results['y_test'] = y_test
		cur_results['app_index'] = app_index
		cur_results['test_index'] = test_index

		kwargs['nb_trial'] = i - 1
# 		for setup in distances.keys():
# 		print("{0} Mode".format(setup))
		setup_results = {}

		# featurize.
		if kwargs['model'].lower() in ['graphconvmodel', 'graphconvmodelext']:
			featurizer = dc.feat.ConvMolFeaturizer()
		elif kwargs['model'].lower() in ['gcnmodel', 'gatmodel', 'gatmodelext', 'gcnmodelext']:
			if kwargs['descriptor'].lower() == 'smiles':
				from dataset.feat import DCMolGraphFeaturizer
				featurizer = DCMolGraphFeaturizer(use_edges=True,
											   use_chirality=True,
											   use_partial_charge=False,
   											   use_distance_stats=False,
											   use_xyz=False)
			elif kwargs['descriptor'].lower() == 'smiles+xyz_obabel':
				from dataset.feat import DCMolGraphFeaturizer
				featurizer = DCMolGraphFeaturizer(use_edges=True,
											   use_chirality=True,
											   use_partial_charge=False,
											   use_distance_stats=False,
											   use_xyz=True,
											   feature_scaling='auto')
			elif kwargs['descriptor'].lower() == 'smiles+dis_stats_obabel':
				from dataset.feat import DCMolGraphFeaturizer
				featurizer = DCMolGraphFeaturizer(use_edges=True,
											   use_chirality=True,
											   use_partial_charge=False,
											   use_distance_stats=True,
											   use_xyz=False,
											   feature_scaling='auto')
		else:
			raise ValueError('Model "%s" can not be recognized.' % kwargs['model'])

# 		X_app = featurizer.featurize(G_app)
		X_train = featurizer.featurize(G_train) # @todo: to change back
		feature_scaler = featurizer.feature_scaler
		X_valid = featurizer.featurize(G_valid, feature_scaler=feature_scaler)
		X_test = featurizer.featurize(G_test, feature_scaler=feature_scaler)
# 		train_dataset = dc.data.NumpyDataset(X=X_app, y=y_app)
		train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train)
		valid_dataset = dc.data.NumpyDataset(X=X_valid, y=y_valid)
		test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)

		# Stratified inner cv.
# 		if stratified:
# # 				rs_in = StratifiedShuffleSplit(n_splits=5, test_size=.2, random_state=0)
# 			rs_in = StratifiedShuffleSplit(n_splits=5, test_size=.2, random_state=None) # StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# 			stra_y_in = [stratified_y[i] for i in train_index]
# 			cv_in = list(rs_in.split(G_app, stra_y_in)) # List cv, otherwise the clf can not be pickled.
# 		else:
# 			cv_in = 5
# 		cv_in = 5
# 		perf_app, perf_test, clf = evaluate_GNN(
		# directory to save all results.
		output_dir = '.'.join(output_file.split('.')[:-1])
		perf_train, perf_valid, perf_test, clf = evaluate_GNN(
			train_dataset, valid_dataset, test_dataset,
			mode=mode, nb_epoch=nb_epoch,
			output_dir=output_dir, trial_index=i-1, **kwargs)

# 		setup_results['perf_app'] = perf_app
		setup_results['perf_train'] = perf_train
		setup_results['perf_valid'] = perf_valid
		setup_results['perf_test'] = perf_test
		setup_results['clf'] = clf

		print("Learning performance with {0} costs".format(perf_train))
		print("Validation performance with {0} costs".format(perf_valid))
		print("Test performance with {0} costs".format(perf_test))
		cur_results = setup_results
		results.append(cur_results)


		### Save the split.

		# Check if the (possible) existing file was updated by another thread during the computation of this split.
		if output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
			with open(output_file, 'rb') as file:
				results_new = pickle.load(file)['results']
				if i - 1 <= len(results_new):
					print('Skip.')
					continue

		# If not, save the results.
		if output_file is not None:
			pickle.dump({'results': results}, open(output_file, 'wb'))

	return results