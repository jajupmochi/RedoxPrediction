#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:45:43 2022

@author: ljia
"""
import os
import sys
sys.path.insert(0, '../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np

from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from gklearn.utils import get_iters

from gat_model import GATModel, GATDataset
from dataset.load_dataset import get_data
from models.utils import split_data


# tf.config.run_functions_eagerly(True) # @todo: this is for debug only.
# np.random.seed(42) # @todo: these are for debug only.
# tf.random.set_seed(42)


path_kw = '/gat/std_y/mae/'
# path_kw = '/mpnn_tranformer/std_y/mse/'
# path_kw = '/mpnn_tranformer/std_y/mape/'


#%% Model

# def are(y_true, y_pred):
#  	"""Average relative error.
#  	"""
# # 	print('I am are()!')
# #  	loss = y_true - y_pred
# #  	loss = tf.math.divide(loss, y_true)
# #  	loss = tf.math.abs(loss)
# #  	loss = 100 * tf.keras.backend.mean(loss)
# #  	return loss

#  	loss = y_true - y_pred
#  	loss = tf.math.divide(loss, y_true)
#  	loss = 100 * tf.keras.backend.mean(loss)
#  	return loss


### Losses.


def R_squared(y_true, y_pred):
    """R_squared computes the coefficient of determination.
    It is a measure of how well the observed outcomes are replicated by the model.

	Reference
	---------
	https://www.kaggle.com/code/rohumca/linear-regression-in-tensorflow/notebook
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def test_losses():
	x = tf.constant(
 	[[ 1 ],
	 [ 2 ],
	 [ 3 ]])
	y = tf.constant(
 	[[ 2 ],
	 [ 4 ],
	 [ 7 ]])
	# loss = are(x, y)
	loss = R_squared(x, y)
	return loss


### Callbacks.

class EvaluateTestSet(keras.callbacks.Callback):
	"""Evalute metrics on test set.
	"""
	def __init__(self, test_dataset,
			  train_dataset=None, valid_dataset=None,
			  batch_size=8):
		super(EvaluateTestSet, self).__init__()
		self.test_dataset = test_dataset
		self.train_dataset = train_dataset
		self.valid_dataset = valid_dataset
		self.batch_size = batch_size


	def on_epoch_end(self, epoch, logs=None):
# 		print('-----------------')
# 		print(logs)
		results = self.model.evaluate(self.test_dataset,
								batch_size=self.batch_size, verbose=0)
		metrics_names = ['test_all_' + nm for nm in self.model.metrics_names]
		logs.update(dict(zip(metrics_names, results)))

		# Add evaluated train and valid scores (to see their diff with the
		# default outputs).
		if self.train_dataset is not None:
			res1 = self.model.evaluate(self.train_dataset,
							  batch_size=self.batch_size, verbose=0)
			names1 = ['train_all_' + nm for nm in self.model.metrics_names]
			logs.update(dict(zip(names1, res1)))
		if self.valid_dataset is not None:
			res2 = self.model.evaluate(self.valid_dataset,
							  batch_size=self.batch_size, verbose=0)
			names2 = ['valid_all_' + nm for nm in self.model.metrics_names]
			logs.update(dict(zip(names2, res2)))
# 		print(dict(zip(metrics_names, results)))
# 		print('-----------------')


class EvaluateTransformed(keras.callbacks.Callback):
	"""Evalute metrics where the targets are transformed back using a
	transformer such as `sklearn.preprocessing.StandardScaler`.
	"""
	def __init__(self, transformer, test_dataset=None):
		super(EvaluateTransformed, self).__init__()
		self.transformer = transformer
		self.test_dataset = test_dataset


	def on_epoch_end(self, epoch, logs=None):
		pass


class NBatchLogger(keras.callbacks.Callback):
	"""Logger after each batch.

	Notes
	-----
	I use this class to check if "val_loss" in batch # n is the same of
	"loss" in batch # n+1, given the same train and valid data.

	Reference
	---------
	https://github.com/keras-team/keras/issues/2850
	"""
	def __init__(self, display=1, average=False, batch_size=8,
			  train_dataset=None, valid_dataset=None):
		"""
        display: Number of batches to wait before outputting loss.
		"""
		super(NBatchLogger, self).__init__()
		self.seen = 0
		self.display = display
		self.batch_size = batch_size
		self.train_dataset = train_dataset
		self.valid_dataset = valid_dataset


	def on_train_batch_end(self, batch, logs={}):
# 		print(logs)
# 		self.seen += logs.get('size', 0)
		if self.seen < self.params['steps']:
			self.seen += 1
		else:
			self.seen = 1

		# Add evaluated train and valid scores (to see their diff with the
		# default outputs).
		if self.train_dataset is not None:
			res1 = self.model.evaluate(self.train_dataset,
							  batch_size=self.batch_size, verbose=0)
			names1 = ['eval_train_' + nm for nm in self.model.metrics_names]
			logs.update(dict(zip(names1, res1)))
		if self.valid_dataset is not None:
			res2 = self.model.evaluate(self.valid_dataset,
							  batch_size=self.batch_size, verbose=0)
			names2 = ['eval_valid_' + nm for nm in self.model.metrics_names]
			logs.update(dict(zip(names2, res2)))

		if self.seen % self.display == 0:
 			print('\n{0}/{1} - Batch: {2}'.format(self.seen, self.params['steps'], str(logs)))


#%% Simple test run

def plot_epoch_curves(history, figure_name):
	from figures.utils import plot_perfs_vs_epochs
	os.makedirs(os.path.dirname(figure_name), exist_ok=True)

	# 1.
# 	all_scores = {'loss': history['loss'],
# 			   'train': history['mae'],
# 			   'valid': history['val_mae'],
# 			   'test': history['test_mae']}
	# 2.
# 	all_scores = {}
# 	for key, val in history.items():
# 		if 'loss' in key or 'mae' in key:
# 			all_scores[key] = val

# 	plot_perfs_vs_epochs(all_scores, figure_name,
# 						 y_label='MAE',
# 						 y_label_loss='MAPE',
# 						 epoch_interval=1)
	# 3.
	y_labels = ['loss (MAE) (K)', 'MAE (K)', '$R^2$']
	for i_m, metric in enumerate(['loss', 'mae', 'R_squared']):
		all_scores = {} # {'loss': history['loss']}
		for key, val in history.items():
			if metric in key:
				all_scores[key] = val
		plot_perfs_vs_epochs(all_scores,
					   figure_name + '.' + metric,
					   y_label=y_labels[i_m],
					   y_label_loss='MAPE',
					   epoch_interval=1)


### Featurizer.

from dataset.feat import MolGNNFeaturizer
af_allowable_sets = {
	'atom_type': ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
# 	'atom_type': ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"],
	'formal_charge': None, # [-2, -1, 0, 1, 2],
	'hybridization': ['SP', 'SP2', 'SP3'],
# 	'hybridization': ['S', 'SP', 'SP2', 'SP3'],
	'acceptor_donor': ['Donor', 'Acceptor'],
	'aromatic': [True, False],
	'degree': [0, 1, 2, 3, 4, 5],
 	# 'n_valence': [0, 1, 2, 3, 4, 5, 6],
	'total_num_Hs': [0, 1, 2, 3, 4],
	'chirality': ['R', 'S'],
	}
# bf_allowable_sets = {
# 	'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
# 	'same_ring': [True, False],
# 	'conjugated': [True, False],
# 	'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'],
# 	}
featurizer = MolGNNFeaturizer(use_edges=False,
						   use_partial_charge=False,
						   af_allowable_sets=af_allowable_sets,
# 						   bf_allowable_sets=bf_allowable_sets
						   )


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

	# Featurize inputs.
	X_train = featurizer.featurize(X_train)
	X_valid = featurizer.featurize(X_valid)
	X_test = featurizer.featurize(X_test)

	# Train and predict.
	# Hperams: tune them one by one rather than grid search.
# 	from sklearn.model_selection import ParameterGrid
	params_grid = { # @todo
# 		'in_feats': [32, 64, 128, 256], # influential?, [32]
		'hidden_feats': [8, 32, 64, 128, 256], # a little influential, [1]
		'message_steps': [1, 2, 3, 4],
		'num_attention_heads': [4, 8, 16], # a liitle influential, [16]
		'predictor_hidden_feats': [128, 256, 512, 1024], # a little influence, [512]
		'batch_size': [4, 8, 16], # a liitle influential, [8]
		'learning_rate': [5e-4, 0.001, 1e-4], # [5e-4, 0.001, 1e-4] influential?, [5e-4]
		'feat_drop': [0, 0.3, 0.6],
		'attn_drop': [0, 0.3, 0.6],
		'negative_slope': [0.2],
		'redurce_lr_factor': [0.5, 1, 0.2, 0.1], # a liile influential, [0.5]
		'residual': [False, True],
		'agg_activation': [None],
		'attn_agg_mode': ['concat'], # ['concat', 'mean'],
		'bias': [True],
		'readout': ['mean'],
		'predictor_activation': ['relu'],
		}

	MAE_min = np.inf
	MAE_all = []
	cur_params = {k: v[0] for k, v in params_grid.items()}
	for idx_params, param_name in get_iters(enumerate(cur_params), desc='Hyperparam tuning', length=len(cur_params)): # for each hyper-parameter:
		start_idx = (0 if idx_params == 0 else 1) # Avoid redundant computations.
		for param_val in params_grid[param_name][start_idx:]:
			cur_params[param_name] = param_val
			params = cur_params.copy() # (Depp) Copy so that the stored best values will not be overrided.

			batch_size = params['batch_size']

			# Initialize model.
			model = GATModel(
				# The following are used by the GATConv layer.
				in_feats=X_train[0][0][0].shape[0],
				hidden_feats=params['hidden_feats'],
				message_steps=params['message_steps'],
				num_attention_heads=params['num_attention_heads'],
				feat_drop=params['feat_drop'],
				attn_drop=params['attn_drop'],
				negative_slope=params['negative_slope'], # for LeakyReLU.
				residual=params['residual'],
				bias=params['bias'],
				# The following are used for aggragation of the multi-head outputs.
				agg_activation=params['agg_activation'],
				attn_agg_mode=params['attn_agg_mode'],
				# The following are used for readout.
				readout=params['readout'],
				batch_size=batch_size, # for transformer readout.
				# The following are used for the final prediction.
				predictor_hidden_feats=params['predictor_hidden_feats'],
				predictor_activation=params['predictor_activation'],
				mode='regression',
				)

			nb_epoch = 5000 # @todo: change it back

			# Choose loss function.
			if '/mae/' in path_kw:
				loss = tf.keras.metrics.mean_absolute_error # tf.keras.losses.MeanAbsoluteError() # tf.keras.metrics.MeanAbsoluteError()
		# 	r2_score = tfa.metrics.RSquare
			elif '/mse/' in path_kw:
				loss = tf.keras.metrics.mean_squared_error
			elif '/mape/' in path_kw:
				loss = tf.keras.metrics.mean_absolute_percentage_error

			model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
					loss=loss,
					# metrics=['mae'])
					metrics=['mae', R_squared],
					run_eagerly=True) # @todo: change as needed.
	# 				metrics=[keras.losses.MeanAbsoluteError(name='mae'), R_squared])
		# 			   metrics=['mae', loss, R_squared])
		# 	model.run_eagerly = True # @todo: this is for debug only.

# 			model.build(((1), (X_train[0][0][0].shape[0])))
# 			keras.utils.plot_model(model,
# 							 path_kw.strip('/').split('/')[0] + '_model.png',
# 							 show_dtype=True, show_shapes=True)

			# callbacks
			EarlyStopping =	tf.keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=50, # 100, # @todo
				verbose=1,
				restore_best_weights=True)
			if params['redurce_lr_factor'] < 1.0:
				ReduceLROnPlateau =	tf.keras.callbacks.ReduceLROnPlateau(
					monitor='val_loss',
					factor=params['redurce_lr_factor'],
					patience=20,
					verbose=1,
					mode='auto',
					min_delta=0.01,
					cooldown=0,
					min_lr=0)
			TensorBoard = tf.keras.callbacks.TensorBoard(
				log_dir='../outputs/' + path_kw + '/tb_logs.t' + str(trial_index))

			# Train model.
			train_dataset = GATDataset(X_train, y_train, batch_size=batch_size, shuffle=False)
			valid_dataset = GATDataset(X_valid, y_valid, batch_size=batch_size, shuffle=False)
			test_dataset = GATDataset(X_test, y_test, batch_size=batch_size, shuffle=False)
			history = model.fit(train_dataset,
				validation_data=valid_dataset,
 	 			callbacks=[
 	 				 EvaluateTestSet(test_dataset, batch_size=batch_size),
	 									# train_dataset=train_dataset,
	 									# valid_dataset=valid_dataset),
 	 				 EarlyStopping,
 	 				 # TensorBoard,
	 	# 				NBatchLogger(train_dataset=train_dataset,
	 	#  				 valid_dataset=valid_dataset,batch_size=batch_size),
					 ] +
 				 ([ReduceLROnPlateau] if params['redurce_lr_factor'] < 1.0 else []),
	 	# 				model_checkpoint,
	 	# 				CSVLogger(log_path[i], append=True)],
	 			batch_size=batch_size,
				epochs=nb_epoch,
	 			shuffle=True,
				verbose=0)

			# Record result if better.
			y_pred_valid = tf.squeeze(model.predict(valid_dataset, batch_size=batch_size), axis=1).numpy()
			if scale_y:
				y_tmp = np.ravel(y_scaler.inverse_transform(y_valid.reshape(-1, 1)))
				y_pred_tmp = np.ravel(y_scaler.inverse_transform(y_pred_valid.reshape(-1, 1)))
				MAE_cur = mean_absolute_error(y_tmp, y_pred_tmp)
			else:
				MAE_cur = mean_absolute_error(y_valid, y_pred_valid)
			MAE_all.append([params, MAE_cur])
			print('\n%s: %.4f.' % (str(params), MAE_cur))
			if MAE_cur < MAE_min:
				MAE_min = MAE_cur
				best_res = [MAE_min, params, model, history]

		cur_params[param_name] = best_res[1][param_name] # Use the best hparam value.

	print('\nbest hyperparams: %s' % str(best_res[1]))
	model = best_res[2]
	history = best_res[3]
	bparams = best_res[1]
	batch_size = bparams['batch_size']

	# Predict.
# 	for key, val in history.history.items():
# 		print('%s, %f.' % (key, val[-1]))
	y_pred_train = tf.squeeze(model.predict(train_dataset, batch_size=batch_size), axis=1).numpy()
	y_pred_valid = tf.squeeze(model.predict(valid_dataset, batch_size=batch_size), axis=1).numpy()
	y_pred_test = tf.squeeze(model.predict(test_dataset, batch_size=batch_size), axis=1).numpy()
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


	### Save predictions.
	fn = '../outputs/' + path_kw + '/y_values.t' + str(trial_index) + '.pkl'
	pickle.dump({'y_train': y_train, 'y_pred_train': y_pred_train,
			  'y_valid': y_valid, 'y_pred_valid': y_pred_valid,
			  'y_test': y_test, 'y_pred_test': y_pred_test,},
			 open(fn, 'wb'))
	fn = '../outputs/' + path_kw + 'mae_all.t' + str(trial_index) + '.pkl'
	pickle.dump(MAE_all, open(fn, 'wb'))


	### plot performance w.r.t. epochs.
	figure_name = '../figures/' + path_kw + '/perfs_vs_epochs.t' + str(trial_index)
	plot_epoch_curves(history.history, figure_name)


	return train_accuracy, valid_accuracy, test_accuracy, bparams, model


def cross_validate(X, targets, families=None,
				   n_splits=30, # @todo
				   stratified=True, # @todo
				   output_file='../outputs/' + path_kw + '/results.pkl',
				   load_exist_results=True, # @todo
				   **kwargs):
	"""Run expriment.
	"""
	### Load existing results if possible.
	if load_exist_results and output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = []
#	results = []

	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split

#	if mode == 'classif':
#		stratified = True
#	else:
#		stratified = kwargs.get('stratified', True)

	cv = '811' # kwargs.get('cv')
	test_size = (0.1 if cv == '811' else 0.2)

#	import collections
#	if np.ceil(test_size * len(X)) < len(collections.Counter(families)):
#		stratified = False # ValueError: The test_size should be greater or equal to the number of classes.
#		# @todo: at least let each data in test set has different family?

	if stratified:
		# @todo: to guarantee that at least one sample is in train set (, valid set and test set).
		rs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
							   random_state=0) # @todo: to change it back.
		split_scheme = rs.split(X, families)
	else:
#		rs = ShuffleSplit(n_splits=n_splits, test_size=.1) #, random_state=0)
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
#		for setup in distances.keys():
#		print("{0} Mode".format(setup))
		setup_results = {}


		# directory to save all results.
		output_dir = '.'.join(output_file.split('.')[:-1])
		os.makedirs(output_dir, exist_ok=True)
		perf_train, perf_valid, perf_test, bparams, model = evaluate_model(
			X_train, y_train, X_valid, y_valid, X_test, y_test,
			output_dir=output_dir, trial_index=i-1, **kwargs)

#		setup_results['perf_app'] = perf_app
		setup_results['perf_train'] = perf_train
		setup_results['perf_valid'] = perf_valid
		setup_results['perf_test'] = perf_test
		setup_results['best_params'] = bparams
		# Save model.
		fn_model = '../outputs/' + path_kw + '/model.t' + str(i - 1) + '.h5'
# 		model.save(fn_model, save_format='h5') # 'tf'
# 		model.save_weights(fn_model, save_format='h5')
		setup_results['model'] = fn_model


#		### Show model stucture.
#		input_ = tf.keras.Input(shape=(100,), dtype='int32', name='input')
#		model_for_plot = tf.keras.Model(inputs=[input_],
#								   outputs=[model.predict(input_)])
		keras.utils.plot_model(model,
						 path_kw.strip('/').split('/')[0] + '_model.png',
						 show_dtype=True, show_shapes=True)

		print()
		print('Performance of current trial:')
		print("train: {0}.".format(perf_train))
		print("valid: {0}.".format(perf_valid))
		print("test: {0}.".format(perf_test))
		cur_results = setup_results
		results.append(cur_results)


		### Save the split.

		# Check if the (possible) existing file was updated by another thread during the computation of this split.
		if load_exist_results and output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
			with open(output_file, 'rb') as file:
				results_new = pickle.load(file)['results']
				if i - 1 <= len(results_new):
					print('Skip.')
					continue

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

#	test_losses()

	### Load dataset.
	for ds_name in ['poly200']: # ['poly200+sugarmono']: ['poly200']
		X, y, families = get_data(ds_name, descriptor='smiles', format_='smiles')

		cross_validate(X, y, families)