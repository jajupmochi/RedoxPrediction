#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:57:02 2022

@author: ljia
"""
import os
import sys
sys.path.insert(0, '../../')
import pickle
import numpy as np
from dataset.load_dataset import get_data
from redox_prediction.models import split_data

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from gklearn.utils import get_iters


# tf.config.run_functions_eagerly(True) # @todo: this is for debug only.
# tf.data.experimental.enable_debug_mode()


path_kw = '/miccio/mae/'


#%% Data processing.


DICTIONARY = ['c', 'n', 'o', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '=', '#', '$', ':', '/', '+', ')', '(', '@', '{', '}', '\\', ' ', '[', ']'] # @Alert!: carefully modify this dict as it may affect the separate_smiles() function.
DICTIONARY2 = ['c', 'n', 'o', 'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '=', '#', '$', ':', '/', '+', ')', '(', '@', '{', '}', '\\', ' ', '[', ']'] # add 'H' for dataset sugarmono. @Alert!: carefully modify this dict as it may affect the separate_smiles() function.
DICTIONARY_USED = DICTIONARY # @todo: change according to dataset.


def show_images(images, dir_='../figures/' + path_kw + '/images/'):
	import matplotlib.pyplot as plt

	os.makedirs(dir_, exist_ok=True)
	for i, img in enumerate(images):
		fn = os.path.join(dir_, str(i))
		plt.imshow(img)
		plt.imsave(fn + '.png', img)


# @Alert!: this function may need modification according to the DICTIONARY variable.
def separate_smiles(smiles, dict_=DICTIONARY_USED):
	"""Separate smiles strings into lists according to a dictionary.
	"""
	dict_len2 = [i for i in dict_ if len(i) == 2]
	lsmiles = []
	i = 0
	while i < len(smiles) - 1:
		s, s2 = smiles[i], smiles[i+1]
		if s + s2 in dict_len2:
			lsmiles.append(s + s2)
			i += 2
		else:
			lsmiles.append(s)
			i += 1

	# Append the last char if needed.
	if i < len(smiles):
		lsmiles.append(smiles[i])

	# Check.
	assert ''.join(lsmiles) == smiles

	return lsmiles


def smiles_to_image(smiles, nrow, dict_=DICTIONARY_USED, flatten=False):
	"""Convert a given smiles to a image.
	"""
	image = np.zeros((nrow, len(dict_)))
	for row, s in enumerate(smiles):
		idx = DICTIONARY_USED.index(s)
		image[row, idx] = 1

	if flatten:
		image = np.reshape(image, (-1, 1))

	return image


def smiles_to_images(smiles):
	"""Convert all smiles to images.
	"""
	lsmiles = [separate_smiles(s) for s in smiles]
	max_len = max([len(i) for i in lsmiles])
	images = [smiles_to_image(s, max_len) for s in lsmiles]
	return images


#%% Model


class FCN(tf.keras.Model):
	def __init__(self,
			  input_shape: tuple = (63, 39),
			  width1: int = 128,
			  width2: int = 128,
			  activation: str = 'ReLU',
			  dropout: float = 0.,
			  ):
		super(FCN, self).__init__()
		self.layer_lists = [
			tf.keras.layers.Flatten(input_shape=input_shape),
			# FC_a
			tf.keras.layers.Dense(width1),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation(activation),
			tf.keras.layers.Dropout(dropout),
			# FC_b
			tf.keras.layers.Dense(width2),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation(activation),
			tf.keras.layers.Dropout(dropout),
			# out layer.
			tf.keras.layers.Dense(1)
			]
# 		self.model = tf.keras.models.Sequential([
# 			tf.keras.layers.Flatten(input_shape=input_shape),
# 			# FC_a
# 			tf.keras.layers.Dense(width1),
# 			tf.keras.layers.BatchNormalization(),
# 			tf.keras.layers.Activation(activation),
# 			tf.keras.layers.Dropout(dropout),
# 			# FC_b
# 			tf.keras.layers.Dense(width2),
# 			tf.keras.layers.BatchNormalization(),
# 			tf.keras.layers.Activation(activation),
# 			tf.keras.layers.Dropout(dropout),
# 			# out layer.
# 			tf.keras.layers.Dense(1)
# 			])


	def call(self, x):
# 		print('I am FCN.call()!')
		for layer in self.layer_lists:
			x = layer(x)
		return x
# 		return self.model(x)


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
	def __init__(self, X_test, y_test,
			  X_train=None, y_train=None, X_valid=None, y_valid=None,
			  batch_size=8):
		super(EvaluateTestSet, self).__init__()
		self.X_test = X_test
		self.y_test = y_test
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid
		self.batch_size = batch_size


	def on_epoch_end(self, epoch, logs=None):
# 		print('-----------------')
# 		print(logs)
		results = self.model.evaluate(self.X_test, self.y_test,
								batch_size=self.batch_size, verbose=0)
		metrics_names = ['test_all_' + nm for nm in self.model.metrics_names]
		logs.update(dict(zip(metrics_names, results)))

		# Add evaluated train and valid scores (to see their diff with the
		# default outputs).
		if self.X_train is not None:
			res1 = self.model.evaluate(self.X_train, self.y_train,
							  batch_size=self.batch_size, verbose=0)
			names1 = ['train_all_' + nm for nm in self.model.metrics_names]
			logs.update(dict(zip(names1, res1)))
		if self.X_valid is not None:
			res2 = self.model.evaluate(self.X_valid, self.y_valid,
							  batch_size=self.batch_size, verbose=0)
			names2 = ['valid_all_' + nm for nm in self.model.metrics_names]
			logs.update(dict(zip(names2, res2)))
# 		print(dict(zip(metrics_names, results)))
# 		print('-----------------')


class EvaluateTransformed(keras.callbacks.Callback):
	"""Evalute metrics where the targets are transformed back using a
	transformer such as `sklearn.preprocessing.StandardScaler`.
	"""
	def __init__(self, transformer, X_test=None, y_test=None):
		super(EvaluateTransformed, self).__init__()
		self.transformer = transformer
		if X_test is not None:
			self.X_test = X_test
			self.y_test = y_test


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
			  X_train=None, y_train=None, X_valid=None, y_valid=None):
		"""
        display: Number of batches to wait before outputting loss.
		"""
		super(NBatchLogger, self).__init__()
		self.seen = 0
		self.display = display
		self.batch_size = batch_size
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid


	def on_train_batch_end(self, batch, logs={}):
# 		print(logs)
# 		self.seen += logs.get('size', 0)
		if self.seen < self.params['steps']:
			self.seen += 1
		else:
			self.seen = 1

		# Add evaluated train and valid scores (to see their diff with the
		# default outputs).
		if self.X_train is not None:
			res1 = self.model.evaluate(self.X_train, self.y_train,
							  batch_size=self.batch_size, verbose=0)
			names1 = ['eval_train_' + nm for nm in self.model.metrics_names]
			logs.update(dict(zip(names1, res1)))
		if self.X_valid is not None:
			res2 = self.model.evaluate(self.X_valid, self.y_valid,
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
	y_labels = ['loss (MAPE) (%)', 'MAE (K)', '$R^2$']
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


def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test,
				   scale_x=False, #@todo
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
#	y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)

	# Convert inputs to tensors.
	X_train, y_train = tf.constant(X_train), tf.constant(y_train)
	X_valid, y_valid = tf.constant(X_valid), tf.constant(y_valid)
	X_test, y_test = tf.constant(X_test), tf.constant(y_test)

	# Train and predict.
	# hyper-params: tune them one by one rather than grid search.
# 	from sklearn.model_selection import ParameterGrid
	params_grid = { # @todo
		'learning_rate': [0.01, 0.001, 1e-4], # [5e-4, 0.001, 1e-4] influential?, [5e-4]
		'beta1': [0.99, 0.9, 0.5, 0.1],
		'beta2': [0.999, 0.99, 0.9, 0.5, 0.1],
		'batch_size': [32, 8, 16, 64, 128, 256], # a liitle influential, [8]
		'hidden_feats': [1024, 8, 16, 64, 256, 2048], # influential?, [32]
		'dropout': [0., 0.2, 0.3],
		}

	MAE_min = np.inf
	MAE_all = []
	cur_params = {k: v[0] for k, v in params_grid.items()}
	for idx_params, param_name in get_iters(enumerate(cur_params), desc='Hyperparam tuning', length=len(cur_params)): # for each hyper-parameter:
		start_idx = (0 if idx_params == 0 else 1) # Avoid redundant computations.
		for param_val in params_grid[param_name][start_idx:]:
			cur_params[param_name] = param_val
			params = cur_params.copy() # (Depp) Copy so that the stored best values will not be overridden.

			batch_size = params['batch_size']

			# Initialize model.
		# 	input_shape = (len(X_train),X_train[0].shape[0], X_train[0].shape[1])
			model = FCN(input_shape=X_train.shape,
					 width1=params['hidden_feats'],
					 width2=params['hidden_feats'],
					 activation='ReLU',
					 dropout=params['dropout'])

			nb_epoch = 5000 # @todo: change it back

			# Choose loss function.
	# 		if '/mae/' in path_kw:
	# 			loss = tf.keras.metrics.mean_absolute_error # tf.keras.losses.MeanAbsoluteError() # tf.keras.metrics.MeanAbsoluteError()
	# 	# 	r2_score = tfa.metrics.RSquare
	# 		elif '/mse/' in path_kw:
	# 			loss = tf.keras.metrics.mean_squared_error
	# 		elif '/mape/' in path_kw:
			loss = tf.keras.metrics.mean_absolute_percentage_error

			model.compile(
				optimizer=Adam(
					learning_rate=params['learning_rate'],
					beta_1=params['beta1'],
					beta_2=params['beta2']
					),
				loss=loss,
				# metrics=['mae'])
				metrics=['mae', R_squared])
		# 			   metrics=['mae', loss, R_squared])
		# 	model.run_eagerly = True # @todo: this is for debug only.

			# callbacks
			EarlyStopping =	tf.keras.callbacks.EarlyStopping(
					monitor='val_loss',
					patience=50, # 100, # @todo
					verbose=1,
					restore_best_weights=True)
			TensorBoard = tf.keras.callbacks.TensorBoard(
				log_dir='../outputs/' + path_kw + '/tb_logs.t' + str(trial_index))
			# Train model.
			history = model.fit(X_train, y_train,
				   validation_data=(X_valid, y_valid),
				   callbacks=[
					   EvaluateTestSet(X_test, y_test, batch_size=batch_size),
		# 								X_train=X_train, y_train=y_train,
		# 								X_valid=X_valid, y_valid=y_valid),
						EarlyStopping,
		#				TensorBoard,
		# 				NBatchLogger(X_train=X_train, y_train=y_train,
		#  				 X_valid=None, y_valid=None, batch_size=batch_size),
						],
		# 				model_checkpoint,
		# 				CSVLogger(log_path[i], append=True)],
				   batch_size=batch_size,
				   epochs=nb_epoch,
				   shuffle=True,
				   verbose=0)

			# Record result if better.
			y_pred_valid = tf.squeeze(model.predict(X_valid, batch_size=batch_size), axis=1).numpy()
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
	y_pred_train = model.predict(X_train, batch_size=batch_size)
	y_pred_valid = model.predict(X_valid, batch_size=batch_size)
	y_pred_test = model.predict(X_test, batch_size=batch_size)
	if scale_y:
		y_pred_train = y_scaler.inverse_transform(y_pred_train)
		y_pred_valid = y_scaler.inverse_transform(y_pred_valid)
		y_pred_test = y_scaler.inverse_transform(y_pred_test)

	# Evaluate.
	train_accuracy, valid_accuracy, test_accuracy = {}, {}, {}
	for metric in [mean_absolute_error, mean_absolute_percentage_error,
				r2_score]:
		train_accuracy[metric.__name__] = metric(y_train, y_pred_train)
		valid_accuracy[metric.__name__] = metric(y_valid, y_pred_valid)
		test_accuracy[metric.__name__] = metric(y_test, y_pred_test)
# 	from sklearn.metrics import mean_absolute_error
# 	train_accuracy = mean_absolute_error(y_train, y_pred_train)
# 	valid_accuracy = mean_absolute_error(y_valid, y_pred_valid)
# 	test_accuracy = mean_absolute_error(y_test, y_pred_test)


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
				   output_file=None,
				   load_exist_results=True, # @todo
				   **kwargs):
	"""Run expriment.
	"""
	# output_file can not be set in the argument directly, as it contains
	# `path_kw`, which is modified in the `__main__`.
	if output_file is None:
		output_file='../outputs/' + path_kw + '/results.pkl'

	### Load existing results if possible.
	if load_exist_results and output_file is not None and os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = []
#	results = []

	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split

# 	if mode == 'classif':
# 		stratified = True # @todo: to change it as needed.
# 	else:
# 		stratified = kwargs.get('stratified', True)

	cv = '811' # kwargs.get('cv')
	test_size = (0.1 if cv == '811' else 0.2)

# 	import collections
# 	if np.ceil(test_size * len(X)) < len(collections.Counter(families)):
# 		stratified = False # ValueError: The test_size should be greater or equal to the number of classes.
# 		# @todo: at least let each data in test set has different family?

	if stratified:
		# @todo: to guarantee that at least one sample is in train set (, valid set and test set).
		rs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
							   random_state=0) # @todo: to change it back.
		split_scheme = rs.split(X, families)
	else:
# 		rs = ShuffleSplit(n_splits=n_splits, test_size=.1) #, random_state=0)
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
# 		for setup in distances.keys():
# 		print("{0} Mode".format(setup))
		setup_results = {}


		# directory to save all results.
		output_dir = '.'.join(output_file.split('.')[:-1])
		os.makedirs(output_dir, exist_ok=True)
		perf_train, perf_valid, perf_test, bparams, model = evaluate_model(
			X_train, y_train, X_valid, y_valid, X_test, y_test,
			output_dir=output_dir, trial_index=i-1, **kwargs)

# 		setup_results['perf_app'] = perf_app
		setup_results['perf_train'] = perf_train
		setup_results['perf_valid'] = perf_valid
		setup_results['perf_test'] = perf_test
		setup_results['best_params'] = bparams
		# Save model.
		fn_model = '../outputs/' + path_kw + '/model.t' + str(i - 1)
		model.save(fn_model)
		setup_results['model'] = fn_model


# 		### Show model stucture.
# 		input_ = tf.keras.Input(shape=(100,), dtype='int32', name='input')
# 		model_for_plot = tf.keras.Model(inputs=[input_],
# 								   outputs=[model.predict(input_)])
# 		keras.utils.plot_model(model_for_plot, '' + path_kw + '_fcn_model.png',
# 						  show_shapes=True)

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


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('-D', "--ds_name", type=str, help='the name of dataset')

	args = parser.parse_args()

	return args


if __name__ == '__main__':

#	test_losses()

	args = parse_args()
	DS_Name_List = (['poly200r'] if args.ds_name is None else [args.ds_name]) # ['poly200+sugarmono']:

	### Load dataset.
	for ds_name in DS_Name_List: #  : ValueError: 'H' is not in list,
		path_kw = '/' + ds_name + path_kw
		smiles, y, families = get_data(ds_name, descriptor='smiles')
		images = smiles_to_images(smiles)
# 		y = np.reshape(y, (-1, 1))
# 		show_images(images, dir_=('../figures/' + path_kw + '/images/' + ds_name))
		cross_validate(images, y, families)