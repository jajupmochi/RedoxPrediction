#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:17:07 2022

@author: ljia
"""
import os
import deepchem as dc
import numpy as np
# import wandb
# wandb.login()
# from deepchem.models import WandbLogger
from deepchem.models import ValidationCallback


def get_best_scores(f_op_train, f_op_valid, f_op_test, save_dir_valid):

	# Find the best validation score and # of epochs.
	valid_score = np.inf
	f_op_valid.seek(0)
	for line in f_op_valid:
		l_split = line.split(' ')
		cur_score = float(l_split[-1].split('=')[-1].strip())
		if cur_score < valid_score:
			valid_score = cur_score
			best_nb_epoch = int(l_split[1].strip())


	# Get corresponding training and test scores.
	# training score.
	f_op_train.seek(0)
	for i in range(1, best_nb_epoch):
		f_op_train.read()
	line = f_op_train.read()
	train_score = float(line.split('=')[-1].strip())
	# test score.
	f_op_test.seek(0)
	for i in range(1, best_nb_epoch):
		f_op_test.read()
	line = f_op_test.read()
	test_score = float(line.split('=')[-1].strip())


	# Get the best model.
	best_model = dc.models.GATModel(model_dir=save_dir_valid, n_tasks=1)
	best_model.restore()


	return train_score, valid_score, test_score, best_model, best_nb_epoch


def split_data(D, y, train_index, test_index):
	D_app = [D[i] for i in train_index]
	D_test = [D[i] for i in test_index]
	y_app = [y[i] for i in train_index]
	y_test = [y[i] for i in test_index]
	return D_app, D_test, y_app, y_test


def evaluate_GCN(train_dataset, valid_dataset, test_dataset, mode='reg',
				 nb_epoch=100, output_dir=None, trial_index=1, **kwargs):
	### Get hyperparameters.
	model_name = (kwargs['model'] if 'model' in kwargs else 'GCN')
	feature_scaling = (kwargs['feature_scaling'] if 'feature_scaling' in kwargs else 'standard_y')
	metric_name = (kwargs['metric'] if 'metric' in kwargs else 'RMSE')

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

	### Set logger and callback.
	# ------------------------------
	os.makedirs(output_dir, exist_ok=True)
	# show and save validation performance.
	fn_op_valid = os.path.join(output_dir, 'valid' + str(trial_index) + '.txt')
	save_dir_valid = os.path.join(output_dir, 'valid' + str(trial_index) + '/')
	f_op_valid = open(fn_op_valid, 'w+')
# 	logger = WandbLogger(name=model_name, entity='COBRA', project='OCTOPUSSY')
	vc_valid_log = ValidationCallback(valid_dataset, interval=1, metrics=[metric],
							   output_file=f_op_valid,
							   save_dir=save_dir_valid,
							   save_on_minimum=True,
							   transformers=transformers)
	vc_valid_show = ValidationCallback(valid_dataset, interval=10, metrics=[metric],
							   transformers=transformers)

	# show and save test performance.
	fn_op_test = os.path.join(output_dir, 'test' + str(trial_index) + '.txt')
	save_dir_test = os.path.join(output_dir, 'test' + str(trial_index) + '/')
	f_op_test = open(fn_op_test, 'w+')
# 	logger = WandbLogger(name=model_name, entity='COBRA', project='OCTOPUSSY')
	vc_test_log = ValidationCallback(test_dataset, interval=1, metrics=[metric],
							   output_file=f_op_test,
							   save_dir=save_dir_test,
							   save_on_minimum=True,
							   transformers=transformers)
# 	vc_test_show = ValidationCallback(test_dataset, interval=10, metrics=[metric],
# 							   transformers=transformers)

	# show and save training performance.
	fn_op_train = os.path.join(output_dir, 'train' + str(trial_index) + '.txt')
	save_dir_train = os.path.join(output_dir, 'train' + str(trial_index) + '/')
	f_op_train = open(fn_op_train, 'w+')
# 	logger = WandbLogger(name=model_name, entity='COBRA', project='OCTOPUSSY')
	vc_train_log = ValidationCallback(train_dataset, interval=1, metrics=[metric],
							   output_file=f_op_train,
							   save_dir=save_dir_train,
							   save_on_minimum=True,
							   transformers=transformers)
# 	vc_train_show = ValidationCallback(train_dataset, interval=10, metrics=[metric],
# 							   transformers=transformers)

# 	# ------------------------------
# 	import tensorflow as tf
# 	from tensorflow.keras.callbacks import CSVLogger
# 	csv_logger = CSVLogger('logs/log.csv', append=True)


	# 	model = dc.models.GraphConvModel(len(chembl_tasks), batch_size=128, mode='regression')
	if model_name.lower() == 'gcn':
		model = dc.models.GCNModel(mode='regression', n_tasks=1, batch_size=16)
# 							 wandb=logger)
	elif model_name.lower() == 'gat':
		model = dc.models.GATModel(mode='regression', n_tasks=1, batch_size=16)
# 							 model_dir='model_test/', log_frequency=10)
# 							 wandb=logger)
# 		model = dc.models.GATModel(mode='regression', n_tasks=len(chembl_tasks),
# 							 batch_size=128, learning_rate=0.001)
	else:
		raise ValueError('Model "%s" can not be recognized.' % model_name)


	# Fit trained model
	loss = model.fit(train_dataset, nb_epoch=nb_epoch,
				  callbacks=[vc_valid_log, vc_valid_show, vc_test_log, vc_train_log])
# 	logger.finish()
# 	print(loss)

	### Get best scores and model.

	# We now evaluate our fitted model on our training and validation sets
	train_scores = model.evaluate(train_dataset, [metric], transformers)[kw_metric]
	valid_scores = model.evaluate(valid_dataset, [metric], transformers)[kw_metric]
	print('training scores:', train_scores)
	print('validation scores:', train_scores)
# 	assert train_scores['mean-rms_score'] < 10.00
	test_scores = model.evaluate(test_dataset, [metric], transformers)[kw_metric]
	print('test scores:', test_scores)
# 	assert valid_scores['mean-rms_score'] < 10.00


	train_scores, valid_scores, test_scores, best_model, best_nb_epochs = get_best_scores(
		f_op_train, f_op_valid, f_op_test, save_dir_valid)


	# Close log files.
	f_op_valid.close()
	f_op_test.close()
	f_op_train.close()

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


	return train_scores, valid_scores, test_scores, model


def xp_GCN(smiles, y_all, mode='reg', nb_epoch=100, output_file=None, **kwargs):
	'''
	Perform a GCN regressor on given dataset.
	'''
	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, train_test_split


# 	stratified = False
	if mode == 'classif':
		stratified = True

# 	if stratified:
# 		rs = StratifiedShuffleSplit(n_splits=5, test_size=.2, random_state=None)
# 	else:
# # 		rs = ShuffleSplit(n_splits=10, test_size=.1) #, random_state=0)
# 		rs = ShuffleSplit(n_splits=5, test_size=.1, random_state=0)
	rs = ShuffleSplit(n_splits=5, test_size=.2, random_state=0) # @todo: to change it back.

# 	if stratified:
# 		split_scheme = rs.split(Gn, stratified_y)
# 	else:
# 		split_scheme = rs.split(Gn)
	split_scheme = rs.split(smiles)

	results = []
	i = 1
	for app_index, test_index in split_scheme:
		print()
		print("Split {0}/{1}".format(i, 5))
		i = i + 1
		cur_results = {}
		# Get splitted data
		G_app, G_test, y_app, y_test = split_data(smiles, y_all,
												  app_index, test_index)

		# Split evaluation set.
		G_train, G_valid, y_train, y_valid = train_test_split(G_app, y_app,
														test_size=0.2/0.8,
														random_state=0,
														shuffle=True)

		cur_results['y_app'] = y_app
		cur_results['y_test'] = y_test
		cur_results['app_index'] = app_index
		cur_results['test_index'] = test_index

		# Feed distances will all methods to compare
# 		distances = {}
# 		distances['random'] = compute_D_random(G_app, G_test, ed_method, **kwargs)
# 		distances['expert'] = compute_D_expert(G_app, G_test, ed_method, **kwargs)
# 		distances['fitted'] = compute_D_fitted(
#  			G_app, y_app, G_test,
#  			y_distance=y_distance,
#  			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
#  			**kwargs)
		kwargs['nb_trial'] = i - 1
# 		distances['Garcia-Hernandez2020'] = compute_D_GH2020(G_app, G_test,
# 													   ed_method, **kwargs)

# 		for setup in distances.keys():
# 		print("{0} Mode".format(setup))
		setup_results = {}
# 		D_app, D_test, edit_costs = distances[setup]
# 		setup_results['D_app'] = D_app
# 		setup_results['D_test'] = D_test
# 		setup_results['edit_costs'] = edit_costs
# 		print(edit_costs)

		# featurize.
		featurizer = dc.feat.MolGraphConvFeaturizer()
# 		X_app = featurizer.featurize(G_app)
		X_train = featurizer.featurize(G_train)
		X_valid = featurizer.featurize(G_valid)
		X_test = featurizer.featurize(G_test)
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
# 		perf_app, perf_test, clf = evaluate_GCN(
		# directory to save all results.
		output_dir = '.'.join(output_file.split('.')[:-1])
		perf_train, perf_valid, perf_test, clf = evaluate_GCN(
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

	return results