#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:26:47 2021

@author: ljia
"""
import sys
import os
sys.path.insert(1, '../../')
from dataset.load_dataset import load_dataset


#%% GCN


def xp_GCN():

	import deepchem as dc
	from deepchem.models import GCNModel, GATModel


	# Prepare dataset.
	# data = load_dataset('thermophysical', format_='smiles')
	# smiles = data['X']
	# smiles = [s for i, s in enumerate(smiles) if i not in [192, 352, 380, 381]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	# y = data['targets']
	# y = [y for i, y in enumerate(y) if i not in [192, 352, 380, 381]]

	data = load_dataset('polyacrylates200', format_='smiles')
	smiles = data['X']
	smiles = [s for i, s in enumerate(smiles) if i not in [6]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	y = data['targets']
	y = [y for i, y in enumerate(y) if i not in [6]]


	featurizer = dc.feat.MolGraphConvFeaturizer()
	X = featurizer.featurize(smiles)
	dataset = dc.data.NumpyDataset(X=X, y=y)

	# Train model.
	model = GCNModel(mode='regression', n_tasks=1,
	 				 batch_size=16, learning_rate=0.001)
	# model = GATModel(mode='regression', n_tasks=1,
	# 				 batch_size=16, learning_rate=0.001)
	loss = model.fit(dataset, nb_epoch=5)


	# # Seperate dataset.
	# train_dataset, valid_dataset, test_dataset = dataset

	# # Train model.
	# model = GCNModel(mode='regression', n_tasks=1,
	# 				 batch_size=16, learning_rate=0.001)
	# # model = GATModel(mode='regression', n_tasks=1,
	# # 				 batch_size=16, learning_rate=0.001)
	# loss = model.fit(train_dataset, nb_epoch=5)




	# 192: (OCCCC)14O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O
	# 352: C[Si](C)(Cl)Cl,CO[Si](C)(C)OC
	# 380: (OCCCC)14O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O
	# 381: C{-}(OC(=O)C)C{n+}


#%% GCN new.


def xp_GCN_2():
	import numpy as np
	import tensorflow as tf
	import deepchem as dc

	# Run before every test for reproducibility
	def seed_all():
		np.random.seed(123)
		tf.random.set_seed(123)


	# Prepare dataset.
	# data = load_dataset('thermophysical', format_='smiles')
	# smiles = data['X']
	# smiles = [s for i, s in enumerate(smiles) if i not in [192, 352, 380, 381]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	# y = data['targets']
	# y = [y for i, y in enumerate(y) if i not in [192, 352, 380, 381]]

	data = load_dataset('polyacrylates200', format_='smiles')
	smiles = data['X']
	smiles = [s for i, s in enumerate(smiles) if i not in [6]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	y = data['targets']
	y = [y for i, y in enumerate(y) if i not in [6]]
	y = np.reshape(y, (len(y), 1))


	featurizer = dc.feat.MolGraphConvFeaturizer()
	X = featurizer.featurize(smiles)
	train_dataset = dc.data.NumpyDataset(X=X, y=y)


	# RMS, averaged across tasks
	avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

	# 	model = dc.models.GraphConvModel(len(chembl_tasks), batch_size=128, mode='regression')
	model = dc.models.GCNModel(mode='regression', n_tasks=1,
						 batch_size=16)
# 	model = dc.models.GATModel(mode='regression', n_tasks=len(chembl_tasks),
# 						 batch_size=128, learning_rate=0.001)


	# Fit trained model
	loss = model.fit(train_dataset, nb_epoch=100)
	print(loss)

	# We now evaluate our fitted model on our training and validation sets
	train_scores = model.evaluate(train_dataset, [avg_rms])
	print(train_scores)
	assert train_scores['mean-rms_score'] < 10.00

# 	valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
# 	print(valid_scores)
# 	assert valid_scores['mean-rms_score'] < 10.00


#%% GCN with testing data.


def xp_GCN_2_testing():
	import numpy as np
	import tensorflow as tf
	import deepchem as dc

	# Run before every test for reproducibility
	def seed_all():
		np.random.seed(123)
		tf.random.set_seed(123)


	data = load_dataset('polyacrylates200', format_='smiles')
	smiles = data['X']
	smiles = [s for i, s in enumerate(smiles) if i not in [6]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	y = data['targets']
	y = [y for i, y in enumerate(y) if i not in [6]]
	y = np.reshape(y, (len(y), 1))


	featurizer = dc.feat.MolGraphConvFeaturizer()
	X = featurizer.featurize(smiles)
	dataset = dc.data.NumpyDataset(X=X, y=y)


	### Split data.
	splitter = dc.splits.RandomSplitter()
	# Splitting dataset into train and test datasets
	train_dataset, test_dataset = splitter.train_test_split(dataset)


	# RMS, averaged across tasks
	avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

	# 	model = dc.models.GraphConvModel(len(chembl_tasks), batch_size=128, mode='regression')
	model = dc.models.GCNModel(mode='regression', n_tasks=1,
						 batch_size=16)
# 	model = dc.models.GATModel(mode='regression', n_tasks=len(chembl_tasks),
# 						 batch_size=128, learning_rate=0.001)


	# Fit trained model
	loss = model.fit(train_dataset, nb_epoch=100)
	print(loss)

	# We now evaluate our fitted model on our training and validation sets
	train_scores = model.evaluate(train_dataset, [avg_rms])
	print(train_scores)
# 	assert train_scores['mean-rms_score'] < 10.00

	test_scores = model.evaluate(test_dataset, [avg_rms])
	print(test_scores)
# 	assert test_scores['mean-rms_score'] < 10.00


#%% GCN with testing data and transformer.


def xp_GCN_2_testing_transformer():
	import numpy as np
	import tensorflow as tf
	import deepchem as dc

	# Run before every test for reproducibility
	def seed_all():
		np.random.seed(123)
		tf.random.set_seed(123)


	data = load_dataset('polyacrylates200', format_='smiles')
	smiles = data['X']
	smiles = [s for i, s in enumerate(smiles) if i not in [6]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	y = data['targets']
	y = [y for i, y in enumerate(y) if i not in [6]]
	y = np.reshape(y, (len(y), 1))


	featurizer = dc.feat.MolGraphConvFeaturizer()
	X = featurizer.featurize(smiles)
	dataset = dc.data.NumpyDataset(X=X, y=y)


	### Split data.
	splitter = dc.splits.RandomSplitter()
	# Splitting dataset into train and test datasets
	train_dataset, test_dataset = splitter.train_test_split(dataset)


	# Transform.
	transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
	train_dataset = transformer.transform(train_dataset)
	test_dataset = transformer.transform(test_dataset)


	# RMS, averaged across tasks
	avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

	# 	model = dc.models.GraphConvModel(len(chembl_tasks), batch_size=128, mode='regression')
	model = dc.models.GCNModel(mode='regression', n_tasks=1,
						 batch_size=16)
# 	model = dc.models.GATModel(mode='regression', n_tasks=len(chembl_tasks),
# 						 batch_size=128, learning_rate=0.001)


	# Fit trained model
	loss = model.fit(train_dataset, nb_epoch=100)
	print(loss)

	# We now evaluate our fitted model on our training and validation sets
	train_scores = model.evaluate(train_dataset, [avg_rms], [transformer])
	print(train_scores)
# 	assert train_scores['mean-rms_score'] < 10.00

	test_scores = model.evaluate(test_dataset, [avg_rms], [transformer])
	print(test_scores)
# 	assert test_scores['mean-rms_score'] < 10.00


#%% With transformer.


def xp_GCN_2_transformer():
	import numpy as np
	import tensorflow as tf
	import deepchem as dc

	# Run before every test for reproducibility
	def seed_all():
		np.random.seed(123)
		tf.random.set_seed(123)


	# Prepare dataset.
	# data = load_dataset('thermophysical', format_='smiles')
	# smiles = data['X']
	# smiles = [s for i, s in enumerate(smiles) if i not in [192, 352, 380, 381]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	# y = data['targets']
	# y = [y for i, y in enumerate(y) if i not in [192, 352, 380, 381]]

	data = load_dataset('polyacrylates200', format_='smiles')
	smiles = data['X']
	smiles = [s for i, s in enumerate(smiles) if i not in [6]]
	# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
	y = data['targets']
	y = [y for i, y in enumerate(y) if i not in [6]]
	y = np.reshape(y, (len(y), 1))


	featurizer = dc.feat.MolGraphConvFeaturizer()
	X = featurizer.featurize(smiles)
	train_dataset = dc.data.NumpyDataset(X=X, y=y)

	# Transform.
	transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
	train_dataset = transformer.transform(train_dataset)

	# RMS, averaged across tasks
	avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

	# 	model = dc.models.GraphConvModel(len(chembl_tasks), batch_size=128, mode='regression')
	model = dc.models.GCNModel(mode='regression', n_tasks=1,
						 batch_size=16)
# 	model = dc.models.GATModel(mode='regression', n_tasks=len(chembl_tasks),
# 						 batch_size=128, learning_rate=0.001)


	# Fit trained model
	loss = model.fit(train_dataset, nb_epoch=100)
	print(loss)

	# We now evaluate our fitted model on our training and validation sets
	train_scores = model.evaluate(train_dataset, [avg_rms], [transformer])
	print(train_scores)
	assert train_scores['mean-rms_score'] < 10.00

# 	valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
# 	print(valid_scores)
# 	assert valid_scores['mean-rms_score'] < 10.00


if __name__ == '__main__':
# 	xp_GCN()
# 	xp_GCN_2()
# 	xp_GCN_2_testing()
	xp_GCN_2_testing_transformer()