#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:34:32 2021

@author: ljia

This script is built to:
	- test if my implementation of featurizer and MPNN model are correct using the keras MPNN example data;
	- compare the performance of using different features on the example data;
	- test roughly if the model can learn on poly200_exp;
	- compare the performance of using different features on poly200_exp;
"""
import sys
import os
sys.path.insert(1, '../')
from dataset.load_dataset import load_dataset

from mpnn_transformer_model import MPNNTransformerModel, MPNNDataset

#%%
# =============================================================================
# Import packages.
# =============================================================================

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
import logging

# tf.get_logger().setLevel(logging.ERROR)
# Temporary suppress warnings and RDKit logs
# warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

np.random.seed(42)
tf.random.set_seed(42)

# tf.config.run_functions_eagerly(True) # @todo: this is for debug only.
# tf.config.experimental_run_functions_eagerly(True)



#%%
# =============================================================================
# Get dataset.
# =============================================================================


# # Prepare dataset.

sys.path.insert(0, '../')
import numpy as np
from dataset.load_dataset import get_data

### Load dataset.
for ds_name in ['poly200']: # ['poly200+sugarmono']: ['poly200']

	X, y, families = get_data(ds_name, descriptor='smiles', format_='smiles')

smiles = np.array(X)
y = np.array(y)



# csv_path = keras.utils.get_file(
# 	"BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
# )

# df = pd.read_csv(csv_path, usecols=[1, 2, 3])
# df.iloc[96:104] # @todo: change as needed


### Featurizer.

from dataset.feat import MolGNNFeaturizer # @todo: change as needed
af_allowable_sets = {
 	'atom_type': ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
# 	'atom_type': ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"],
 	# 'formal_charge': None, # [-2, -1, 0, 1, 2],
 	'hybridization': ['SP', 'SP2', 'SP3'],
# 	'hybridization': ['S', 'SP', 'SP2', 'SP3'],
 	# 'acceptor_donor': ['Donor', 'Acceptor'],
 	# 'aromatic': [True, False],
 	# 'degree': [0, 1, 2, 3, 4, 5],
 	'n_valence': [0, 1, 2, 3, 4, 5, 6],
	'total_num_Hs': [0, 1, 2, 3, 4],
 	# 'chirality': ['R', 'S'],
	}
bf_allowable_sets = {
	'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
 	# 'same_ring': [True, False],
	'conjugated': [True, False],
 	# 'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'],
	}
featurizer = MolGNNFeaturizer(use_edges=True,
						   use_partial_charge=False,
						   af_allowable_sets=af_allowable_sets,
						   bf_allowable_sets=bf_allowable_sets
						   )


# # Shuffle array of indices ranging from 0 to 2049 # @todo: change as needed
# permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# # Train set: 80 % of data
# train_index = permuted_indices[: int(df.shape[0] * 0.8)]
# x_train = featurizer.featurize(df.iloc[train_index].smiles)
# # y_train = df.iloc[train_index].target
# y_train = df.iloc[train_index].p_np

# # Valid set: 19 % of data
# valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
# x_valid = featurizer.featurize(df.iloc[valid_index].smiles)
# # y_valid = df.iloc[valid_index].target
# y_valid = df.iloc[valid_index].p_np

# # Test set: 1 % of data
# test_index = permuted_indices[int(df.shape[0] * 0.99) :]
# x_test = featurizer.featurize(df.iloc[test_index].smiles)
# # y_test = df.iloc[test_index].target
# y_test = df.iloc[test_index].p_np


# Shuffle array of indices ranging from 0 to 2049
permuted_indices = np.random.permutation(np.arange(len(smiles)))

# Train set: 80 % of data
train_index = permuted_indices[: int(len(smiles) * 0.8)]
x_train = featurizer.featurize(smiles[train_index])
# y_train = df.iloc[train_index].target
y_train = y[train_index]

# Valid set: 10 % of data
valid_index = permuted_indices[int(len(smiles) * 0.8) : int(len(smiles) * 0.9)]
x_valid = featurizer.featurize(smiles[valid_index])
# y_valid = df.iloc[valid_index].target
y_valid = y[valid_index]

# Test set: 10 % of data
test_index = permuted_indices[int(len(smiles) * 0.9) :]
x_test = featurizer.featurize(smiles[test_index])
# y_test = df.iloc[test_index].target
y_test = y[test_index]


# Scale targets.
from sklearn.preprocessing import StandardScaler
y_scaler = StandardScaler().fit(np.reshape(y_train, (-1, 1)))
y_train = y_scaler.transform(np.reshape(y_train, (-1, 1)))
y_valid = y_scaler.transform(np.reshape(y_valid, (-1, 1)))
y_test = y_scaler.transform(np.reshape(y_test, (-1, 1)))
y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)



# # =============================================================================
# # Test the functions.
# # =============================================================================
# # print(f"Name:\t{df.name[100]}\nSMILES:\t{df.smiles[100]}\ntarget:\t{df.target[100]}")
# print(f"Name:\t{df.name[100]}\nSMILES:\t{df.smiles[100]}\nBBBP:\t{df.p_np[100]}")
# molecule = molecule_from_smiles(df.iloc[100].smiles)
# print("Molecule:")
# molecule

# graph = graph_from_molecule(molecule)
# print("Graph (including self-loops):")
# print("\tatom features\t", graph[0].shape)
# print("\tbond features\t", graph[1].shape)
# print("\tpair indices\t", graph[2].shape)


# # =============================================================================
# # Create a tf.data.Dataset.
# # =============================================================================

# def prepare_batch(x_batch, y_batch):
# 	"""Merges (sub)graphs of batch into a single global (disconnected) graph
# 	"""

# 	atom_features, bond_features, pair_indices = x_batch

# 	# Obtain number of atoms and bonds for each graph (molecule)
# 	num_atoms = atom_features.row_lengths()
# 	num_bonds = bond_features.row_lengths()

# 	# Obtain partition indices (molecule_indicator), which will be used to
# 	# gather (sub)graphs from global graph in model later on
# 	molecule_indices = tf.range(len(num_atoms))
# 	molecule_indicator = tf.repeat(molecule_indices, num_atoms)

# 	# Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
# 	# 'pair_indices' (and merging ragged tensors) actualizes the global graph
# 	gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
# 	increment = tf.cumsum(num_atoms[:-1])
# 	increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
# 	pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
# 	pair_indices = pair_indices + increment[:, tf.newaxis]
# 	atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
# 	bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

# 	return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


# def MPNNDataset(X, y, batch_size=32, shuffle=False):
# # 	return prepare_batch(X, y)
# 	dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
# 	if shuffle:
# 		dataset = dataset.shuffle(1024)
# 	return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)
# #	 return dataset.batch(batch_size).map(
# # 		(lambda x: tf.py_function(prepare_batch, [x], [tf.int64])), -1)


# #%%
# # =============================================================================
# # Model
# # =============================================================================

# class EdgeNetwork(layers.Layer):
# 	def __init__(self, **kwargs):
# 		super().__init__(**kwargs)

# 	def build(self, input_shape):
# 		self.atom_dim = input_shape[0][-1]
# 		self.bond_dim = input_shape[1][-1]
# 		self.kernel = self.add_weight(
# 			shape=(self.bond_dim, self.atom_dim * self.atom_dim),
# 			trainable=True,
# 			initializer="glorot_uniform",
# 			name="kernel",
# 		)
# 		self.bias = self.add_weight(
# 			shape=(self.atom_dim * self.atom_dim), trainable=True, initializer="zeros", name="bias",
# 		)
# 		self.built = True

# 	def call(self, inputs):
# 		atom_features, bond_features, pair_indices = inputs

# 		# Apply linear transformation to bond features
# 		bond_features = tf.matmul(bond_features, self.kernel) + self.bias

# 		# Reshape for neighborhood aggregation later
# 		bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

# 		# Obtain atom features of neighbors
# 		atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
# 		atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

# 		# Apply neighborhood aggregation
# 		transformed_features = tf.matmul(bond_features, atom_features_neighbors)
# 		transformed_features = tf.squeeze(transformed_features, axis=-1)
# 		aggregated_features = tf.math.unsorted_segment_sum(
# 			transformed_features,
# 			pair_indices[:, 0],
# 			num_segments=tf.shape(atom_features)[0],
# 		)
# 		return aggregated_features


# class MessagePassing(layers.Layer):
# 	def __init__(self, units, steps=4, **kwargs):
# 		super().__init__(**kwargs)
# 		self.units = units
# 		self.steps = steps

# 	def build(self, input_shape):
# 		self.atom_dim = input_shape[0][-1]
# 		self.message_step = EdgeNetwork()
# 		self.pad_length = max(0, self.units - self.atom_dim)
# 		self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
# 		self.built = True

# 	def call(self, inputs):
# 		atom_features, bond_features, pair_indices = inputs

# 		# Pad atom features if number of desired units exceeds atom_features dim.
# 		# Alternatively, a dense layer could be used here.
# 		atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

# 		# Perform a number of steps of message passing
# 		for i in range(self.steps):
# 			# Aggregate information from neighbors
# 			atom_features_aggregated = self.message_step(
# 				[atom_features_updated, bond_features, pair_indices]
# 			)

# 			# Update node state via a step of GRU
# 			atom_features_updated, _ = self.update_step(
# 				atom_features_aggregated, atom_features_updated
# 			)
# 		return atom_features_updated


# # =============================================================================
# # Readout.
# # =============================================================================

# class PartitionPadding(layers.Layer):
# 	def __init__(self, batch_size, **kwargs):
# 		super().__init__(**kwargs)
# 		self.batch_size = batch_size

# 	def call(self, inputs):

# 		atom_features, molecule_indicator = inputs

# 		# Obtain subgraphs
# 		atom_features_partitioned = tf.dynamic_partition(
# 			atom_features, molecule_indicator, self.batch_size
# 		)

# 		# Pad and stack subgraphs
# 		num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
# 		max_num_atoms = tf.reduce_max(num_atoms)
# 		atom_features_stacked = tf.stack(
# 			[
# 				tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
# 				for f, n in zip(atom_features_partitioned, num_atoms)
# 			],
# 			axis=0,
# 		)

# 		# Remove empty subgraphs (usually for last batch in dataset)
# 		gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
# 		gather_indices = tf.squeeze(gather_indices, axis=-1)
# 		return tf.gather(atom_features_stacked, gather_indices, axis=0)


# class TransformerEncoderReadout(layers.Layer):
# 	def __init__(
# 		self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
# 	):
# 		super().__init__(**kwargs)

# 		self.partition_padding = PartitionPadding(batch_size)
# 		self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
# 		self.dense_proj = keras.Sequential(
# 			[layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
# 		)
# 		self.layernorm_1 = layers.LayerNormalization()
# 		self.layernorm_2 = layers.LayerNormalization()
# 		self.average_pooling = layers.GlobalAveragePooling1D()

# 	def call(self, inputs):
# 		x = self.partition_padding(inputs)
# 		padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
# 		padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
# 		attention_output = self.attention(x, x, attention_mask=padding_mask)
# 		proj_input = self.layernorm_1(x + attention_output)
# 		proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
# 		return self.average_pooling(proj_output)


# # =============================================================================
# # Message Passing Neural Network (MPNN)
# # =============================================================================

# def MPNNModel(
# 	atom_dim,
# 	bond_dim,
# 	batch_size=32,
# 	message_units=64,
# 	message_steps=4,
# 	num_attention_heads=8,
# 	dense_units=512,
# ):

# 	atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
# 	bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
# 	pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
# 	molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")

# 	x = MessagePassing(message_units, message_steps)(
# 		[atom_features, bond_features, pair_indices]
# 	)

# 	x = TransformerEncoderReadout(
#         num_attention_heads, message_units, dense_units, batch_size
#     )([x, molecule_indicator])

# 	x = layers.Dense(dense_units, activation="relu")(x)
# 	x = layers.Dense(1, activation="sigmoid")(x)

# 	model = keras.Model(
# 		inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
# 		outputs=[x],
# 	)
# 	return model


# mpnn = MPNNModel(
# 	atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
# )


mpnn = MPNNTransformerModel(
		x_train[0][0][0].shape[0],
		x_train[1][0][0].shape[0],
		batch_size=32,
		message_units=64,
		message_steps=4,
		num_attention_heads=8,
		dense_units=512,
		)

# @todo: change as needed
# mpnn.compile(
# 	loss=keras.losses.BinaryCrossentropy(),
# 	optimizer=keras.optimizers.Adam(learning_rate=5e-4),
# 	metrics=[keras.metrics.AUC(name="AUC")],
# )


mpnn.compile(
# 	loss=keras.losses.MeanAbsoluteError(),
	loss=keras.metrics.mean_absolute_error,
	optimizer=keras.optimizers.Adam(learning_rate=5e-4),
# 	metrics=[keras.metrics.MeanAbsoluteError(name="MAE")],
	metrics=['mae'],
)

keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)


#%%
# =============================================================================
# Training.
# =============================================================================

train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)

# @todo: change as needed
# history = mpnn.fit(
# 	train_dataset,
# 	validation_data=valid_dataset,
# 	epochs=40,
# 	verbose=2,
# 	class_weight={0: 2.0, 1: 0.5},
# )

history = mpnn.fit(
	train_dataset,
	validation_data=valid_dataset,
	epochs=100, # @todo: change as needed
	verbose=2,
)


# @todo: change as needed
# plt.figure(figsize=(10, 6))
# plt.plot(history.history["AUC"], label="train AUC")
# plt.plot(history.history["val_AUC"], label="valid AUC")
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("AUC", fontsize=16)
# plt.legend(fontsize=16)

plt.figure(figsize=(10, 6))
plt.plot(history.history["mae"], label="train mae")
plt.plot(history.history["val_mae"], label="valid mae")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("MAE", fontsize=16)
plt.legend(fontsize=16)


# =============================================================================
# Predicting.
# =============================================================================

# @todo: change as needed
y_true = [y[index] for index in test_index]
y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)
y_pred = y_scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_true, y_pred)

# molecules = [featurizer.featurize(df.smiles.values[index]) for index in test_index]
# y_true = [df.p_np.values[index] for index in test_index]
# y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)

# legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
# MolsToGridImage(molecules, molsPerRow=4, legends=legends)
