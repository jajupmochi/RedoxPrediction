#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:36:17 2022

@author: ljia
"""
import os

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import warnings

# tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# tf.config.run_functions_eagerly(True) # @todo: this is for debug only.
# tf.config.experimental_run_functions_eagerly(True)


#%%
# =============================================================================
# Create a tf.data.Dataset.
# =============================================================================

def prepare_batch(x_batch, y_batch):
	"""Merges (sub)graphs of batch into a single global (disconnected) graph
	"""

	atom_features, bond_features, pair_indices = x_batch

	# Obtain number of atoms and bonds for each graph (molecule)
	num_atoms = atom_features.row_lengths()
	num_bonds = bond_features.row_lengths()

	# Obtain partition indices (molecule_indicator), which will be used to
	# gather (sub)graphs from global graph in model later on
	molecule_indices = tf.range(len(num_atoms))
	molecule_indicator = tf.repeat(molecule_indices, num_atoms)

	# Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
	# 'pair_indices' (and merging ragged tensors) actualizes the global graph
	gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
	increment = tf.cumsum(num_atoms[:-1])
	increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
	pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
	pair_indices = pair_indices + increment[:, tf.newaxis]
	atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
	bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

	return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
# 	return prepare_batch(X, y)
	dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
	if shuffle:
		dataset = dataset.shuffle(1024)
	return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)
#	 return dataset.batch(batch_size).map(
# 		(lambda x: tf.py_function(prepare_batch, [x], [tf.int64])), -1)


#%%
# =============================================================================
# Model
# =============================================================================


class EdgeNetwork(layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	def build(self, input_shape):
		self.atom_dim = input_shape[0][-1]
		self.bond_dim = input_shape[1][-1]
		self.kernel = self.add_weight(
			shape=(self.bond_dim, self.atom_dim * self.atom_dim),
			trainable=True,
			initializer="glorot_uniform",
			name="kernel",
		)
		self.bias = self.add_weight(
			shape=(self.atom_dim * self.atom_dim), trainable=True, initializer="zeros", name="bias",
		)
		self.built = True


	def call(self, inputs):
		atom_features, bond_features, pair_indices = inputs

		# Apply linear transformation to bond features
		bond_features = tf.matmul(bond_features, self.kernel) + self.bias

		# Reshape for neighborhood aggregation later
		bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

		# Obtain atom features of neighbors
		atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
		atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

		# Apply neighborhood aggregation
		transformed_features = tf.matmul(bond_features, atom_features_neighbors)
		transformed_features = tf.squeeze(transformed_features, axis=-1)
		aggregated_features = tf.math.unsorted_segment_sum(
			transformed_features,
			pair_indices[:, 0],
			num_segments=tf.shape(atom_features)[0],
		)
		return aggregated_features


class MessagePassing(layers.Layer):
	def __init__(self, units, steps=4, **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.steps = steps


	def build(self, input_shape):
		self.atom_dim = input_shape[0][-1]
		self.message_step = EdgeNetwork()
		self.pad_length = max(0, self.units - self.atom_dim)
		self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
		self.built = True


	def call(self, inputs):
		atom_features, bond_features, pair_indices = inputs

		# Pad atom features if number of desired units exceeds atom_features dim.
		# Alternatively, a dense layer could be used here.
		atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

		# Perform a number of steps of message passing
		for i in range(self.steps):
			# Aggregate information from neighbors
			atom_features_aggregated = self.message_step(
				[atom_features_updated, bond_features, pair_indices]
			)

			# Update node state via a step of GRU
			atom_features_updated, _ = self.update_step(
				atom_features_aggregated, atom_features_updated
			)
		return atom_features_updated


	def get_config(self):
		config = super().get_config()
		config.update({
			'units': self.units,
			'steps': self.steps,
		})
		return config

# =============================================================================
# Readout.
# =============================================================================

class PartitionPadding(layers.Layer):
	def __init__(self, batch_size, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size


	def call(self, inputs):

		atom_features, molecule_indicator = inputs

		# Obtain subgraphs
		atom_features_partitioned = tf.dynamic_partition(
			atom_features, molecule_indicator, self.batch_size
		)

		# Pad and stack subgraphs
		num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
		max_num_atoms = tf.reduce_max(num_atoms)
		atom_features_stacked = tf.stack(
			[
				tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
				for f, n in zip(atom_features_partitioned, num_atoms)
			],
			axis=0,
		)

		# Remove empty subgraphs (usually for last batch in dataset)
		gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
		gather_indices = tf.squeeze(gather_indices, axis=-1)
		return tf.gather(atom_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(layers.Layer):
	def __init__(
		self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
	):
		super().__init__(**kwargs)
		self.num_heads = num_heads
		self.embed_dim = embed_dim
		self.dense_dim = dense_dim
		self.batch_size = batch_size

		self.partition_padding = PartitionPadding(batch_size)
		self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
		self.dense_proj = keras.Sequential(
			[layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
		)
		self.layernorm_1 = layers.LayerNormalization()
		self.layernorm_2 = layers.LayerNormalization()
		self.average_pooling = layers.GlobalAveragePooling1D()


	def call(self, inputs):
		x = self.partition_padding(inputs)
		padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
		padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
		attention_output = self.attention(x, x, attention_mask=padding_mask)
		proj_input = self.layernorm_1(x + attention_output)
		proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
		return self.average_pooling(proj_output)


	def get_config(self):
		config = super().get_config()
		config.update({
			'num_heads': self.num_heads,
			'embed_dim': self.embed_dim,
			'dense_dim': self.dense_dim,
			'batch_size': self.batch_size,
		})
		return config


# =============================================================================
# Message Passing Neural Network (MPNN)
# =============================================================================


# class MPNNTransformerModel(tf.keras.Model):
# 	def __init__(self,
#  			  atom_dim,
#  			  bond_dim,
#  			  batch_size: int = 32,
#  			  message_units: int = 64,
#  			  message_steps: int = 4,
#  			  num_attention_heads: int = 8,
#  			  dense_units: int = 512,
# mode: str = 'regression',
#  			  ):
# 		super(MPNNTransformerModel, self).__init__()

# 		atom_features = layers.Input((atom_dim), dtype='float32', name='atom_features')
# 		bond_features = layers.Input((bond_dim), dtype='float32', name='bond_features')
# 		pair_indices = layers.Input((2), dtype='int32', name='pair_indices')
# 		molecule_indicator = layers.Input((), dtype='int32', name='molecule_indicator')

# 		x = MessagePassing(message_units, message_steps)(
#  			[atom_features, bond_features, pair_indices]
# 		)

# 		x = TransformerEncoderReadout(
#  	        num_attention_heads,
#  			message_units,
#  			dense_units,
#  			batch_size
#  	    )([x, molecule_indicator])

# 		x = layers.Dense(dense_units, activation='relu')(x)
# activation_out = ('linear' if mode == 'regression' else 'sigmoid')
# 		x = layers.Dense(1, activation=activation_out)(x)

# 		self.model = keras.Model(
#  			inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
#  			outputs=[x],
#  			)


# 	def call(self, inputs):
# 		x = self.model(inputs)
# 		return x


def MPNNTransformerModel(
		atom_dim,
		bond_dim,
		batch_size: int = 32,
		message_units: int = 64,
		message_steps: int = 4,
		num_attention_heads: int = 8,
		dense_units: int = 512,
		mode: str = 'regression',
		):

	atom_features = layers.Input((atom_dim), dtype='float32', name='atom_features')
	bond_features = layers.Input((bond_dim), dtype='float32', name='bond_features')
	pair_indices = layers.Input((2), dtype='int32', name='pair_indices')
	molecule_indicator = layers.Input((), dtype='int32', name='molecule_indicator')

	x = MessagePassing(message_units, message_steps)(
		[atom_features, bond_features, pair_indices]
	)

	x = TransformerEncoderReadout(
        num_attention_heads,
		message_units,
		dense_units,
		batch_size
    )([x, molecule_indicator])

	x = layers.Dense(dense_units, activation='relu')(x)
	activation_out = ('linear' if mode == 'regression' else 'sigmoid')
	x = layers.Dense(1, activation=activation_out)(x)

	model = keras.Model(
		inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
		outputs=[x],
		)

	return model