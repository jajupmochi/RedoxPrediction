#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:25:50 2022

@author: ljia
"""
import os

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from dgl.nn.tensorflow.conv import GraphConv
from dgl.nn.tensorflow.glob import AvgPooling
# import numpy as np
import warnings

# try:
import dgl
# except ModuleNotFoundError:
# 	raise ImportError('This function requires DGL to be installed.')

# from dataset import convert, batch

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

	node_features, edge_features, pair_indices = x_batch

	# Obtain number of atoms and bonds for each graph (molecule)
	num_nodes = node_features.row_lengths()
	num_edges = pair_indices.row_lengths() # edge_features may be empty.

	# Obtain partition indices (molecule_indicator), which will be used to
	# gather (sub)graphs from global graph in model later on
	molecule_indices = tf.range(len(num_nodes))
# 	molecule_indicator = tf.repeat(molecule_indices, num_nodes)

	# Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
	# 'pair_indices' (and merging ragged tensors) actualizes the global graph
	gather_indices = tf.repeat(molecule_indices[:-1], num_edges[1:])
	increment = tf.cumsum(num_nodes[:-1])
	increment = tf.pad(tf.gather(increment, gather_indices), [(num_edges[0], 0)])
	pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
	pair_indices = pair_indices + increment[:, tf.newaxis]
	pair_indices = tf.transpose(pair_indices) # Transpose so to facilicate creating DGL graph.
	pair_indices = (pair_indices[0, :], pair_indices[1, :])
	node_features = node_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
# 	edge_features = edge_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
	edge_features = edge_features.to_tensor()

	return (node_features, edge_features, pair_indices, num_nodes, num_edges), y_batch


def GCNDataset(X, y, batch_size=32, shuffle=False):
# 	return prepare_batch(X, y)
	dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
	if shuffle:
		dataset = dataset.shuffle(1024)
	return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)
#	 return dataset.batch(batch_size).map(
# 		(lambda x: tf.py_function(prepare_batch, [x], [tf.int64])), -1)


#%%
# =============================================================================
# Message passing model
# =============================================================================


class MessagePassing(layers.Layer):
	def __init__(self,
	  		in_feats: int = 32,
			hidden_feats: int = 32,
			message_steps: int = 1,
			norm: str = 'both',
			weight: bool = True,
			bias: bool = True,
# 			feat_drop: float = 0.,
# 			residual: bool = False,
			agg_activation: str = None,
			**kwargs):
		super().__init__(**kwargs)
		self.message_steps = message_steps
# 		self.agg_activation = agg_activation
		self.gcn_conv = []
		if agg_activation == 'relu':
			activation_fun = layers.ReLU()
		for i in range(message_steps):
			self.gcn_conv.append(GraphConv(
				in_feats,
				hidden_feats,
				norm=norm,
				weight=weight,
				bias=bias,
# 				feat_drop=feat_drop,
# 				residual=residual,
				activation=activation_fun)
				)
			in_feats = hidden_feats

# 	def build(self, input_shape):
# 		self.atom_dim = input_shape[0][-1]
# 		self.message_step = EdgeNetwork()
# 		self.pad_length = max(0, self.units - self.atom_dim)
# 		self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
# 		self.built = True


	def call(self, inputs):
		graph, feat = inputs

		# Perform a number of steps of message passing
		for gcn_conv in self.gcn_conv:
			# Update node features using a GCN layer.
			feat = gcn_conv(graph, feat)

		return feat


	def get_config(self):
		config = super().get_config()
		config.update({
			'message_steps': self.message_steps,
# 			'agg_activation': self.agg_activation,
			'gcn_conv': self.gcn_conv,
		})
		return config

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
# 		self.num_heads = num_heads
# 		self.embed_dim = embed_dim
# 		self.dense_dim = dense_dim
# 		self.batch_size = batch_size

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


# 	def get_config(self):
# 		config = super().get_config()
# 		config.update({
# 			'num_heads': self.num_heads,
# 			'embed_dim': self.embed_dim,
# 			'dense_dim': self.dense_dim,
# 			'batch_size': self.batch_size,
# 		})
# 		return config


# =============================================================================
# Graph convolutional network (GCN)
# =============================================================================


class GCNModel(tf.keras.Model):
	def __init__(self,
			# The following are used by the GCNConv layer.
	  		in_feats: int = 32,
			hidden_feats: int = 32,
			message_steps: int = 1,
			norm: str = 'both',
			weight: bool = True,
			bias: bool = True,
# 			feat_drop: float = 0.,
# 			residual: bool = False,
			# The following are used for aggragation of the outputs.
			agg_activation: str = 'relu',
			# The following are used for readout.
			readout: str = 'mean',
			batch_size: int = 32, # for transformer readout
			# The following are used for the final prediction.
			predictor_hidden_feats: int = 512,
			predictor_activation: str = 'relu',
			mode: str = 'regression',
			):
		super(GCNModel, self).__init__()

		# Message passing
		self.msg_passing = MessagePassing(
			in_feats=in_feats,
			hidden_feats=hidden_feats,
			message_steps=message_steps,
			norm=norm,
			weight=weight,
			bias=bias,
# 			feat_drop=feat_drop,
# 			residual=residual,
			agg_activation=agg_activation,
			)

		# Readout
		if readout == 'mean':
			self.readout = AvgPooling()

		# Predict.
		self.dense1 = layers.Dense(predictor_hidden_feats, activation=predictor_activation)
		activation_out = ('linear' if mode == 'regression' else 'sigmoid')
		self.dense2 = layers.Dense(1, activation=activation_out)


	def call(self, inputs):
		# @todo: a more efficient way. The current implementation reconstruct the DGL graph of each (sub-)graph inside every batch, which is time consuming.
		graphs, node_features = self.to_dgl_graph(inputs)
		x = self.msg_passing((graphs, node_features))
		x = self.readout(graphs, x)
		x = self.dense1(x)
		x = self.dense2(x)
		return x


# 	def build(self, input_shape):
# 		super(GCNModel, self).build(input_shape)


	def to_dgl_graph(self, inputs):
		"""Construct a DGL batched graph from inputs (batch_size sub-graphs).
		"""
		node_features, edge_features, pair_indices, num_nodes, num_edges = inputs

		# Construct DGL graphs.
		g = dgl.graph(pair_indices)
		g.ndata['x'] = node_features

		# Set batch information.
		g.set_batch_num_nodes(num_nodes)
		g.set_batch_num_edges(num_edges)

		# 	edge_features = edge_features

		return g, node_features



# def GCNModel(
# 		node_dim,
# 		# The following are used by the GCNConv layer.
#   		in_feats: int = 32,
# 		hidden_feats: int = 32,
# 		message_steps: int = 1,
# 		norm: str = 'both',
# 		weight: bool = True,
# 		bias: bool = True,
# # 			feat_drop: float = 0.,
# # 			residual: bool = False,
# 		# The following are used for aggragation of the outputs.
# 		agg_activation: str = None,
# 		# The following are used for readout.
# 		readout: str = 'mean',
# 		batch_size: int = 32, # for transformer readout
# 		# The following are used for the final prediction.
# 		predictor_hidden_feats: int = 512,
# 		predictor_activation: str = 'relu',
# 		mode: str = 'regression',
# 		):

# 	graphs = layers.Input((1), dtype=dgl.graph, name='dgl_graphs')
# 	node_features = layers.Input((node_dim), dtype='float32', name='node_features')
# # 	pair_indices = layers.Input((2), dtype='int32', name='pair_indices')
# # 	molecule_indicator = layers.Input((), dtype='int32', name='molecule_indicator')

# 	# Message passing
# 	x = MessagePassing(
# 		in_feats=in_feats,
# 		hidden_feats=hidden_feats,
# 		message_steps=message_steps,
# 		norm=norm,
# 		weight=weight,
# 		bias=bias,
# # 			feat_drop=feat_drop,
# # 			residual=residual,
# 		agg_activation=agg_activation,
# 		)(
# 		[graphs, node_features]
# 	)

# 	# Readout
# 	if readout == 'mean':
# 		x = AvgPooling()(graphs, x)

# 	# Predict.
# 	x = layers.Dense(predictor_hidden_feats, activation=predictor_activation)(x)
# 	activation_out = ('linear' if mode == 'regression' else 'sigmoid')
# 	x = layers.Dense(1, activation=activation_out)(x)

# 	model = keras.Model(
# 		inputs=[graphs, node_features],
# 		outputs=[x],
# 		)

# 	return model