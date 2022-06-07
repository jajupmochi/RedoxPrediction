#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:48:19 2022

@author: ljia
"""
import os

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from dgl.nn.tensorflow.conv import GATConv
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


def GATDataset(X, y, batch_size=32, shuffle=False):
# 	return prepare_batch(X, y)
	dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
	if shuffle:
		dataset = dataset.shuffle(1024)
	return dataset.batch(batch_size).prefetch(-1)
#	 return dataset.batch(batch_size).map(
# 		(lambda x: tf.py_function(prepare_batch, [x], [tf.int64])), -1)


# def prepare_batch(x_batch, y_batch):
# 	"""Merges (sub)graphs of batch into a single global (disconnected) graph.
# 	"""

# # 	atom_features, bond_features, pair_indices = x_batch

# # 	# Construct DGL graphs.
# # 	graphs = []
# # 	for i, a_feats in enumerate(atom_features):
# # 		graphs.append(to_dgl_graph((a_feats, bond_features[i], pair_indices[i])))

# 	# Batch DGL graphs.
# 	bgraphs = batch.batch(x_batch)
# 	atom_features = bgraphs.ndata['x']
# # 	bond_features = bgraphs.edata['x']

# # 	return (bgraphs, atom_features, bond_features), y_batch
# 	return (bgraphs, atom_features), y_batch


# def GATDataset(X, y, batch_size=32, shuffle=False):
# 	# Construct DGL graphs.
# 	atom_features, bond_features, pair_indices = X
# 	graphs = [to_dgl_graph((a_feats, bond_features[i], pair_indices[i]))
# 		   for i, a_feats in enumerate(atom_features)]

# 	for i_graph in range(0, len(graphs), batch_size):
# 		yield prepare_batch(
# 			graphs[i_graph:i_graph+batch_size], y[i_graph:i_graph+batch_size])

# # 	hahaha = []
# # 	for i_graph in range(0, len(graphs), batch_size):
# # 		hahaha.append(prepare_batch(
# # 			graphs[i_graph:i_graph+batch_size], y[i_graph:i_graph+batch_size]))
# # 	return hahaha


#%%
# =============================================================================
# Message passing model
# =============================================================================


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


class MessagePassing(layers.Layer):
	def __init__(self,
	  		in_feats: int = 32,
			hidden_feats: int = 32,
			message_steps: int = 1,
			num_attention_heads: int = 8,
			feat_drop: float = 0.,
			attn_drop: float = 0.,
			negative_slope: float = 0.2, # for LeakyReLU.
			residual: bool = False,
			bias: bool = False,
			agg_activation: str = None,
			attn_agg_mode: str = 'concat',
			**kwargs):
		super().__init__(**kwargs)
		self.message_steps = message_steps
		self.attn_agg_mode = attn_agg_mode
		self.agg_activation = agg_activation
		self.gat_convs = []
		for i in range(message_steps):
			self.gat_convs.append(GATConv(
				in_feats,
				hidden_feats,
				num_attention_heads,
				feat_drop=feat_drop,
				attn_drop=attn_drop,
				negative_slope=negative_slope,
				residual=residual,
# 				bias=bias,
				activation=None)
				)
			if attn_agg_mode == 'concat':
				in_feats = hidden_feats * num_attention_heads
			else:
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
		for gat_conv in self.gat_convs:
			# Update node features using a GAT layer.
			feat = gat_conv(graph, feat)
			# Aggragate multi-head attention outputs.
			if self.attn_agg_mode == 'concat':
				feat = tf.keras.layers.Flatten()(feat)
			else:
				feat = tf.reduce_mean(feat, axis=1)
			# Add activation to the aggregated multi-head results if asked.
			if self.agg_activation is not None:
				feat = self.agg_activation(feat)

		return feat


	def get_config(self):
		config = super().get_config()
		config.update({
			'message_steps': self.message_steps,
			'attn_agg_mode': self.attn_agg_mode,
			'agg_activation': self.agg_activation,
			'gat_convs': self.gat_convs,
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
# Graph attention network (GAT)
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


class GATModel(tf.keras.Model):
	def __init__(self,
			node_dim, # @todo: Check paper for all the defaults.
			# The following are used by the GATConv layer.
			in_feats: int = 32,
			hidden_feats: int = 32,
			message_steps: int = 1,
			num_attention_heads: int = 8,
			feat_drop: float = 0.,
			attn_drop: float = 0.,
			negative_slope: float = 0.2, # for LeakyReLU
			residual: bool = False,
			bias: bool = False,
			# The following are used for aggragation of the multi-head outputs.
			agg_activation: str = None,
			attn_agg_mode: str = 'concat',
			# The following are used for readout.
			readout: str = 'mean',
			# The following are used for the final prediction.
			predictor_hidden_feats: int = 512,
			predictor_activation: str = 'relu',
			batch_size: int = 32, # for transformer readout
			mode: str = 'regression',
			):
		super(GATModel, self).__init__()

		# Message passing
		self.msg_passing = MessagePassing(
			in_feats=in_feats,
			hidden_feats=hidden_feats,
			message_steps=message_steps,
			num_attention_heads=num_attention_heads,
			feat_drop=feat_drop,
			attn_drop=attn_drop,
			negative_slope=negative_slope,
			residual=residual,
			bias=bias,
			agg_activation=agg_activation,
			attn_agg_mode=attn_agg_mode,
			)

		# Readout
		if readout == 'mean':
			self.readout = AvgPooling()

		# Predict.
		self.dense1 = layers.Dense(predictor_hidden_feats, activation=predictor_activation)
		activation_out = ('linear' if mode == 'regression' else 'sigmoid')
		self.dense2 = layers.Dense(1, activation=activation_out)


	def call(self, inputs):
		# @todo: a more efficient way. The current implementation recomputes the DGL graph of each (sub-)graph inside every batch, which is time consuming.
		graphs, node_features = self.to_dgl_graph(inputs)
		x = self.msg_passing((graphs, node_features))
		x = self.readout(graphs, x)
		x = self.dense1(x)
		x = self.dense2(x)
		return x


# 	def build(self, input_shape):
# 		super(GATModel, self).build(input_shape)


	def to_dgl_graph(self, inputs):
		"""Construct a DGL batched graph from inputs (batch_size sub-graphs).
		"""
		# Construct DGL graphs.
		node_features, edge_features, pair_indices = inputs
# 		graphs = tf.map_fn(self._to_dgl_graph, inputs)

		graphs = [self._to_dgl_graph((n_feats, edge_features[i], pair_indices[i]))
			for i, n_feats in enumerate(node_features)]

		# Batch DGL graphs.
		bgraphs = dgl.batch(graphs)
		node_features = bgraphs.ndata['x']
		# 	edge_features = bgraphs.edata['x']

		return bgraphs, node_features


	def _to_dgl_graph(self, inputs):
		"""Construct a DGL graph from inputs (of one sub-graph).
		"""
		node_features, edge_features, pair_indices = inputs

		srcs, dsts = [], []
		for i in range(pair_indices.shape[0]):
			srcs.append(pair_indices[i][0])
			dsts.append(pair_indices[i][1])
		srcs = tf.transpose(srcs)
		dsts = tf.transpose(dsts)
# 		srcs = tf.transpose([i[0] for i in pair_indices])
# 		dsts = tf.transpose([i[1] for i in pair_indices])
# 		srcs, dsts = tf.transpose(pair_indices.to_tensor())
		g = dgl.graph((srcs, dsts))
		g.ndata['x'] = node_features.to_tensor()
	# 	g.edata['x'] = edge_features.to_tensor()

		return g


# def GATModel(
# 		node_dim, # @todo: Check paper for all the defaults.
# 		# The following are used by the GATConv layer.
# 		in_feats: int = 32,
# 		hidden_feats: int = 32,
# 		message_steps: int = 1,
# 		num_attention_heads: int = 8,
# 		feat_drop: float = 0.,
# 		attn_drop: float = 0.,
# 		negative_slope: float = 0.2, # for LeakyReLU
# 		residual: bool = False,
# 		bias: bool = False,
# 		# The following are used for aggragation of the multi-head outputs.
# 		agg_activation: str = None,
# 		attn_agg_mode: str = 'concat',
# 		# The following are used for readout.
# 		readout: str = 'mean',
# 		# The following are used for the final prediction.
# 		predictor_hidden_feats: int = 512,
# 		predictor_activation: str = 'relu',
# 		batch_size: int = 32, # for transformer readout
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
# 		num_attention_heads=num_attention_heads,
# 		feat_drop=feat_drop,
# 		attn_drop=attn_drop,
# 		negative_slope=negative_slope,
# 		residual=residual,
# 		bias=bias,
# 		agg_activation=agg_activation,
# 		attn_agg_mode=attn_agg_mode,
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