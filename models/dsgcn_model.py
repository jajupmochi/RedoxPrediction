#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:25:30 2022

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

from models.utils import Identity
from models.readout import GlobalAverageMaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D

# tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# tf.config.run_functions_eagerly(True) # @todo: this is for debug only.
# tf.config.experimental_run_functions_eagerly(True)


#%%
# =============================================================================
# Create a tf.data.Dataset.
# =============================================================================


def convolution_supports(adj_matrix_dense, V, U, kernel_type='custom'):
	supports = []
	if kernel_type == 'cheb':
		chebnet = chebyshev_polynomials(A[i], nkernel-1,True)
		for j in range(0,nkernel):
			supports[i,j,0:n,0:n]=chebnet[j].toarray()
	elif kernel_type == 'gcn':
		supports[i,0,0:n,0:n]= (normalize_adj(A[i] + tf.shape(sp.eye(A[i])[0]))).toarray()
	else:
		v_max = tf.reduce_max(V)
		supports.append(tf.linalg.eye(tf.shape(adj_matrix_dense)[0])) # identical
		supports.append(tf.matmul( #
			tf.matmul(U, tf.linalg.diag(
				tf.math.exp(-1 * (V - 0.) ** 2))), U, transpose_b=True))
		supports.append(tf.matmul(
			tf.matmul(U, tf.linalg.diag(
				tf.math.exp(-1 * (V - 0.5 * v_max) ** 2))), U, transpose_b=True))
		supports.append(tf.matmul(
			tf.matmul(U, tf.linalg.diag(
				tf.math.exp(-1 * (V - v_max) ** 2))), U, transpose_b=True))
		supports = tf.stack(supports)
		return supports


def eigen_decomposition(adj_matrix_dense):
	degrees = tf.reduce_sum(adj_matrix_dense, axis=0)
	degrees = tf.math.rsqrt(tf.cast(degrees, dtype=tf.float32)) # Normalize.
	degrees = tf.where(tf.math.is_nan(degrees), tf.zeros_like(degrees), degrees) # Set all NaN to 0.
	degrees = tf.where(tf.math.is_inf(degrees), tf.zeros_like(degrees), degrees) # Set all Inf to 0.
	D_matrix = tf.linalg.diag(degrees)
	L_matrix = tf.eye(tf.shape(D_matrix)[0]) - tf.matmul(
		tf.matmul(D_matrix, tf.cast(
			adj_matrix_dense, dtype=tf.float32), transpose_a=True), D_matrix) # normalized L = I - (D^(-1/2))^TWD^(-1/2)
	V, U = tf.linalg.eigh(L_matrix) # decomposition
	V = tf.where(tf.math.less(V, 0), tf.zeros_like(V), V)
	return V, U


def prepare_batch(x_batch, y_batch, kernel_type='custom'):
	"""Merges (sub)graphs of batch into a single global (disconnected) graph.

	The eigen decomposition and convolution supports (filters) are computed here
	a piori.
	"""
	node_features, edge_features, pair_indices = x_batch

	# Obtain number of nodes and edges for each graph (graph)
	num_nodes = node_features.row_lengths()
	num_edges = edge_features.row_lengths()

	# Obtain partition indices (graph_indicator), which will be used to
	# gather (sub)graphs from global graph in model later on
	graph_indices = tf.range(len(num_nodes))
	graph_indicator = tf.repeat(graph_indices, num_nodes)

	# Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
	# 'pair_indices' (and merging ragged tensors) actualizes the global graph
	gather_indices = tf.repeat(graph_indices[:-1], num_edges[1:])
	increment = tf.cumsum(num_nodes[:-1])
	increment = tf.pad(tf.gather(increment, gather_indices), [(num_edges[0], 0)])
	pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
	pair_indices = pair_indices + increment[:, tf.newaxis]
	node_features = node_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
	edge_features = edge_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

	# Construct adjacency matrix.
	adj_matrix = tf.sparse.SparseTensor(
		indices=pair_indices,
		values=tf.ones(tf.shape(pair_indices)[0]),
		dense_shape=tf.cast(tf.stack(
			(tf.shape(node_features)[0], tf.shape(node_features)[0])),
			dtype=tf.int64),
		)
	adj_matrix_dense = tf.sparse.to_dense(tf.sparse.reorder(adj_matrix)) # to dense.

	# Eigen decomposition.
	V, U = eigen_decomposition(adj_matrix_dense)
	supports = convolution_supports(adj_matrix_dense, V, U, kernel_type=kernel_type)

	return (node_features, edge_features, pair_indices, supports, graph_indicator), y_batch


def DSGCNDataset(X, y, batch_size=32, shuffle=False):
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


class DepSepGraphConv(layers.Layer):
	def __init__(
			self,
			in_feats: int = 32,
			out_feats: int = 32,
			is_depthwise: bool = False,
			feat_drop: float = 0.,
			kernel_drop: float = 0.,
			bias: bool = True,
			weight_decay: float = 0.,
			residual: bool = False,
			activation: object = layers.ReLU(),
			**kwargs
			):
		super().__init__(**kwargs)
		self.in_feats = in_feats
		self.out_feats = out_feats
		self.is_depthwise = is_depthwise
		self.feat_drop = layers.Dropout(rate=feat_drop)
		self.kernel_drop = layers.Dropout(rate=kernel_drop)
		self.bias = bias
		self.weight_decay = weight_decay
		self.activation = activation
		if residual:
			if in_feats != out_feats:
				xinit = tf.keras.initializers.VarianceScaling(
					scale=np.sqrt(2), mode="fan_avg",
					distribution="untruncated_normal")
				self.res_fc = layers.Dense(
					out_feats, use_bias=False, kernel_initializer=xinit)
			else:
				self.res_fc = Identity()
		else:
			self.res_fc = None
# 		self.weights_ = [] # trainable weights


	def build(self, input_shape):
		self.nb_supports = input_shape[1][0]

		# Initialize weight for each filter.
		if self.nb_supports > 1 and self.is_depthwise:
			i = 0
			self.weights_.append(self.add_weight(
				shape=(self.in_feats, self.out_feats),
				trainable=True,
				initializer='glorot_uniform',
				regularizer=(tf.keras.regularizers.L2(self.weight_decay) if self.weight_decay > 0. else None),
				name='weights_' + str(i),
			))

			if self.firstDSWS:
				self.vars['sdweight_' + str(i)] = ones([input_dim],name='sdweight_' + str(i))
                    #self.vars['sdweight_' + str(i)] = glorot([input_dim,1],name='sdweight_' + str(i))
                    #self.vars['sdweight_' + str(i)]=tf.squeeze(self.vars['sdweight_' + str(i)])

			for i in range(1,self.support.shape[1]):
				self.vars['sdweight_' + str(i)] = zeros([input_dim],name='sdweight_' + str(i))
                    #self.vars['sdweight_' + str(i)] = glorot([input_dim,1],name='sdweight_' + str(i))
                    #self.vars['sdweight_' + str(i)]=tf.squeeze(self.vars['sdweight_' + str(i)])

		if not self.is_depthwise:
			self.supp_weights = self.add_weight(
				shape=(self.nb_supports, self.in_feats, self.out_feats),
				trainable=True,
				initializer='glorot_uniform',
				regularizer=(tf.keras.regularizers.L2(self.weight_decay) if self.weight_decay > 0. else None),
				name='supp_weights',
			)
# 			for i in range(1, self.nb_supports): # for each filter.
# 				self.weights_.append(self.add_weight(
# 					shape=(self.in_feats, self.out_feats),
# 					trainable=True,
# 					initializer='glorot_uniform',
# 					regularizer=(tf.keras.regularizers.L2(self.weight_decay) if self.weight_decay > 0. else None),
# 					name='weights_' + str(i),
# 				))
# 		self.weights_ = tf.stack(self.weights_)

		# bias
		if self.bias:
			self.bias_fn = self.add_weight(
				shape=(self.out_feats), trainable=True, initializer='zeros', name='bias',
			)
		else:
			self.bias_fn = None

		self.built = True


	def call(self, inputs):
		node_features, supports = inputs

		# input dropout
		node_features = self.feat_drop(node_features)

		# Convolve.
		if self.is_depthwise:

			supports = list()

			for i in range(0,self.support.shape[1]):
				if self.isdropout[1]:
					tmp=tf.nn.dropout(self.support[:,i,:,:], 1-self.dropout)
					s0=tf.matmul(tmp,x)
				else:
					s0=tf.matmul(self.support[:,i,:,:],x)

				if self.support.shape[1]>1 and (i>0 or self.firstDSWS):
					s0=s0*self.vars['sdweight_'+str(i)]
				supports.append(s0)

			output = tf.add_n(supports)
			output=tf.tensordot(output,self.vars['weights_' + str(0)],[2, 0])

		else:
# 			i = tf.constant(0)
# 			while_condition = lambda x: tf.less(x, tf.shape(supports)[0])
# 			def while_body(i):
# 				feat_supp = self.kernel_drop(supports[i])
# 				feat_supp = tf.matmul(feat_supp, node_features)
# 				feat_supp = tf.matmul(feat_supp, self.vars['weights_' + str(i)])
# 				feat_supports.append(feat_supp)
# 				return
# 			tf.while_loop(while_condition, while_body, [i])

# 			feat_supports = list()
# 			for i in tf.range(0, tf.shape(supports)[0]):
# 				feat_supp = self.kernel_drop(supports[i])
# 				feat_supp = tf.matmul(feat_supp, node_features)
# 				feat_supp = tf.matmul(feat_supp, self.vars['weights_' + str(i)])
# 				feat_supports.append(feat_supp)
# 			feat_supports = tf.map_fn(
# 				lambda x: self.compute_support(node_features, supports, x),
# # 				lambda x: self.kernel_drop(supports[x]),
# 				tf.range(0, tf.shape(supports)[0]),
# 				fn_output_signature=tf.float32,
# 				)
			feat_supports = self.kernel_drop(supports)
			feat_supports = tf.matmul(feat_supports, node_features)
			feat_supports = tf.matmul(feat_supports, self.supp_weights)
			rst = tf.math.reduce_sum(feat_supports, axis=0)
# 			feat_supports = tf.map_fn(
# 				lambda x: self.compute_support(node_features, x[0], x[1]),
# # 				lambda x: self.kernel_drop(supports[x]),
# 				(supports, self.supp_weights),
# 				fn_output_signature=(tf.float32, tf.float32),
# 				)
# 			rst = tf.math.add_n(feat_supports)

		# bias
		if self.bias_fn is not None:
			 rst = rst + self.bias_fn

 		# residual
		if self.res_fc is not None:
			resval = self.res_fc(rst)
			rst = rst + resval

		# avtivation
		if self.activation is not None:
			rst = self.activation(rst)

		return rst


# 	def compute_support(self, node_features, support, weights):
# 		feat_supp = self.kernel_drop(support)
# 		feat_supp = tf.matmul(feat_supp, node_features)
# 		feat_supp = tf.matmul(feat_supp, weights)
# 		return feat_supp


class MessagePassing(layers.Layer):
	def __init__(
			self,
			in_feats: int = 32,
			hidden_feats: int = 32,
			message_steps: int = 1,
			is_depthwise: bool = False,
			feat_drop: float = 0.,
			kernel_drop: float = 0.,
			bias: bool = True,
			weight_decay: float = 0.,
			residual: bool = False,
			agg_activation: str = 'relu',
			**kwargs
			):
		super().__init__(**kwargs)
		self.message_steps = message_steps
		self.graph_conv = []
		if agg_activation == 'relu':
			activation_fun = layers.ReLU()
		for i in range(message_steps):
			self.graph_conv.append(DepSepGraphConv(
				in_feats,
				hidden_feats,
				is_depthwise=is_depthwise,
				feat_drop=feat_drop,
				kernel_drop=kernel_drop,
				bias=bias,
				weight_decay=weight_decay,
				residual=residual,
				activation=activation_fun)
				)
			in_feats = hidden_feats


# 	def build(self, input_shape):
# 		self.node_dim = input_shape[0][-1]
# 		self.message_step = EdgeNetwork()
# 		self.pad_length = max(0, self.units - self.node_dim)
# 		self.update_step = layers.GRUCell(self.node_dim + self.pad_length)
# 		self.built = True


	def call(self, inputs):
		feats, supports = inputs

		# Perform a number of steps of message passing
		for graph_conv in self.graph_conv:
			# Update node features using a GCN layer.
			feats = graph_conv([feats, supports])

		return feats


	def get_config(self):
		config = super().get_config()
		config.update({
			'message_steps': self.message_steps,
			'graph_conv': self.graph_conv,
		})
		return config

# =============================================================================
# Readout.
# =============================================================================


class GraphPartition(layers.Layer):
	def __init__(self, batch_size, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size


	def call(self, inputs):

		node_features, graph_indicator = inputs

		# Obtain subgraphs
		node_features_partitioned = tf.dynamic_partition(
			node_features, graph_indicator, self.batch_size
		)

		# Remove empty subgraphs (usually for last batch in dataset)
		num_nodes = [tf.shape(f)[0] for f in node_features_partitioned]
		masks = tf.cast(num_nodes, tf.bool)
		node_features_partitioned = tf.ragged.boolean_mask(
			tf.ragged.stack(node_features_partitioned), masks)

# 		gather_indices = tf.where(~tf.math.equal(num_nodes, 0))
# 		gather_indices = tf.squeeze(gather_indices, axis=-1)
# 		masks = tf.

# 		masks = tf.map_fn(
# 			lambda i: (1 if tf.cond(tf.math.equal(tf.shape(i)[0], 0), lambda: 0, lambda: 1) else 0),
# 			node_features_partitioned)

		return node_features_partitioned


# class TransformerEncoderReadout(layers.Layer):
# 	def __init__(
# 		self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
# 	):
# 		super().__init__(**kwargs)
# 		self.num_heads = num_heads
# 		self.embed_dim = embed_dim
# 		self.dense_dim = dense_dim
# 		self.batch_size = batch_size

# 		self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
# 		self.dense_proj = keras.Sequential(
# 			[layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
# 		)
# 		self.layernorm_1 = layers.LayerNormalization()
# 		self.layernorm_2 = layers.LayerNormalization()
# 		self.average_pooling = layers.GlobalAveragePooling1D()


# 	def call(self, inputs):
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
# Deepwise Separable GCN (DSGCN)
# =============================================================================

class DSGCNModel(tf.keras.Model):
	def __init__(self,
			# The following are used by the DSGCN layer.
			in_feats: int = 32,
			hidden_feats: int = 32,
			message_steps: int = 1,
			is_depthwise: bool = False,
			feat_drop: float = 0.,
			kernel_drop: float = 0.,
			bias: bool = True,
			weight_decay: float = 0.,
			residual: bool = False,
			# The following are used for aggragation of the outputs.
			agg_activation: str = 'relu',
			# The following are used for readout.
			batch_size: int = 32, # for (sub-)graph partition and transformer readou.
			readout: str = 'mean',
			# The following are used for the final prediction.
			predictor_hidden_feats: int = 512,
			predictor_feat_drop: float = 0.,
# 			predictor_kernel_drop: float = 0.,
			predictor_bias: bool = True,
			predictor_weight_decay: float = 0.,
# 			predictor_residual: bool = False,
			predictor_activation: str = 'relu',
			mode: str = 'regression',
			):
		super(DSGCNModel, self).__init__()

		# Message passing
		self.msg_passing = MessagePassing(
			in_feats=in_feats,
			hidden_feats=hidden_feats,
			message_steps=message_steps,
			is_depthwise=is_depthwise,
			feat_drop=feat_drop,
			kernel_drop=kernel_drop,
			bias=bias,
			weight_decay=weight_decay,
			residual=residual,
			agg_activation=agg_activation,
			)

		# Readout
		self.graph_partition = GraphPartition(batch_size)
		if readout == 'mean':
			self.readout = GlobalAveragePooling1D()
		elif readout == 'max':
			self.readout = GlobalMaxPooling1D()
		elif readout == 'meanmax':
			self.readout = GlobalAverageMaxPooling1D()

		# Predict.
		if predictor_feat_drop > 0.:
			self.pred_fdrop_fc = layers.Dropout(rate=predictor_feat_drop)
		else:
			self.pred_fdrop_fc = None

		self.dense1 = layers.Dense(
			predictor_hidden_feats,
			activation=predictor_activation,
			use_bias=predictor_bias,
			kernel_regularizer=(tf.keras.regularizers.L2(predictor_weight_decay) if predictor_weight_decay > 0. else None),
			)
		activation_out = ('linear' if mode == 'regression' else 'sigmoid')
		self.dense2 = layers.Dense(1, activation=activation_out)


	def call(self, inputs):
		node_features, edge_features, pair_indices, supports, graph_indicator = inputs
		x = self.msg_passing([node_features, supports])
		x = self.graph_partition([x, graph_indicator])
		x = self.readout(x)
		if self.pred_fdrop_fc is not None:
			x = self.pred_fdrop_fc(x)
		x = self.dense1(x)
		x = self.dense2(x)
		return x


# 	def build(self, input_shape):
# 		super(GCNModel, self).build(input_shape)


# def MPNNTransformerModel(
# 		node_dim,
# 		edge_dim,
# 		batch_size: int = 32,
# 		message_units: int = 64,
# 		message_steps: int = 4,
# 		num_attention_heads: int = 8,
# 		dense_units: int = 512,
# 		mode: str = 'regression',
# 		):

# 	node_features = layers.Input((node_dim), dtype='float32', name='node_features')
# 	edge_features = layers.Input((edge_dim), dtype='float32', name='edge_features')
# 	pair_indices = layers.Input((2), dtype='int32', name='pair_indices')
# 	graph_indicator = layers.Input((), dtype='int32', name='graph_indicator')

# 	x = MessagePassing(message_units, message_steps)(
# 		[node_features, edge_features, pair_indices]
# 	)

# 	x = TransformerEncoderReadout(
#         num_attention_heads,
# 		message_units,
# 		dense_units,
# 		batch_size
#     )([x, graph_indicator])

# 	x = layers.Dense(dense_units, activation='relu')(x)
# 	activation_out = ('linear' if mode == 'regression' else 'sigmoid')
# 	x = layers.Dense(1, activation=activation_out)(x)

# 	model = keras.Model(
# 		inputs=[node_features, edge_features, pair_indices, graph_indicator],
# 		outputs=[x],
# 		)

# 	return model