#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:27:46 2022

@author: ljia
"""
try:
	from collections.abc import Sequence as SequenceCollection
except:
	from collections import Sequence as SequenceCollection

import deepchem as dc
import numpy as np
import tensorflow as tf

from typing import List, Union, Tuple, Iterable, Dict, Optional
from deepchem.utils.typing import OneOrMany, LossFn, ActivationFn
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, Loss
from deepchem.models.graph_models import TrimGraphOutput
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization


class _GraphConvKerasModelExt(tf.keras.Model):

	def __init__(self,
				 n_tasks,
				 graph_conv_layers,
				 dense_layer_size=128,
 				 activation_fns=None,
				 graph_pools=None,
				 dropout=0.0,
				 mode="classification",
				 number_atom_features=75,
				 n_classes=2,
				 batch_normalize=True,
				 uncertainty=False,
				 batch_size=100):
		"""An extended version of deepchem.models.graph_models._GraphConvKerasModelExt,
		which allows user-defined activation function and pooling method.

		The graph convolutions use a nonstandard control flow so the
		standard Keras functional API can't support them. We instead
		use the imperative "subclassing" API to implement the graph
		convolutions.

		All arguments have the same meaning as in GraphConvModel.
		"""
		super(_GraphConvKerasModelExt, self).__init__()
		if mode not in ['classification', 'regression']:
			raise ValueError("mode must be either 'classification' or 'regression'")

		self.mode = mode
		self.uncertainty = uncertainty

		if not isinstance(dropout, SequenceCollection):
			dropout = [dropout] * (len(graph_conv_layers) + 1)
		if len(dropout) != len(graph_conv_layers) + 1:
			raise ValueError('Wrong number of dropout probabilities provided')
		if uncertainty:
			if mode != "regression":
				raise ValueError("Uncertainty is only supported in regression mode")
			if any(d == 0.0 for d in dropout):
				raise ValueError(
						'Dropout must be included in every layer to predict uncertainty')

		# Set structural parameters (Linlin).
		n_layers = len(graph_conv_layers)
		if activation_fns is None:
			activation_fns = [tf.nn.relu for _ in range(n_layers)]
		if graph_pools is None:
			self.graph_pools = [layers.GraphPool() for _ in graph_conv_layers]
		else:
			self.graph_pools = graph_pools

		self.graph_convs = [
				layers.GraphConv(layer_size, activation_fn=activation_fns[i])
				for i, layer_size in enumerate(graph_conv_layers)
		]
		self.batch_norms = [
				BatchNormalization(fused=False) if batch_normalize else None
				for _ in range(len(graph_conv_layers) + 1)
		]
		self.dropouts = [
				Dropout(rate=rate) if rate > 0.0 else None for rate in dropout
		]
		self.dense = Dense(dense_layer_size, activation=tf.nn.relu)
		self.graph_gather = layers.GraphGather(
				batch_size=batch_size, activation_fn=tf.nn.tanh)
		self.trim = TrimGraphOutput()
		if self.mode == 'classification':
			self.reshape_dense = Dense(n_tasks * n_classes)
			self.reshape = Reshape((n_tasks, n_classes))
			self.softmax = Softmax()
		else:
			self.regression_dense = Dense(n_tasks)
			if self.uncertainty:
				self.uncertainty_dense = Dense(n_tasks)
				self.uncertainty_trim = TrimGraphOutput()
				self.uncertainty_activation = Activation(tf.exp)


	def call(self, inputs, training=False):
		atom_features = inputs[0]
		degree_slice = tf.cast(inputs[1], dtype=tf.int32)
		membership = tf.cast(inputs[2], dtype=tf.int32)
		n_samples = tf.cast(inputs[3], dtype=tf.int32)
		deg_adjs = [tf.cast(deg_adj, dtype=tf.int32) for deg_adj in inputs[4:]]

		in_layer = atom_features
		for i in range(len(self.graph_convs)):
			gc_in = [in_layer, degree_slice, membership] + deg_adjs
			gc1 = self.graph_convs[i](gc_in)
			if self.batch_norms[i] is not None:
				gc1 = self.batch_norms[i](gc1, training=training)
			if training and self.dropouts[i] is not None:
				gc1 = self.dropouts[i](gc1, training=training)
			gp_in = [gc1, degree_slice, membership] + deg_adjs
			in_layer = self.graph_pools[i](gp_in)
		dense = self.dense(in_layer)
		if self.batch_norms[-1] is not None:
			dense = self.batch_norms[-1](dense, training=training)
		if training and self.dropouts[-1] is not None:
			dense = self.dropouts[-1](dense, training=training)
		neural_fingerprint = self.graph_gather([dense, degree_slice, membership] +
																					 deg_adjs)
		if self.mode == 'classification':
			logits = self.reshape(self.reshape_dense(neural_fingerprint))
			logits = self.trim([logits, n_samples])
			output = self.softmax(logits)
			outputs = [output, logits, neural_fingerprint]
		else:
			output = self.regression_dense(neural_fingerprint)
			output = self.trim([output, n_samples])
			if self.uncertainty:
				log_var = self.uncertainty_dense(neural_fingerprint)
				log_var = self.uncertainty_trim([log_var, n_samples])
				var = self.uncertainty_activation(log_var)
				outputs = [output, var, output, log_var, neural_fingerprint]
			else:
				outputs = [output, neural_fingerprint]

		return outputs


class GraphConvModelExt(KerasModel):
	"""An extended version of deepchem.models.GraphConvModel, which allows
	user-defined activation function and pooling method.
	"""

	def __init__(self,
				 n_tasks: int,
				 graph_conv_layers: List[int] = [64, 64],
				 dense_layer_size: int = 128,
				 activation_fns=None,
				 graph_pools=None,
				 dropout: float = 0.0,
				 mode: str = "classification",
				 number_atom_features: int = 75,
				 n_classes: int = 2,
				 batch_size: int = 100,
				 batch_normalize: bool = True,
				 uncertainty: bool = False,
				 **kwargs):
		"""The wrapper class for graph convolutions.

		Note that since the underlying _GraphConvKerasModel class is
		specified using imperative subclassing style, this model
		cannout make predictions for arbitrary outputs.

		Parameters
		----------
		n_tasks: int
			Number of tasks
		graph_conv_layers: list of int
			Width of channels for the Graph Convolution Layers
		dense_layer_size: int
			Width of channels for Atom Level Dense Layer after GraphPool
		dropout: list or float
			the dropout probablity to use for each layer.	The length of this list
			should equal len(graph_conv_layers)+1 (one value for each convolution
			layer, and one for the dense layer).	Alternatively this may be a single
			value instead of a list, in which case the same value is used for every
			layer.
		mode: str
			Either "classification" or "regression"
		number_atom_features: int
			75 is the default number of atom features created, but
			this can vary if various options are passed to the
			function atom_features in graph_features
		n_classes: int
			the number of classes to predict (only used in classification mode)
		batch_normalize: True
			if True, apply batch normalization to model
		uncertainty: bool
			if True, include extra outputs and loss terms to enable the uncertainty
			in outputs to be predicted
		"""
		self.mode = mode
		self.n_tasks = n_tasks
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.uncertainty = uncertainty
		model = _GraphConvKerasModelExt(
				n_tasks,
				graph_conv_layers=graph_conv_layers,
				dense_layer_size=dense_layer_size,
				activation_fns=activation_fns,
				graph_pools=graph_pools,
				dropout=dropout,
				mode=mode,
				number_atom_features=number_atom_features,
				n_classes=n_classes,
				batch_normalize=batch_normalize,
				uncertainty=uncertainty,
				batch_size=batch_size)
		if mode == "classification":
			output_types = ['prediction', 'loss', 'embedding']
# 			loss: Union[Loss, LossFn] = SoftmaxCrossEntropy()
			kwargs['loss']: Union[Loss, LossFn] = kwargs.get('loss', SoftmaxCrossEntropy())
		else:
			if self.uncertainty:
				output_types = ['prediction', 'variance', 'loss', 'loss', 'embedding']

				def loss(outputs, labels, weights):
					output, labels = dc.models.losses._make_tf_shapes_consistent(
							outputs[0], labels[0])
					output, labels = dc.models.losses._ensure_float(output, labels)
					losses = tf.square(output - labels) / tf.exp(outputs[1]) + outputs[1]
					w = weights[0]
					if len(w.shape) < len(losses.shape):
						if tf.is_tensor(w):
							shape = tuple(w.shape.as_list())
						else:
							shape = w.shape
						shape = tuple(-1 if x is None else x for x in shape)
						w = tf.reshape(w, shape + (1,) * (len(losses.shape) - len(w.shape)))
					return tf.reduce_mean(losses * w) + sum(self.model.losses)
			else:
				output_types = ['prediction', 'embedding']
# 				loss = L2Loss()
				kwargs['loss'] = kwargs.get('loss', L2Loss())
		super(GraphConvModelExt, self).__init__(
# 				model, loss, output_types=output_types, batch_size=batch_size, **kwargs)
				model, output_types=output_types, batch_size=batch_size, **kwargs)


	def default_generator(self,
						dataset,
						epochs=1,
						mode='fit',
						deterministic=True,
						pad_batches=True):
		for epoch in range(epochs):
			for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
					batch_size=self.batch_size,
					deterministic=deterministic,
					pad_batches=pad_batches):
				if y_b is not None and self.mode == 'classification':
					y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
							-1, self.n_tasks, self.n_classes)
				multiConvMol = ConvMol.agglomerate_mols(X_b)
				n_samples = np.array(X_b.shape[0])
				inputs = [
						multiConvMol.get_atom_features(), multiConvMol.deg_slice,
						np.array(multiConvMol.membership), n_samples
				]
				for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
					inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
				yield (inputs, [y_b], [w_b])

