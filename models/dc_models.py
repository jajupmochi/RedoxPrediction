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
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, Tuple, Iterable, Dict, Optional
from deepchem.utils.typing import OneOrMany, LossFn, ActivationFn
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, Loss, SparseSoftmaxCrossEntropy
from deepchem.models.graph_models import TrimGraphOutput
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization


#%%


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


#%%


class GATExt(nn.Module):
	"""Model for Graph Property Prediction Based on Graph Attention Networks (GAT).

	This model proceeds as follows:

	* Update node representations in graphs with a variant of GAT
	* For each graph, compute its representation by 1) a weighted sum of the node
		representations in the graph, where the weights are computed by applying a
		gating function to the node representations 2) a max pooling of the node
		representations 3) concatenating the output of 1) and 2)
	* Perform the final prediction using an MLP

	Examples
	--------

	>>> import deepchem as dc
	>>> import dgl
	>>> from deepchem.models import GAT
	>>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
	>>> featurizer = dc.feat.MolGraphConvFeaturizer()
	>>> graphs = featurizer.featurize(smiles)
	>>> print(type(graphs[0]))
	<class 'deepchem.feat.graph_data.GraphData'>
	>>> dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
	>>> # Batch two graphs into a graph of two connected components
	>>> batch_dgl_graph = dgl.batch(dgl_graphs)
	>>> model = GAT(n_tasks=1, mode='regression')
	>>> preds = model(batch_dgl_graph)
	>>> print(type(preds))
	<class 'torch.Tensor'>
	>>> preds.shape == (2, 1)
	True

	References
	----------
	.. [1] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò,
				 and Yoshua Bengio. "Graph Attention Networks." ICLR 2018.

	Notes
	-----
	This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
	(https://github.com/awslabs/dgl-lifesci) to be installed.
	"""

	def __init__(self,
							 n_tasks: int,
							 graph_attention_layers: list = None,
							 n_attention_heads: int = 8,
							 agg_modes: list = None,
							 activation_fns=None,
							 graph_pools=None,
							 residual: bool = True,
							 dropout: float = 0.,
							 alpha: float = 0.2,
							 predictor_hidden_feats: int = 128,
							 predictor_dropout: float = 0.,
							 mode: str = 'regression',
							 number_atom_features: int = 30,
							 n_classes: int = 2,
							 nfeat_name: str = 'x'):
		"""
		Parameters
		----------
		n_tasks: int
			Number of tasks.
		graph_attention_layers: list of int
			Width of channels per attention head for GAT layers. graph_attention_layers[i]
			gives the width of channel for each attention head for the i-th GAT layer. If
			both ``graph_attention_layers`` and ``agg_modes`` are specified, they should have
			equal length. If not specified, the default value will be [8, 8].
		n_attention_heads: int
			Number of attention heads in each GAT layer.
		agg_modes: list of str
			The way to aggregate multi-head attention results for each GAT layer, which can be
			either 'flatten' for concatenating all-head results or 'mean' for averaging all-head
			results. ``agg_modes[i]`` gives the way to aggregate multi-head attention results for
			the i-th GAT layer. If both ``graph_attention_layers`` and ``agg_modes`` are
			specified, they should have equal length. If not specified, the model will flatten
			multi-head results for intermediate GAT layers and compute mean of multi-head results
			for the last GAT layer.
		activation: activation function or None
			The activation function to apply to the aggregated multi-head results for each GAT
			layer. If not specified, the default value will be ELU.
		residual: bool
			Whether to add a residual connection within each GAT layer. Default to True.
		dropout: float
			The dropout probability within each GAT layer. Default to 0.
		alpha: float
			A hyperparameter in LeakyReLU, which is the slope for negative values. Default to 0.2.
		predictor_hidden_feats: int
			The size for hidden representations in the output MLP predictor. Default to 128.
		predictor_dropout: float
			The dropout probability in the output MLP predictor. Default to 0.
		mode: str
			The model type, 'classification' or 'regression'. Default to 'regression'.
		number_atom_features: int
			The length of the initial atom feature vectors. Default to 30.
		n_classes: int
			The number of classes to predict per task
			(only used when ``mode`` is 'classification'). Default to 2.
		nfeat_name: str
			For an input graph ``g``, the model assumes that it stores node features in
			``g.ndata[nfeat_name]`` and will retrieve input node features from that.
			Default to 'x'.
		"""
		try:
			import dgl
		except:
			raise ImportError('This class requires dgl.')
		try:
			import dgllife
		except:
			raise ImportError('This class requires dgllife.')

		if mode not in ['classification', 'regression']:
			raise ValueError("mode must be either 'classification' or 'regression'")

		super(GATExt, self).__init__()

		self.n_tasks = n_tasks
		self.mode = mode
		self.n_classes = n_classes
		self.nfeat_name = nfeat_name
		if mode == 'classification':
			out_size = n_tasks * n_classes
		else:
			out_size = n_tasks

		from dgllife.model import GATPredictor as DGLGATPredictor

		if isinstance(graph_attention_layers, list) and isinstance(agg_modes, list):
			assert len(graph_attention_layers) == len(agg_modes), \
				'Expect graph_attention_layers and agg_modes to have equal length, ' \
				'got {:d} and {:d}'.format(len(graph_attention_layers), len(agg_modes))

		# Decide first number of GAT layers
		if graph_attention_layers is not None:
			num_gnn_layers = len(graph_attention_layers)
# 		elif agg_modes is not None:
# 			num_gnn_layers = len(agg_modes)
		else:
			num_gnn_layers = 2

		if graph_attention_layers is None:
			graph_attention_layers = [8] * num_gnn_layers
		if agg_modes is None:
			agg_modes = ['flatten' for _ in range(num_gnn_layers - 1)]
			agg_modes.append('mean')
		else:
			agg_modes = [agg_modes for _ in range(num_gnn_layers)]

# 		if activation is not None:
# 			activation = [activation] * num_gnn_layers
		if activation_fns is None:
			activation_fns = [F.elu] * num_gnn_layers

		self.model = DGLGATPredictor(
				in_feats=number_atom_features,
				hidden_feats=graph_attention_layers,
				num_heads=[n_attention_heads] * num_gnn_layers,
				feat_drops=[dropout] * num_gnn_layers,
				attn_drops=[dropout] * num_gnn_layers,
				alphas=[alpha] * num_gnn_layers,
				residuals=[residual] * num_gnn_layers,
				agg_modes=agg_modes,
				activations=activation_fns,
				n_tasks=out_size,
				predictor_hidden_feats=predictor_hidden_feats,
				predictor_dropout=predictor_dropout)


	def forward(self, g):
		"""Predict graph labels

		Parameters
		----------
		g: DGLGraph
			A DGLGraph for a batch of graphs. It stores the node features in
			``dgl_graph.ndata[self.nfeat_name]``.

		Returns
		-------
		torch.Tensor
			The model output.

			* When self.mode = 'regression',
				its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
			* When self.mode = 'classification', the output consists of probabilities
				for classes. Its shape will be
				``(dgl_graph.batch_size, self.n_tasks, self.n_classes)`` if self.n_tasks > 1;
				its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if self.n_tasks is 1.
		torch.Tensor, optional
			This is only returned when self.mode = 'classification', the output consists of the
			logits for classes before softmax.
		"""
		node_feats = g.ndata[self.nfeat_name]
		out = self.model(g, node_feats)

		if self.mode == 'classification':
			if self.n_tasks == 1:
				logits = out.view(-1, self.n_classes)
				softmax_dim = 1
			else:
				logits = out.view(-1, self.n_tasks, self.n_classes)
				softmax_dim = 2
			proba = F.softmax(logits, dim=softmax_dim)
			return proba, logits
		else:
			return out


class GATModelExt(TorchModel):
	"""Model for Graph Property Prediction Based on Graph Attention Networks (GAT).

	This model proceeds as follows:

	* Update node representations in graphs with a variant of GAT
	* For each graph, compute its representation by 1) a weighted sum of the node
		representations in the graph, where the weights are computed by applying a
		gating function to the node representations 2) a max pooling of the node
		representations 3) concatenating the output of 1) and 2)
	* Perform the final prediction using an MLP

	Examples
	--------
	>>> import deepchem as dc
	>>> from deepchem.models import GATModel
	>>> # preparing dataset
	>>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
	>>> labels = [0., 1.]
	>>> featurizer = dc.feat.MolGraphConvFeaturizer()
	>>> X = featurizer.featurize(smiles)
	>>> dataset = dc.data.NumpyDataset(X=X, y=labels)
	>>> # training model
	>>> model = GATModel(mode='classification', n_tasks=1,
	...									batch_size=16, learning_rate=0.001)
	>>> loss = model.fit(dataset, nb_epoch=5)

	References
	----------
	.. [1] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò,
				 and Yoshua Bengio. "Graph Attention Networks." ICLR 2018.

	Notes
	-----
	This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
	(https://github.com/awslabs/dgl-lifesci) to be installed.
	"""

	def __init__(self,
							 n_tasks: int,
							 graph_attention_layers: list = None,
							 n_attention_heads: int = 8,
							 agg_modes: list = None,
							 activation_fns=None,
							 graph_pools=None,
							 residual: bool = True,
							 dropout: float = 0.,
							 alpha: float = 0.2,
							 predictor_hidden_feats: int = 128,
							 predictor_dropout: float = 0.,
							 mode: str = 'regression',
							 number_atom_features: int = 30,
							 n_classes: int = 2,
							 self_loop: bool = True,
							 **kwargs):
		"""
		Parameters
		----------
		n_tasks: int
			Number of tasks.
		graph_attention_layers: list of int
			Width of channels per attention head for GAT layers. graph_attention_layers[i]
			gives the width of channel for each attention head for the i-th GAT layer. If
			both ``graph_attention_layers`` and ``agg_modes`` are specified, they should have
			equal length. If not specified, the default value will be [8, 8].
		n_attention_heads: int
			Number of attention heads in each GAT layer.
		agg_modes: list of str
			The way to aggregate multi-head attention results for each GAT layer, which can be
			either 'flatten' for concatenating all-head results or 'mean' for averaging all-head
			results. ``agg_modes[i]`` gives the way to aggregate multi-head attention results for
			the i-th GAT layer. If both ``graph_attention_layers`` and ``agg_modes`` are
			specified, they should have equal length. If not specified, the model will flatten
			multi-head results for intermediate GAT layers and compute mean of multi-head results
			for the last GAT layer.
		activation: activation function or None
			The activation function to apply to the aggregated multi-head results for each GAT
			layer. If not specified, the default value will be ELU.
		residual: bool
			Whether to add a residual connection within each GAT layer. Default to True.
		dropout: float
			The dropout probability within each GAT layer. Default to 0.
		alpha: float
			A hyperparameter in LeakyReLU, which is the slope for negative values. Default to 0.2.
		predictor_hidden_feats: int
			The size for hidden representations in the output MLP predictor. Default to 128.
		predictor_dropout: float
			The dropout probability in the output MLP predictor. Default to 0.
		mode: str
			The model type, 'classification' or 'regression'. Default to 'regression'.
		number_atom_features: int
			The length of the initial atom feature vectors. Default to 30.
		n_classes: int
			The number of classes to predict per task
			(only used when ``mode`` is 'classification'). Default to 2.
		self_loop: bool
			Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
			When input graphs have isolated nodes, self loops allow preserving the original feature
			of them in message passing. Default to True.
		kwargs
			This can include any keyword argument of TorchModel.
		"""
		model = GATExt(
				n_tasks=n_tasks,
				graph_attention_layers=graph_attention_layers,
				n_attention_heads=n_attention_heads,
				agg_modes=agg_modes,
				activation_fns=activation_fns,
				graph_pools=graph_pools,
				residual=residual,
				dropout=dropout,
				alpha=alpha,
				predictor_hidden_feats=predictor_hidden_feats,
				predictor_dropout=predictor_dropout,
				mode=mode,
				number_atom_features=number_atom_features,
				n_classes=n_classes)
		if mode == 'regression':
			# loss: Loss = L2Loss()
			kwargs['loss']: Loss = kwargs.get('loss', L2Loss)
			output_types = ['prediction']
		else:
			kwargs['loss'] = kwargs.get('loss', SparseSoftmaxCrossEntropy())
			output_types = ['prediction', 'loss']
		super(GATModelExt, self).__init__(
# 				model, loss=loss, output_types=output_types, **kwargs)
				model, output_types=output_types, **kwargs)

		self._self_loop = self_loop

	def _prepare_batch(self, batch):
		"""Create batch data for GAT.

		Parameters
		----------
		batch: tuple
			The tuple is ``(inputs, labels, weights)``.

		Returns
		-------
		inputs: DGLGraph
			DGLGraph for a batch of graphs.
		labels: list of torch.Tensor or None
			The graph labels.
		weights: list of torch.Tensor or None
			The weights for each sample or sample/task pair converted to torch.Tensor.
		"""
		try:
			import dgl
		except:
			raise ImportError('This class requires dgl.')

		inputs, labels, weights = batch
		dgl_graphs = [
				graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
		]
		inputs = dgl.batch(dgl_graphs).to(self.device)
		_, labels, weights = super(GATModelExt, self)._prepare_batch(([], labels,
																															 weights))
		return inputs, labels, weights


#%%


class GCNExt(nn.Module):
  """Model for Graph Property Prediction Based on Graph Convolution Networks (GCN).

  This model proceeds as follows:

  * Update node representations in graphs with a variant of GCN
  * For each graph, compute its representation by 1) a weighted sum of the node
    representations in the graph, where the weights are computed by applying a
    gating function to the node representations 2) a max pooling of the node
    representations 3) concatenating the output of 1) and 2)
  * Perform the final prediction using an MLP

  Examples
  --------

  >>> import deepchem as dc
  >>> import dgl
  >>> from deepchem.models import GCN
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = dc.feat.MolGraphConvFeaturizer()
  >>> graphs = featurizer.featurize(smiles)
  >>> print(type(graphs[0]))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
  >>> # Batch two graphs into a graph of two connected components
  >>> batch_dgl_graph = dgl.batch(dgl_graphs)
  >>> model = GCN(n_tasks=1, mode='regression')
  >>> preds = model(batch_dgl_graph)
  >>> print(type(preds))
  <class 'torch.Tensor'>
  >>> preds.shape == (2, 1)
  True

  References
  ----------
  .. [1] Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph
         Convolutional Networks." ICLR 2017.

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.

  This model is different from deepchem.models.GraphConvModel as follows:

  * For each graph convolution, the learnable weight in this model is shared across all nodes.
    ``GraphConvModel`` employs separate learnable weights for nodes of different degrees. A
    learnable weight is shared across all nodes of a particular degree.
  * For ``GraphConvModel``, there is an additional GraphPool operation after each
    graph convolution. The operation updates the representation of a node by applying an
    element-wise maximum over the representations of its neighbors and itself.
  * For computing graph-level representations, this model computes a weighted sum and an
    element-wise maximum of the representations of all nodes in a graph and concatenates them.
    The node weights are obtained by using a linear/dense layer followd by a sigmoid function.
    For ``GraphConvModel``, the sum over node representations is unweighted.
  * There are various minor differences in using dropout, skip connection and batch
    normalization.
  """

  def __init__(self,
               n_tasks: int,
               graph_conv_layers: list = None,
               activation_fns=None,
			   graph_pools=None,
               residual: bool = True,
               batchnorm: bool = False,
               dropout: float = 0.,
               predictor_hidden_feats: int = 128,
               predictor_dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               n_classes: int = 2,
               nfeat_name: str = 'x'):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    graph_conv_layers: list of int
      Width of channels for GCN layers. graph_conv_layers[i] gives the width of channel
      for the i-th GCN layer. If not specified, the default value will be [64, 64].
    activation: callable
      The activation function to apply to the output of each GCN layer.
      By default, no activation function will be applied.
    residual: bool
      Whether to add a residual connection within each GCN layer. Default to True.
    batchnorm: bool
      Whether to apply batch normalization to the output of each GCN layer.
      Default to False.
    dropout: float
      The dropout probability for the output of each GCN layer. Default to 0.
    predictor_hidden_feats: int
      The size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout: float
      The dropout probability in the output MLP predictor. Default to 0.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    nfeat_name: str
      For an input graph ``g``, the model assumes that it stores node features in
      ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
      Default to 'x'.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')
    try:
      import dgllife
    except:
      raise ImportError('This class requires dgllife.')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    super(GCNExt, self).__init__()

    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes
    self.nfeat_name = nfeat_name
    if mode == 'classification':
      out_size = n_tasks * n_classes
    else:
      out_size = n_tasks

    from dgllife.model import GCNPredictor as DGLGCNPredictor

    if graph_conv_layers is None:
      graph_conv_layers = [64, 64]
    num_gnn_layers = len(graph_conv_layers)

    if activation_fns is None:
      activation_fns = [F.elu] * num_gnn_layers

    self.model = DGLGCNPredictor(
        in_feats=number_atom_features,
        hidden_feats=graph_conv_layers,
        activation=activation_fns,
        residual=[residual] * num_gnn_layers,
        batchnorm=[batchnorm] * num_gnn_layers,
        dropout=[dropout] * num_gnn_layers,
        n_tasks=out_size,
        predictor_hidden_feats=predictor_hidden_feats,
        predictor_dropout=predictor_dropout)


  def forward(self, g):
    """Predict graph labels

    Parameters
    ----------
    g: DGLGraph
      A DGLGraph for a batch of graphs. It stores the node features in
      ``dgl_graph.ndata[self.nfeat_name]``.

    Returns
    -------
    torch.Tensor
      The model output.

      * When self.mode = 'regression',
        its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
      * When self.mode = 'classification', the output consists of probabilities
        for classes. Its shape will be ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)``
        if self.n_tasks > 1; its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if
        self.n_tasks is 1.
      torch.Tensor, optional
        This is only returned when self.mode = 'classification', the output consists of the
        logits for classes before softmax.
    """
    node_feats = g.ndata[self.nfeat_name]
    out = self.model(g, node_feats)

    if self.mode == 'classification':
      if self.n_tasks == 1:
        logits = out.view(-1, self.n_classes)
        softmax_dim = 1
      else:
        logits = out.view(-1, self.n_tasks, self.n_classes)
        softmax_dim = 2
      proba = F.softmax(logits, dim=softmax_dim)
      return proba, logits
    else:
      return out



class GCNModelExt(TorchModel):
	"""Model for Graph Property Prediction Based on Graph Convolution Networks (GCN).

	This model proceeds as follows:

	* Update node representations in graphs with a variant of GCN
	* For each graph, compute its representation by 1) a weighted sum of the node
		representations in the graph, where the weights are computed by applying a
		gating function to the node representations 2) a max pooling of the node
		representations 3) concatenating the output of 1) and 2)
	* Perform the final prediction using an MLP

	Examples
	--------
	>>> import deepchem as dc
	>>> from deepchem.models import GCNModel
	>>> # preparing dataset
	>>> smiles = ["C1CCC1", "CCC"]
	>>> labels = [0., 1.]
	>>> featurizer = dc.feat.MolGraphConvFeaturizer()
	>>> X = featurizer.featurize(smiles)
	>>> dataset = dc.data.NumpyDataset(X=X, y=labels)
	>>> # training model
	>>> model = GCNModel(mode='classification', n_tasks=1,
	...									batch_size=16, learning_rate=0.001)
	>>> loss = model.fit(dataset, nb_epoch=5)

	References
	----------
	.. [1] Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph
				 Convolutional Networks." ICLR 2017.

	Notes
	-----
	This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
	(https://github.com/awslabs/dgl-lifesci) to be installed.

	This model is different from deepchem.models.GraphConvModel as follows:

	* For each graph convolution, the learnable weight in this model is shared across all nodes.
		``GraphConvModel`` employs separate learnable weights for nodes of different degrees. A
		learnable weight is shared across all nodes of a particular degree.
	* For ``GraphConvModel``, there is an additional GraphPool operation after each
		graph convolution. The operation updates the representation of a node by applying an
		element-wise maximum over the representations of its neighbors and itself.
	* For computing graph-level representations, this model computes a weighted sum and an
		element-wise maximum of the representations of all nodes in a graph and concatenates them.
		The node weights are obtained by using a linear/dense layer followd by a sigmoid function.
		For ``GraphConvModel``, the sum over node representations is unweighted.
	* There are various minor differences in using dropout, skip connection and batch
		normalization.
	"""

	def __init__(self,
							 n_tasks: int,
							 graph_conv_layers: list = None,
							 activation_fns=None,
				 graph_pools=None,
							 residual: bool = True,
							 batchnorm: bool = False,
							 dropout: float = 0.,
							 predictor_hidden_feats: int = 128,
							 predictor_dropout: float = 0.,
							 mode: str = 'regression',
							 number_atom_features=30,
							 n_classes: int = 2,
							 self_loop: bool = True,
							 **kwargs):
		"""
		Parameters
		----------
		n_tasks: int
			Number of tasks.
		graph_conv_layers: list of int
			Width of channels for GCN layers. graph_conv_layers[i] gives the width of channel
			for the i-th GCN layer. If not specified, the default value will be [64, 64].
		activation: callable
			The activation function to apply to the output of each GCN layer.
			By default, no activation function will be applied.
		residual: bool
			Whether to add a residual connection within each GCN layer. Default to True.
		batchnorm: bool
			Whether to apply batch normalization to the output of each GCN layer.
			Default to False.
		dropout: float
			The dropout probability for the output of each GCN layer. Default to 0.
		predictor_hidden_feats: int
			The size for hidden representations in the output MLP predictor. Default to 128.
		predictor_dropout: float
			The dropout probability in the output MLP predictor. Default to 0.
		mode: str
			The model type, 'classification' or 'regression'. Default to 'regression'.
		number_atom_features: int
			The length of the initial atom feature vectors. Default to 30.
		n_classes: int
			The number of classes to predict per task
			(only used when ``mode`` is 'classification'). Default to 2.
		self_loop: bool
			Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
			When input graphs have isolated nodes, self loops allow preserving the original feature
			of them in message passing. Default to True.
		kwargs
			This can include any keyword argument of TorchModel.
		"""
		model = GCNExt(
				n_tasks=n_tasks,
				graph_conv_layers=graph_conv_layers,
				activation_fns=activation_fns,
		graph_pools=graph_pools,
				residual=residual,
				batchnorm=batchnorm,
				dropout=dropout,
				predictor_hidden_feats=predictor_hidden_feats,
				predictor_dropout=predictor_dropout,
				mode=mode,
				number_atom_features=number_atom_features,
				n_classes=n_classes)
		if mode == 'regression':
			# loss: Loss = L2Loss()
			kwargs['loss']: Loss = kwargs.get('loss', L2Loss)
			output_types = ['prediction']
		else:
# 			loss = SparseSoftmaxCrossEntropy()
			kwargs['loss'] = kwargs.get('loss', SparseSoftmaxCrossEntropy())
			output_types = ['prediction', 'loss']
		super(GCNModelExt, self).__init__(
# 				model, loss=loss, output_types=output_types, **kwargs)
				model, output_types=output_types, **kwargs)

		self._self_loop = self_loop

	def _prepare_batch(self, batch):
		"""Create batch data for GCN.

		Parameters
		----------
		batch: tuple
			The tuple is ``(inputs, labels, weights)``.

		Returns
		-------
		inputs: DGLGraph
			DGLGraph for a batch of graphs.
		labels: list of torch.Tensor or None
			The graph labels.
		weights: list of torch.Tensor or None
			The weights for each sample or sample/task pair converted to torch.Tensor.
		"""
		try:
			import dgl
		except:
			raise ImportError('This class requires dgl.')

		inputs, labels, weights = batch
		dgl_graphs = [
				graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
		]
		inputs = dgl.batch(dgl_graphs).to(self.device)
		_, labels, weights = super(GCNModelExt, self)._prepare_batch(([], labels,
																															 weights))
		return inputs, labels, weights

