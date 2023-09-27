"""
dgcnn

Deep Graph Convolutional Neural Network (DGCNN) for graph regression and classification.

@Author: linlin
@Date: 01.09.23
@References:
	- https://github.com/diningphil/gnn-comparison/blob/master/models/graph_classifiers/DGCNN.py
	- https://github.com/CheshireCat12/filter_reduction/blob/main/models/DGCNN.py
"""
#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from typing import List, Union

import networkx as nx

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, SortAggregation
from torch_geometric.utils import add_self_loops, degree

from redox_prediction.models.nn.activation import get_activation

import logging

logging.captureWarnings(True)


class DGCNNConv(MessagePassing):
	"""
	Extended from tuorial on GCNs of Pytorch Geometrics
	"""


	def __init__(self, in_channels, out_channels):
		super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
		self.lin = nn.Linear(in_channels, out_channels)
		self.in_channels = in_channels
		self.out_channels = out_channels


	def forward(self, x, edge_index):
		# x has shape [N, in_channels]
		# edge_index has shape [2, E]

		# Step 1: Add self-loops to the adjacency matrix.
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

		# Step 2: Linearly transform node feature matrix.
		x = self.lin(x)

		# Step 3-5: Start propagating messages.
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)


	def message(self, x_j, edge_index, size):
		# x_j has shape [E, out_channels]

		# Step 3: Normalize node features.
		src, dst = edge_index  # we assume source_to_target message passing
		deg = degree(src, size[0], dtype=x_j.dtype)
		deg = deg.pow(-1)
		norm = deg[dst]

		return norm.view(
			-1, 1
		) * x_j  # broadcasting the normalization term to all out_channels === hidden features


	def update(self, aggr_out):
		# aggr_out has shape [N, out_channels]

		# Step 5: Return new node embeddings.
		return aggr_out


	def __repr__(self):
		return '{}({}, {})'.format(
			self.__class__.__name__, self.in_channels,
			self.out_channels
		)


class DGCNN(nn.Module):
	"""
	Uses fixed architecture.
	"""


	def __init__(
			self,
			in_feats,
			dim_target,
			k,
			hidden_feats,
			message_steps,
			predictor_hidden_feats,
			predictor_clf_activation,
			**kwargs
	):
		super(DGCNN, self).__init__()

		self.k = k
		self.hidden_feats = hidden_feats
		self.message_steps = message_steps

		self.convs = []
		for layer in range(self.message_steps):
			input_dim = in_feats if layer == 0 else self.hidden_feats
			self.convs.append(DGCNNConv(input_dim, self.hidden_feats))
		self.total_latent_dim = self.message_steps * self.hidden_feats

		# Add last embedding
		self.convs.append(DGCNNConv(self.hidden_feats, 1))
		self.total_latent_dim += 1

		self.convs = nn.ModuleList(self.convs)

		self.sort_pooling = SortAggregation(self.k)

		# should we leave this fixed?
		self.conv1d_params1 = nn.Conv1d(
			1, 16, self.total_latent_dim, self.total_latent_dim
		)
		self.maxpool1d = nn.MaxPool1d(2, 2)
		self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

		dense_dim = int((self.k - 2) / 2 + 1)
		self.input_dense_dim = (dense_dim - 5 + 1) * 32

		self.predictor_hidden_feats = predictor_hidden_feats
		self.dense_layer = nn.Sequential(
			nn.Linear(self.input_dense_dim, self.predictor_hidden_feats),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(self.predictor_hidden_feats, dim_target)
		)

		self.predictor_clf_activation = get_activation(predictor_clf_activation)


	def forward(self, data, output='prediction'):

		# data = Batch.from_data_list([self.dataset[i] for i in index])
		# Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
		# note: this can be decomposed in one smaller linear model per layer
		x, edge_index, batch = data.x, data.edge_index, data.batch

		hidden_repres = []

		for conv in self.convs:
			x = torch.tanh(conv(x, edge_index))
			hidden_repres.append(x)

		# apply sortpool
		x_to_sortpool = torch.cat(hidden_repres, dim=1)
		x_1d = self.sort_pooling(
			x_to_sortpool, index=batch
		)  # in the code the authors sort the last channel only

		# apply 1D convolutional layers
		x_1d = torch.unsqueeze(x_1d, dim=1)
		conv1d_res = F.relu(self.conv1d_params1(x_1d))
		conv1d_res = self.maxpool1d(conv1d_res)
		conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
		conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1)

		# apply dense layer
		out_dense = self.dense_layer(conv1d_res)

		# apply activation:
		log_probs = self.predictor_clf_activation(out_dense)

		return log_probs


def get_sort_pooling_k(
		graphs: List[Union[Data, nx.Graph]],
		percentile: float = 0.6,
		min_k: int = 10
) -> int:
	"""
	Get the first dimension of the sort pooling output according to the
	percentile of the number of nodes in the graphs.

	Notes
	-----
	For certain dataset (e.g., COIL-RAG, Letter-high, ...) the graphs are too small.
	Thus, the `k` could be too small and raise an exception. In this case, we set
	the `k` to 10 by `min_k`. This is a quick fix.

	References
	----------
	https://github.com/CheshireCat12/filter_reduction/blob/main/main_gnn.py#L23
	"""
	if isinstance(graphs[0], Data):
		nbr = [graph.num_nodes for graph in graphs]
	else:
		nbr = [graph.number_of_nodes() for graph in graphs]

	sorted_nbr = sorted(nbr)

	for i, x in enumerate(sorted_nbr):
		if (1 + i) / len(sorted_nbr) >= percentile:
			if x < min_k:
				return min_k
			return x

	return False
