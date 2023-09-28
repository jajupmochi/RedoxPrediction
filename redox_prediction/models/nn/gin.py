"""
gin



@Author: linlin
@Date: 28.09.23
@References:
 - https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
"""
import torch
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from redox_prediction.models.nn.activation import get_activation


class GIN(torch.nn.Module):

	def __init__(
			self,
			in_feats: int,
			hidden_feats: int = 32,
			message_steps: int = 1,
			dim_target: int = 1,  # todo: set this for classification
			train_eps: bool = True,
			aggregation: str = 'sum',
			readout: str = 'concat',
			predictor_hidden_feats: int = 32,
			predictor_drop: float = 0.,
			predictor_clf_activation: str = 'log_softmax',
			mode: str = 'regression',
			**kwargs
	):
		super(GIN, self).__init__()

		self.n_layers = message_steps
		self.predictor_drop = predictor_drop
		self.mode = mode

		if aggregation == 'sum':
			self.pooling = global_add_pool
		elif aggregation == 'mean':
			self.pooling = global_mean_pool
		else:
			raise ValueError('Invalid aggregation: {}.'.format(aggregation))

		self.nns = []
		self.convs = []
		for _ in range(message_steps):
			self.nns.append(
				Sequential(
					Linear(in_feats, hidden_feats),
					BatchNorm1d(hidden_feats),
					ReLU(),
					Linear(hidden_feats, hidden_feats),
					BatchNorm1d(hidden_feats),
					ReLU()
				)
			)
			self.convs.append(
				GINConv(self.nns[-1], train_eps=train_eps)
			)  # Eq. 4.2
			in_feats = hidden_feats

		self.nns = torch.nn.ModuleList(self.nns)
		self.convs = torch.nn.ModuleList(self.convs)

		if mode == 'regression':
			dim_target = 1

		self.input_dense_dim = hidden_feats * self.n_layers
		self.dense_layer = Sequential(
			Linear(self.input_dense_dim, predictor_hidden_feats),
			ReLU(),
			Dropout(p=self.predictor_drop),
			Linear(predictor_hidden_feats, dim_target)
		)

		if mode == 'classification':
			self.predictor_clf_activation = get_activation(
				predictor_clf_activation
			)


	def forward(self, data, output='prediction'):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		hidden_repres = []

		for layer in range(self.n_layers):
			# Layer l ("convolution" layer)
			x = self.convs[layer](x, edge_index)
			hidden_repres.append(x)

		# Apply pooling:
		x = torch.cat(hidden_repres, dim=1)
		x = self.pooling(x, batch)

		# Apply dense layer:
		x = self.dense_layer(x)

		# Apply activation:
		if self.mode == 'classification':
			x = self.predictor_clf_activation(x)

		return x

# %% The following are modified based on the code of paper
# "How Powerful are Graph Neural Networks?" (https://arxiv.org/abs/1810.00826)
# GitHub: https://github.com/diningphil/gnn-comparison/blob/master/models/graph_classifiers/GIN.py
# It does not work well on Redox dataset.

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
# import torch
# import torch.nn.functional as F
# from torch.nn import BatchNorm1d
# from torch.nn import Sequential, Linear, ReLU
# from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
#
#
# class GIN(torch.nn.Module):
#
# 	def __init__(
# 			self,
# 			in_feats,
# 			dim_target: int = 1,
# 			mode: str = 'regression',
# 			**kwargs
# 	):
# 		super(GIN, self).__init__()
#
# 		config = {
# 			'dropout': kwargs.get('predictor_drop'),
# 			'hidden_units': [kwargs.get('hidden_feats')] * (
# 						kwargs.get('message_steps') + 1),
# 			'train_eps': kwargs.get('train_eps'),
# 			'aggregation': kwargs.get('aggregation'),
# 		}
#
# 		self.config = config
# 		self.dropout = config['dropout']
# 		self.embeddings_dim = [config['hidden_units'][0]] + config[
# 			'hidden_units']
# 		self.no_layers = len(self.embeddings_dim)
# 		self.first_h = []
# 		self.nns = []
# 		self.convs = []
# 		self.linears = []
#
# 		train_eps = config['train_eps']
# 		if config['aggregation'] == 'sum':
# 			self.pooling = global_add_pool
# 		elif config['aggregation'] == 'mean':
# 			self.pooling = global_mean_pool
#
# 		if mode == 'regression':
# 			dim_target = 1
#
# 		for layer, out_emb_dim in enumerate(self.embeddings_dim):
#
# 			if layer == 0:
# 				self.first_h = Sequential(
# 					Linear(in_feats, out_emb_dim), BatchNorm1d(out_emb_dim),
# 					ReLU(),
# 					Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim),
# 					ReLU()
# 				)
# 				self.linears.append(Linear(out_emb_dim, dim_target))
# 			else:
# 				input_emb_dim = self.embeddings_dim[layer - 1]
# 				self.nns.append(
# 					Sequential(
# 						Linear(input_emb_dim, out_emb_dim),
# 						BatchNorm1d(out_emb_dim), ReLU(),
# 						Linear(out_emb_dim, out_emb_dim),
# 						BatchNorm1d(out_emb_dim), ReLU()
# 					)
# 				)
# 				self.convs.append(
# 					GINConv(self.nns[-1], train_eps=train_eps)
# 				)  # Eq. 4.2
#
# 				self.linears.append(Linear(out_emb_dim, dim_target))
#
# 		self.nns = torch.nn.ModuleList(self.nns)
# 		self.convs = torch.nn.ModuleList(self.convs)
# 		self.linears = torch.nn.ModuleList(
# 			self.linears
# 		)  # has got one more for initial input
#
#
# 	def forward(self, data, output='prediction'):
# 		x, edge_index, batch = data.x, data.edge_index, data.batch
#
# 		out = 0
#
# 		for layer in range(self.no_layers):
# 			if layer == 0:
# 				x = self.first_h(x)
# 				out += F.dropout(
# 					self.pooling(self.linears[layer](x), batch), p=self.dropout
# 				)
# 			else:
# 				# Layer l ("convolution" layer)
# 				x = self.convs[layer - 1](x, edge_index)
# 				out += F.dropout(
# 					self.linears[layer](self.pooling(x, batch)), p=self.dropout,
# 					training=self.training
# 				)
#
# 		return out
