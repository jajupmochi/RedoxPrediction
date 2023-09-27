"""
mlp



@Author: linlin
@Date: 25.05.23
"""

import torch
from torch import nn
import torch.nn.functional as F

from .activation import get_activation


class MLP(nn.Module):
	def __init__(
			self,
			in_feats,
			hidden_feats: int = 512,
			n_hidden_layers: int = 1,
			out_feats: int = 1,
			feat_drop: float = 0.,
			kernel_drop: float = 0.,
			bias: bool = True,
			# 			residual: bool = False,
			batchnorm: bool = False,
			activation: str = 'relu',
			clf_activation: str = 'sigmoid',
			n_classes: int = None,
			mode: str = 'regression',
			**kwargs
	):
		super().__init__()
		self.n_hidden_layers = n_hidden_layers
		activation_fun = get_activation(activation)

		for i in range(n_hidden_layers):
			if feat_drop > 0.:
				self.add_module(
					'feat_drop_{}'.format(i), nn.Dropout(p=feat_drop)
				)
			self.add_module(
				'linear_{}'.format(i),
				nn.Linear(in_feats, hidden_feats, bias=bias)
			)
			if batchnorm:
				self.add_module(
					'batchnorm_{}'.format(i), nn.BatchNorm1d(hidden_feats)
				)
			# activation:
			if activation_fun is not None:
				self.add_module(
					'activation_{}'.format(i), activation_fun
				)
			if kernel_drop > 0.:
				self.add_module(
					'kernel_drop_{}'.format(i), nn.Dropout(p=kernel_drop)
				)

			in_feats = hidden_feats

		self.n_layers_embeddings = len(self._modules)


		# Define the proper number of output features:
		if mode == 'regression':
			out_feats = out_feats
		elif mode == 'classification':
			if clf_activation == 'sigmoid':
				out_feats = 1
			elif clf_activation in ['softmax', 'log_softmax']:
				out_feats = n_classes
		self.add_module(
			'linear_out', nn.Linear(in_feats, out_feats, bias=bias)
		)
		if mode == 'classification':
			# self.add_module(
			# 	'batchnorm_out', nn.BatchNorm1d(out_feats)  # @TODO: check if this is necessary
			# )
			self.add_module(
				'activation_out', get_activation(clf_activation)
			)


	def forward(self, x, output='prediction'):
		keys = list(self._modules.keys())
		for module in keys[:self.n_layers_embeddings]:
			x = self._modules[module](x)
		if output == 'embedding':
			return x
		embedding = x
		for module in keys[self.n_layers_embeddings:]:
			x = self._modules[module](x)
		if output == 'prediction':
			return x
		elif output == 'both':
			return x, embedding
		# if return_embeddings:
		# 	embeddings = x
		# for module in keys[self.n_layers_embeddings:]:
		# 	x = self._modules[module](x)
		# if return_embeddings:
		# 	return embeddings
		# else:
		# 	return x


	def reset_parameters(self):
		for layer in self.layers:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)

# def __repr__(self):
# 	"""
# 	Override the __repr__ function to print the model architecture.
# 	Include the model name and the model parameters.
# 	"""
# 	return '{}(message_steps={})'.format(
# 		self.__class__.__name__,
# 		self.message_steps
# 	)
