"""
gcn_model



@Author: linlin
@Date: 24.05.23
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .activation import get_activation
from .readout import get_readout
from .mlp import MLP


class MessagePassing(nn.Module):
	def __init__(
			self,
			in_feats,
			hidden_feats: int = 32,
			message_steps: int = 1,
			normalize: bool = True,
			# weight: bool = True,
			bias: bool = True,
			feat_drop: float = 0.,
			kernel_drop: float = 0.,
			# 			residual: bool = False,
			agg_activation: str = 'relu',
			**kwargs
	):
		super().__init__(**kwargs)
		self.message_steps = message_steps

		for i in range(message_steps):
			n_h_feats = (
				hidden_feats[i] if isinstance(hidden_feats, list) else hidden_feats
			)

			# Feat dropout:
			if isinstance(feat_drop, list):
				fp = feat_drop[i]
			else:
				fp = feat_drop
			if fp > 0.:
				self.add_module('dropout_{}'.format(i), nn.Dropout(p=fp))

			# Convolutional layer:
			conv = GCNConv(
				in_feats,
				n_h_feats,
				improved=False,  # default is False
				add_self_loops=True,  # default is True
				normalize=normalize,  # default is True
				# weight=weight,
				bias=bias,  # default is True
				# 				feat_drop=feat_drop,
				# 				residual=residual,
			)
			self.add_module('conv_{}'.format(i), conv)

			# Activation function:
			activ_func = (
				agg_activation[i] if isinstance(agg_activation, list) else agg_activation
			)
			if activ_func is not None:
				activ_func = get_activation(activ_func)
				self.add_module('activ_func_{}'.format(i), activ_func)

			# Dropout:
			if isinstance(kernel_drop, list):
				kp = kernel_drop[i]
			else:
				kp = kernel_drop
			if kp > 0.:
				self.add_module('dropout_{}'.format(i), nn.Dropout(p=kp))

			in_feats = n_h_feats


	def forward(self, x, edge_index):
		for module in self._modules:
			if module.startswith('conv_'):
				x = self._modules[module](x, edge_index)
			else:
				x = self._modules[module](x)
		return x


	def reset_parameters(self):
		for conv in self.gcn_conv:
			conv.reset_parameters()


	# def __repr__(self):
	# 	"""
	# 	Override the __repr__ function to print the model architecture.
	# 	Include the model name and the model parameters.
	# 	"""
	# 	return '{}(message_steps={})'.format(
	# 		self.__class__.__name__,
	# 		self.message_steps
	# 	)


class GCN(torch.nn.Module):
	def __init__(
			self,
			# The followings are used by the GCNConv layer:
			in_feats,
			hidden_feats: int = 32,
			message_steps: int = 1,
			normalize: bool = True,
			# weight: bool = True,
			bias: bool = True,
			feat_drop: float = 0.,
			kernel_drop: float = 0.,
			# 			residual: bool = False,
			# The followings are used for aggragation of the outputs:
			agg_activation: str = 'relu',
			# The followings are used for readout:
			readout: str = 'mean',
			mode: str = 'regression',
			# The following are PyTorch settings:
			device: str = 'cpu',
			**kwargs  # batch_size: int = 32,  # for transformer readout
	):
		super().__init__()

		# Message passing
		self.msg_passing = MessagePassing(
			in_feats,
			hidden_feats=hidden_feats,
			message_steps=message_steps,
			normalize=normalize,
			# weight=weight,
			bias=bias,
			feat_drop=feat_drop,
			kernel_drop=kernel_drop,
			# 			residual=residual,
			agg_activation=agg_activation,
		)

		# Readout
		self.readout = get_readout(readout)

		# Predict.
		n_h_feats = (
			hidden_feats[-1] if isinstance(hidden_feats, list) else hidden_feats
		)
		# Node level prediction:
		if self.readout is None:
			self.predict = []  # [nn.Linear(n_h_feats, 1, bias=predictor_bias)]
			if mode == 'classification':
				self.predict.append(get_activation(predictor_clf_activation))
			self.predict = nn.Sequential(*self.predict)
		# Graph level prediction:
		else:
			self.predict = MLP(
				n_h_feats,
				out_feats=1,
				# 			residual=predictor_residual,
				mode=mode,
				n_classes=kwargs.get('n_classes'),
				**{k[10:]: v for k, v in kwargs.items() if k.startswith('predictor_')}
			)


	def forward(self, data, output='prediction'):
		x, edge_index, ptr = data.x, data.edge_index, data.ptr

		x = self.msg_passing(x, edge_index)
		if self.readout is not None:
			x = self.readout(x, ptr=ptr)
		x = self.predict(x, output=output)

		return x


	def reset_parameters(self):
		self.msg_passing.reset_parameters()
		self.predict.reset_parameters()


	# def __repr__(self):
	# 	"""
	# 	Override the __repr__ function to print the model architecture.
	# 	Include the model name and the model parameters.
	# 	"""
	# 	return '{}(msg_passing={}, readout={}, predict={})'.format(
	# 		self.__class__.__name__,
	# 		self.msg_passing,
	# 		self.readout,
	# 		self.predict,
	# 	)


if __name__ == '__main__':

	# %% Test my GCN implementation for node classification:
	from torch_geometric.datasets import Planetoid

	dataset = Planetoid(root='/tmp/Cora', name='Cora')

	import torch
	import torch.nn.functional as F

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	model = GCN(
		in_feats=dataset.num_node_features,
		hidden_feats=[16, dataset.num_classes],
		message_steps=2,
		normalize=True,
		bias=True,
		kernel_drop=[0.5, 0.],
		agg_activation=['relu', None],
		readout=None,
		predictor_clf_activation='log_softmax',
		mode='classification',
		device=device
	).to(device)
	data = dataset[0].to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

	model.train()
	for epoch in range(200):
		optimizer.zero_grad()
		out = model(data)
		loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
		loss.backward()
		optimizer.step()

	model.eval()
	pred = model(data).argmax(dim=1)
	correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
	acc = int(correct) / int(data.test_mask.sum())
	print(f'Accuracy: {acc:.4f}')



	# # %% Original GCN example:
	# from torch_geometric.datasets import Planetoid
	#
	# dataset = Planetoid(root='/tmp/Cora', name='Cora')
	#
	# import torch
	# import torch.nn.functional as F
	# from torch_geometric.nn import GCNConv
	#
	#
	# class GCN(torch.nn.Module):
	# 	def __init__(self):
	# 		super().__init__()
	# 		self.conv1 = GCNConv(dataset.num_node_features, 16)
	# 		self.conv2 = GCNConv(16, dataset.num_classes)
	#
	#
	# 	def forward(self, data):
	# 		x, edge_index = data.x, data.edge_index
	#
	# 		x = self.conv1(x, edge_index)
	# 		x = F.relu(x)
	# 		x = F.dropout(x, training=self.training)
	# 		x = self.conv2(x, edge_index)
	#
	# 		return F.log_softmax(x, dim=1)
	#
	#
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# print(device)
	# model = GCN().to(device)
	# data = dataset[0].to(device)
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	#
	# model.train()
	# for epoch in range(200):
	# 	optimizer.zero_grad()
	# 	out = model(data)
	# 	loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
	# 	loss.backward()
	# 	optimizer.step()
	#
	# model.eval()
	# pred = model(data).argmax(dim=1)
	# correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
	# acc = int(correct) / int(data.test_mask.sum())
	# print(f'Accuracy: {acc:.4f}')
