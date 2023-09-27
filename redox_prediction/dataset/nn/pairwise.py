"""
pairwise



@Author: linlin
@Date: 26.07.23
"""
import itertools
import numpy as np

import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset


class PairwiseDataset(Dataset):
	def __init__(
			self,
			dataset: pyg.data.Dataset,
			metric_matrix: np.ndarray = None,
	):
		"""Construct a pairwise dataset from a PyG dataset that is compatible
		with the PyG DataLoader.

		Parameters
		----------
		dataset : torch_geometric.data.Dataset
			A PyG dataset. Must have a `data_list` attribute. The `data_list`
			attribute must be a list of PyG Data objects.
			Each Data object in the dataset will have the following attributes:
				- x: Node feature matrix. Shape [num_nodes, num_node_features]
				- edge_index: Graph connectivity in COO format with shape [2, num_edges]
				- edge_attr: Edge feature matrix. Shape [num_edges, num_edge_features]
				- y: Target to be used for training. Shape [1]
				- graph: The original NetworkX graph. Only included if keep_nx_graphs
				  is True. Shape [1]
			Some redundant attributes may be included in the Data objects. They may
			be removed in the future.

		metric_matrix : np.ndarray, optional (default=None)
			A matrix of pairwise distances between the graphs in the dataset.
		"""
		# super(PairwiseDataset, self).__init__()
		#
		# self.dataset = dataset
		# self.data_list = dataset.data_list
		#
		# self.pairs = list(itertools.combinations(range(len(self.data_list)), 2))
		# self.pair_data = [self.get_pair_data(pair) for pair in self.pairs]


		# super(PairwiseDataset, self).__init__()
		#
		# # Check that the dataset has a `data_list` attribute.
		# if not hasattr(dataset, 'data_list'):
		# 	raise ValueError('dataset must have a `data_list` attribute.')
		# # Check that the `data_list` attribute is a list.
		# if not isinstance(dataset.data_list, list):
		# 	raise ValueError('dataset.data_list must be a list.')
		# # Check that the `data_list` attribute is not empty.
		# if len(dataset.data_list) == 0:
		# 	raise ValueError('dataset.data_list must not be empty.')
		# # Check that the `data_list` attribute is a list of PyG data objects.
		# if not all(
		# 		[isinstance(data, pyg.data.Data) for data in dataset.data_list]
		# ):
		# 	raise ValueError(
		# 		'dataset.data_list must be a list of PyG data objects.'
		# 	)
		#
		# # Assign the pairwise data to the `self.pair_data` attribute:
		# self.pairs = list(
		# 	itertools.permutations(range(len(dataset.data_list)), 2)
		# )
		# self.pair_data = [(
		# 	dataset.data_list[ind[0]], dataset.data_list[ind[1]]
		# ) for ind in self.pairs]

		super(PairwiseDataset, self).__init__()

		self.dataset = dataset
		# self.pairs = list(
		# 	itertools.permutations(range(dataset.len()), 2)
		# )
		self.pairs = list(
			itertools.combinations(range(dataset.len()), 2)
		)
		if metric_matrix is not None:
			self.metric_matrix = metric_matrix


	def get(self, index):
		return self.dataset[self.pairs[index][0]], \
			self.dataset[self.pairs[index][1]], \
			self.metric_matrix[self.pairs[index][0], self.pairs[index][1]]


	def len(self):
		return len(self.pairs)


# def collate_fn_pairwise(batch):
# 	"""
# 	Parameters
# 	----------
# 	batch : list
# 		A list of tuples of PyG data objects.
#
# 	Returns
# 	-------
#
#
# 	Reference
# 	---------
# 	https://github.com/priba/siamese_ged/blob/master/datasets/load_data.py#L108C6-L108C6
# 	"""
# 	return batch
# 	n_batch = len(batch)
#
# 	num_nodes1 = torch.LongTensor([x.x.size(0) for x in batch])
# 	num_nodes2 = torch.LongTensor([x[2].size(0) for x in batch])
#
# 	graph_size1 = torch.LongTensor(
# 		[[x[0].size(0), x[0].size(1), x[1].size(2)] for x in batch]
# 	)
# 	graph_size2 = torch.LongTensor(
# 		[[x[2].size(0), x[2].size(1), x[3].size(2)] for x in batch]
# 	)
#
# 	sz1, _ = graph_size1.max(dim=0)
# 	sz2, _ = graph_size2.max(dim=0)
#
# 	n_labels1 = torch.zeros(n_batch, sz1[0], sz1[1])
# 	n_labels2 = torch.zeros(n_batch, sz2[0], sz2[1])
#
# 	am1 = torch.zeros(n_batch, sz1[0], sz1[0], sz1[2])
# 	am2 = torch.zeros(n_batch, sz2[0], sz2[0], sz2[2])
#
# 	targets = torch.cat([x[-1] for x in batch])
#
# 	for i in range(n_batch):
# 		# Node Features
# 		n_labels1[i, :num_nodes1[i], :] = batch[i][0]
# 		n_labels2[i, :num_nodes2[i], :] = batch[i][2]
#
# 		# Adjacency matrix
# 		am1[i, :num_nodes1[i], :num_nodes1[i], :] = batch[i][1]
# 		am2[i, :num_nodes2[i], :num_nodes2[i], :] = batch[i][3]
#
# 	return n_labels1, am1, num_nodes1, n_labels2, am2, num_nodes2, targets
