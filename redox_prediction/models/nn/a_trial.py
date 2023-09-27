"""
test



@Author: linlin
@Date: 26.05.23
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx


class NetworkXGraphDataset(Dataset):
	def __init__(
			self, graph_list, target_list,
			node_label_names=None, edge_label_names=None,
			node_attr_names=None, edge_attr_names=None
	):
		"""
		Initialize a PyTorch Geometric compatible dataset from a list of NetworkX
		graphs.

		parameters
		----------
		graph_list : list
			List of NetworkX graphs.
		target_list : list
			List of targets.
		node_label_names : list
			List of symbolic node attribute names to be encoded as one-hot.
		edge_label_names : list
			List of symbolic edge attribute names to be encoded as one-hot.
		node_attr_names : list
			List of node attribute names to be included in the node feature matrix.
		edge_attr_names : list
			List of edge attribute names to be included in the edge feature matrix.
		"""
		super(NetworkXGraphDataset, self).__init__()

		# Get node and edge attribute names:
		node_label_names, edge_label_names = self.get_attribute_names(
			graph_list,
			node_attr_names=node_label_names,
			edge_attr_names=edge_label_names
		)

		# Get all unique node and edge attributes in the label names:
		unique_node_attrs, unique_edge_attrs = self.get_all_attributes(
			graph_list,
			node_attr_names=node_label_names,
			edge_attr_names=edge_label_names
		)

		self.data_list = []
		for graph, target in zip(graph_list, target_list):
			# Convert each NetworkX graph to a PyTorch Geometric Data object
			data = from_networkx(graph)

			# Process node labels:
			# Initialize x as an empty tensor:
			if len(node_label_names) != 0:
				data.x = torch.empty((data.num_nodes, 0), dtype=torch.float)
			# Concatenate the node feature matrix with the node labels:
			for attr_name in node_label_names:
				num_classes = len(unique_node_attrs[attr_name]) + 1
				# Convert symbolic node labels to one-hot encodings:
				encoding = F.one_hot(data[attr_name], num_classes=num_classes)
				# Concatenate the one-hot encodings to the node feature matrix:
				data.x = torch.cat((data.x, encoding), dim=-1)
			# Concatenate the node feature matrix with the node attributes:
			for attr_name in node_attr_names:
				data.x = torch.cat((data.x, data[attr_name]), dim=-1)

			# Process edge labels:
			# Initialize edge_attr as an empty tensor:
			if len(edge_label_names) != 0:
				data.edge_attr = torch.empty(
					(data.num_edges, 0), dtype=torch.float
				)
			# Concatenate the edge feature matrix with the edge labels:
			for attr_name in edge_label_names:
				num_classes = len(unique_edge_attrs[attr_name]) + 1
				# Convert symbolic edge labels to one-hot encodings:
				encoding = F.one_hot(data[attr_name], num_classes=num_classes)
				# Concatenate the one-hot encodings to the edge feature matrix:
				data.edge_attr = torch.cat((data.edge_attr, encoding), dim=-1)
			# Concatenate the edge feature matrix with the edge attributes:
			for attr_name in edge_attr_names:
				data.edge_attr = torch.cat(
					(data.edge_attr, data[attr_name]), dim=-1
				)

			# Add the target to the data object:
			data.y = torch.tensor([target], dtype=torch.long)
			self.data_list.append(data)

		# Store the node and edge label names:
		self.node_label_names = node_label_names
		self.edge_label_names = edge_label_names
		# Store the node and edge attribute names:
		self.node_attr_names = node_attr_names
		self.edge_attr_names = edge_attr_names


	def len(self):
		return len(self.data_list)


	def get(self, idx):
		return self.data_list[idx]


	def get_attribute_names(
			graph_list, node_attr_names=None, edge_attr_names=None
	):
		if node_attr_names is None:
			node_attr_names = []
		elif node_attr_names == 'all':
			node_attr_names = list(
				set(
					[attr for graph in graph_list for node, attr in
					 graph.nodes(data=True)]
				)
			)
		if edge_attr_names is None:
			edge_attr_names = []
		elif edge_attr_names == 'all':
			edge_attr_names = list(
				set(
					[attr for graph in graph_list for edge, attr in
					 graph.edges(data=True)]
				)
			)
		return node_attr_names, edge_attr_names


	def get_all_attributes(
			graph_list, node_attr_names=None, edge_attr_names=None
	):
		"""
		Return all unique attributes on nodes and edges in a list of graphs,
		given a list of attribute names. If an attribute name list is None,
		that dict is set to empty. If an attribute name list is 'all', return all
		attributes in that dict.

		Args:
			graph_list: a list of NetworkX graphs
			node_attr_names: a list of node attribute names
			edge_attr_names: a list of edge attribute names

		Returns:
			node_attr_dict: a dictionary of node attributes
			edge_attr_dict: a dictionary of edge attributes
		"""
		node_attr_dict = {}
		edge_attr_dict = {}

		node_attr_names, edge_attr_names = self.get_attribute_names(
			graph_list, node_attr_names, edge_attr_names
		)

		for attr_name in node_attr_names:
			node_attr_dict[attr_name] = set(
				[attr[attr_name] for graph in graph_list for node, attr in
				 graph.nodes(data=True)]
			)
		for attr_name in edge_attr_names:
			edge_attr_dict[attr_name] = set(
				[attr[attr_name] for graph in graph_list for edge, attr in
				 graph.edges(data=True)]
			)

		return node_attr_dict, edge_attr_dict


if __name__ == '__main__':

	# from gklearn.dataset import Dataset
	# from gklearn.experiments import DATASET_ROOT
	#
	# ds = Dataset('MUTAG', root=DATASET_ROOT, verbose=True)
	# Gn = ds.graphs
	# y_all = ds.targets
	# dataset = NetworkXGraphDataset(
	# 	Gn, y_all,
	# 	node_label_names=ds.node_labels, edge_label_names=ds.edge_labels,
	# 	node_attr_names=ds.node_attributes, edge_attr_names=ds.edge_attributes
	# )
	#
	# # Create a PyTorch Geometric data loader
	# batch_size = 32
	# shuffle = True  # Set to False if you want to preserve the order of graphs
	# num_workers = 4  # Number of subprocesses for data loading
	# loader = DataLoader(
	# 	dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
	# )
	#
	# for batch in loader:
	# 	# Access the batched data and targets
	# 	x = batch.x  # Node features
	# 	edge_index = batch.edge_index  # Edge indices
	# 	y = batch.y  # Prediction targets
	#
	# 	# Perform further operations on the batched data and targets
	# 	# ...



	import argparse
	import os.path as osp

	import torch
	import torch.nn.functional as F

	from torch_geometric.datasets import TUDataset
	from torch_geometric.loader import DataLoader
	from torch_geometric.logging import init_wandb, log
	from torch_geometric.nn import MLP, GINConv, global_add_pool

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='MUTAG')
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--hidden_channels', type=int, default=32)
	parser.add_argument('--num_layers', type=int, default=5)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--wandb', action='store_true', help='Track experiment')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	init_wandb(name=f'GIN-{args.dataset}', batch_size=args.batch_size, lr=args.lr,
	           epochs=args.epochs, hidden_channels=args.hidden_channels,
	           num_layers=args.num_layers, device=device)

	path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
	dataset = TUDataset(path, name=args.dataset).shuffle()

	train_dataset = dataset[len(dataset) // 10:]
	train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

	test_dataset = dataset[:len(dataset) // 10]
	test_loader = DataLoader(test_dataset, args.batch_size)


	class Net(torch.nn.Module):
	    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
	        super().__init__()

	        self.convs = torch.nn.ModuleList()
	        for _ in range(num_layers):
	            mlp = MLP([in_channels, hidden_channels, hidden_channels])
	            self.convs.append(GINConv(nn=mlp, train_eps=False))
	            in_channels = hidden_channels

	        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
	                       norm=None, dropout=0.5)

	    def forward(self, x, edge_index, batch):
	        for conv in self.convs:
	            x = conv(x, edge_index).relu()
	        x = global_add_pool(x, batch)
	        return self.mlp(x)


	model = Net(dataset.num_features, args.hidden_channels, dataset.num_classes,
	            args.num_layers).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


	def train():
	    model.train()

	    total_loss = 0
	    for data in train_loader:
	        data = data.to(device)
	        optimizer.zero_grad()
	        out = model(data.x, data.edge_index, data.batch)
	        loss = F.cross_entropy(out, data.y)
	        loss.backward()
	        optimizer.step()
	        total_loss += float(loss) * data.num_graphs
	    return total_loss / len(train_loader.dataset)


	@torch.no_grad()
	def mytest(loader):
	    model.eval()

	    total_correct = 0
	    for data in loader:
	        data = data.to(device)
	        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
	        total_correct += int((pred == data.y).sum())
	    return total_correct / len(loader.dataset)


	for epoch in range(1, args.epochs + 1):
	    loss = train()
	    train_acc = mytest(train_loader)
	    test_acc = mytest(test_loader)
	    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
