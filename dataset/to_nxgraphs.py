#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:13:48 2022

@author: ljia
"""

import sys
sys.path.insert(0, '../')
import numpy as np
from dataset.load_dataset import load_dataset


def get_data(ds_name, descriptor='smiles', format_='smiles'):

	def get_poly200(descriptor, format_):
		data = load_dataset('polyacrylates200', descriptor=descriptor, format_=format_)
		smiles = data['X']
		# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
		y = data['targets']
		smiles = [s for i, s in enumerate(smiles) if i not in [6]]
		y = [y for i, y in enumerate(y) if i not in [6]]
		y = np.reshape(y, (len(y), 1))
		families = [ds_name] * len(smiles)
		return smiles, y, families


	def get_sugarmono(descriptor, format_):
		data = load_dataset('sugarmono', descriptor=descriptor, format_=format_)
		smiles = data['X']
		y = data['targets']
		families = data['families']
		idx_rm = [17] # Generated atomic 3d coordinates are all 0s.
		smiles = [s for i, s in enumerate(smiles) if i not in idx_rm]
		y = [y for i, y in enumerate(y) if i not in idx_rm]
		families = [f for i, f in enumerate(families) if i not in idx_rm]
		y = np.reshape(y, (len(y), 1))
		return smiles, y, families

	if ds_name.lower() == 'poly200':
		smiles, y, families = get_poly200(descriptor, format_)

	elif ds_name.lower() == 'sugarmono':
		smiles, y, families = get_sugarmono(descriptor, format_)

	else:
		raise ValueError('Dataset name %s can not be recognized.' % ds_name)


	return smiles, y, families


def smiles_to_nxgraphs(smiles):
	from dataset.feat import DCMolGraphFeaturizer
	featurizer = DCMolGraphFeaturizer(use_edges=True,
								   use_chirality=True,
								   use_partial_charge=False,
								   use_distance_stats=False,
								   use_xyz=False)

	graphs = featurizer.featurize(smiles)

	# to nx graphs.
	import networkx as nx
	for i_g, g in enumerate(graphs):
		g_new = nx.Graph(idx=str(i_g))
		for n, feat in enumerate(g.node_features):
			g_new.add_node(n, feat=feat)

		for i_e in range(0, g.edge_index.shape[1], 2):
			n1, n2 = g.edge_index[0, i_e], g.edge_index[1, i_e]
			g_new.add_edge(n1, n2, feat=g.edge_features[i_e])

		graphs[i_g] = g_new

	return graphs


def draw_molecule_from_smiles(smiles, fig_name='fig.pdf'):
	import matplotlib.pyplot as plt
	import networkx as nx
	from dataset.format_dataset import smiles2nxgraph_rdkit
	graph = smiles2nxgraph_rdkit(smiles, add_hs=False)
	nlabels = nx.get_node_attributes(graph, 'symbol')
	elabels = nx.get_edge_attributes(graph, 'bond_type_double')

	pos = nx.spring_layout(graph)
	nx.draw(graph, pos)
	nx.draw_networkx_labels(graph, pos, labels=nlabels)
	nx.draw_networkx_edge_labels(graph, pos, edge_labels=elabels)
	plt.savefig(fig_name)
	plt.show()

# 	nx.draw(graph, labels=nlabels, edge_labels=elabels)
# 	plt.savefig(fig_name)
# 	plt.show()

if __name__ == '__main__':
	import pickle

	### Load dataset.
	for ds_name in ['poly200', 'sugarmono']:
		smiles, y, families = get_data(ds_name, descriptor='smiles')
		graphs = smiles_to_nxgraphs(smiles)

		pickle.dump((graphs, y, families), open(ds_name + '.pkl', 'wb'))

		### read.
		dataset = pickle.load(open(ds_name + '.pkl', 'rb'))
