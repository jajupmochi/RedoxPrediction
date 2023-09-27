#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:06:33 2022

@author: ljia
"""
import os
import sys
sys.path.insert(0, '../../')
import pickle
import numpy as np
from dataset.load_dataset import get_data


from gklearn.kernels import Treelet
from gklearn.utils.kernels import tanimoto_kernel
from gklearn.utils import get_iters


path_kw = '/treelet_tani_krr/atom_bond_types/std_y/'
# path_kw = '/treelet_tani_krr/1hot/'


#%% Simple test run


def check_treelets(X_train,
				   **kwargs):
	fn_canonkeys = 'canonkeys.pkl'
	if os.path.isfile(fn_canonkeys):
		canonkeys = pickle.load(open(fn_canonkeys, 'rb'))
	else:
		# Initialize model.
		nl_names = list(X_train[0].nodes[list(X_train[0].nodes)[0]].keys())
		el_names = list(X_train[0].edges[list(X_train[0].edges)[0]].keys())
		graph_kernel = Treelet(node_labels=nl_names, edge_labels=el_names,
						   ds_infos={'directed': False},
						   sub_kernel=tanimoto_kernel,
						   normalize=True,
						   verbose=0)

		# train the model by train set
		gram_mat_train = graph_kernel.fit_transform(X_train, save_gm_train=True)
		canonkeys = graph_kernel._canonkeys
		pickle.dump(canonkeys, open(fn_canonkeys, 'wb'))


	# Compute treelet spectrum.
	all_keys = []
	for c in canonkeys:
		all_keys += list(c.keys())
	all_keys = sorted(list(set(all_keys)))

	return

#%%


if __name__ == '__main__':

# 	test_losses()

	### Load dataset.
	for ds_name in ['poly200']: # ['poly200+sugarmono']: ['poly200']

		if 'atom_bond_types' in path_kw:
			### 1. Get descriptors: atom and bond types.
			X, y, families = get_data(ds_name, descriptor='smiles', format_='networkx',
							 add_edge_feats=False)

		elif '1hot' in path_kw:
			### 2. Get descriptors: one-hot descriptors on nodes and edges.
			from dataset.feat import DCMolGraphFeaturizer
			featurizer = DCMolGraphFeaturizer(use_edges=True,
									   use_chirality=True,
									   use_partial_charge=False,
									   use_distance_stats=False,
									   use_xyz=False,
									   return_int=True)
			X, y, families = get_data(ds_name, descriptor=featurizer, format_='networkx')

		check_treelets(X)