#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:48:39 2021

@author: ljia
"""
import os
import numpy as np


def to_rdkit_mols(data, descriptor='smiles+xyz_obabel',
				  ds_name='', ds_dir='', **kwargs):
	if descriptor in ['smiles+dis_stats_obabel', 'smiles+xyz_obabel']:
		mols = []
		data = to_smiles(data, ds_name.lower(), **kwargs)
		smiles = data['X']

		coords_dir = os.path.join(ds_dir, 'obabel.sdf')
		os.makedirs(coords_dir, exist_ok=True)
		# for each smiles, labelings starting from 1.
		for i, sm in enumerate(smiles):
			# Compute 3d coordinates and generate the .sdf file by OpenBabel
			# command line tools if the file does not exist.
# 			print(i)
			fn_sdf = os.path.join(coords_dir, 'out' + str(i + 1) + '.sdf')
			if not os.path.isfile(fn_sdf):
				command = 'obabel -:"' + sm + '" -O "' + fn_sdf + '" -h --gen3d'
				os.system(command)

			# Create the rdkit mol from the .sdf file.
			from rdkit import Chem
			# Add Hydrogens.
# 			with Chem.SDMolSupplier(fn_sdf, removeHs=False) as suppl:
# 				mol = suppl[0]
			mol = Chem.SDMolSupplier(fn_sdf, removeHs=False)[0]
# 				if mol is None:
# 					raise Exception('Molecule # %d can not be loaded as a rdkit mol.' % (i + 1))
# 			for a in mol.GetAtoms():
# 			    print(a.GetSymbol())

			mols.append(mol)

	data['X'] = mols
	return data


def to_nxgraphs(data, ds_name,
				descriptor='none',
				add_hs=False,
				verbose=True,
				**kwargs):

	data_tmp = to_smiles(data, ds_name, **kwargs)


	### Convert smiles to nxgraphs.
	## if descriptor is a predefined string:
	if isinstance(descriptor, str):
		add_edge_feats = kwargs.get('add_edge_feats', True)
		ds_size = len(data_tmp['X'])
		X, idx_true = [], []
		for i in range(ds_size):
			try:
				g = smiles2nxgraph_rdkit(data_tmp['X'][i],
							 add_edge_feats=add_edge_feats,
							 add_hs=add_hs)
			except AttributeError:
				pass
			else:
				X.append(g)
				idx_true.append(i)

		if verbose and (ds_size != len(idx_true)):
			print('%d graph(s) are removed due to unrecognized smiles.' %
				(ds_size - len(idx_true)))

		# Save nxgraphs to data.
		data = {'X': X}
		for k in data_tmp.keys():
			if k != 'X' and len(data_tmp['X']) == len(data_tmp[k]):
				data[k] = [data_tmp[k][i] for i in idx_true]


	## if descriptor is a provided featurizer:
	else:
		# Remove smiles that can not be handled by RDKit.
		# @todo: this maybe a better way.
		from rdkit import Chem
		ds_size = len(data_tmp['X'])
		X, idx_true = [], []
		for i in range(ds_size):
			mol = Chem.MolFromSmiles(data_tmp['X'][i])
			if mol is not None:
				X.append(mol)
				idx_true.append(i)

		if verbose and (ds_size != len(idx_true)):
			print('%d graph(s) are removed due to unrecognized smiles.' %
				(ds_size - len(idx_true)))

		# Save valid SMILES:
		data = {'X': X}
		for k in data_tmp.keys():
			if k != 'X' and len(data_tmp['X']) == len(data_tmp[k]):
				data[k] = [data_tmp[k][i] for i in idx_true]

		# Featurize:
		graphs = descriptor.featurize(data['X'])

		# to nx graphs:
		import networkx as nx
		for i_g, g in enumerate(graphs):
			g_new = nx.Graph(idx=str(i_g))
			for n, feat in enumerate(g.node_features):
				g_new.add_node(n, **{str(i) : v for i, v in enumerate(feat)})

			for i_e in range(0, g.edge_index.shape[1], 2):
				n1, n2 = g.edge_index[0, i_e], g.edge_index[1, i_e]
				g_new.add_edge(n1, n2, **{str(i) : v for i, v in enumerate(g.edge_features[i_e])})

			data['X'][i_g] = g_new


	return data


def to_smiles(data, ds_name, **kwargs):
	if ds_name == 'thermophysical':
		t_type = kwargs.get('t_type', 'cal')
		rm_replicate = kwargs.get('rm_replicate', True)
		return thermophysical_to_smiles(data, t_type=t_type, rm_replicate=rm_replicate)

	elif ds_name == 'polyacrylates200':
		return data

	elif ds_name.lower() == 'sugarmono':
		return data


def thermophysical_to_smiles(data, t_type='cal', rm_replicate=True):

	dataset = {'X': [], 'targets': []}

	smiles = data.iloc[:, 2]
	if t_type == 'exp':
		tgt = data.iloc[:, 5]
	elif t_type == 'cal':
		tgt = data.iloc[:, 7]

	# Get data while removing useless lines.
	for idx, t in enumerate(tgt):
		try:
			tf = float(t)
		except ValueError:
# 			raise
			pass
		else:
			if not np.isnan(tf) and isinstance(smiles[idx], str):
				dataset['X'].append(smiles[idx])
				dataset['targets'].append(tf)

	if rm_replicate:
		X, targets = [], []
		for idx, s in enumerate(dataset['X']):
			if s not in X:
				X.append(s)
				targets.append(dataset['targets'][idx])
		dataset['X'] = X
		dataset['targets'] = targets

	return dataset


#%%


def smiles2nxgraph_rdkit(smiles_string, add_edge_feats=True, add_hs=False):
	"""Converts SMILES string to NetworkX graph object by RDKit library.

	Parameters
	----------
	smiles_string : string
		SMILES string.

	Returns
	-------
	Graph object.
	"""
	from rdkit import Chem
	mol = Chem.MolFromSmiles(smiles_string)
	if add_hs:
		mol = Chem.AddHs(mol)
	mol = mol_to_nx(mol, add_edge_feats=add_edge_feats)
	return mol


def mol_to_nx(mol, add_edge_feats=True):
	"""Converts RDKit mol to NetworkX graph object.

	Parameters
	----------
	mol : RDKit mol
		RDKit molecule.

	Returns
	-------
	G : Networkx Graph.
		Graph object.

	References
	----------
	`keras-molecules <https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py#L17>`__
	"""
	import networkx as nx
	G = nx.Graph()

	for atom in mol.GetAtoms():
		G.add_node(atom.GetIdx(),
			       symbol=atom.GetSymbol(),
# 				   atomic_num=atom.GetAtomicNum(),
# 				   formal_charge=atom.GetFormalCharge(),
# 				   chiral_tag=atom.GetChiralTag(),
# 				   hybridization=atom.GetHybridization(),
# 				   num_explicit_hs=atom.GetNumExplicitHs(),
# 				   is_aromatic=atom.GetIsAromatic()
				   )
	if add_edge_feats:
		for bond in mol.GetBonds():
			G.add_edge(bond.GetBeginAtomIdx(),
					   bond.GetEndAtomIdx(),
	   				   bond_type_double=str(bond.GetBondTypeAsDouble()))
	# 				   bond_type=bond.GetBondType())
	else:
		for bond in mol.GetBonds():
			G.add_edge(bond.GetBeginAtomIdx(),
					   bond.GetEndAtomIdx(),
	   				   bond_type_double='0')
	# 				   bond_type=bond.GetBondType())

	return G
