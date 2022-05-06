#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:39:22 2021

@author: ljia
"""
import os
import sys
import pandas as pd
sys.path.insert(0, '../')
import inspect


def load_dataset(ds_name, descriptor='smiles', format_='smiles', **kwargs):
	"""Load pre-defined datasets.

	Parameters
	----------
	ds_name : string
		The name of the dataset. Case insensitive.

	**kwargs : keyword arguments
		Auxilary arguments.

	Returns
	-------
	data : uncertain (depends on the dataset)
		The loaded molecules.

	"""
	if ds_name.lower() == '' or ds_name.lower() == 'polyacrylates200':

		kwargs_data = {key: kwargs[key] for key in kwargs.keys()
				 & set(inspect.signature(load_polyacrylates200).parameters)}
		data = load_polyacrylates200(**kwargs_data)

	elif ds_name.lower() == 'sugarmono':
		kwargs_data = {key: kwargs[key] for key in kwargs.keys()
				 & set(inspect.signature(load_sugarmono).parameters)}
		data = load_sugarmono(**kwargs_data)

	elif ds_name.lower() == 'thermophysical':
		kwargs_data = ({'path': kwargs['path']} if 'path' in kwargs else {})
		data = load_thermophysical(**kwargs_data)

	# Format data.
	if format_.lower() == 'smiles':
		from dataset.format_dataset import to_smiles
		data = to_smiles(data, ds_name.lower(), **kwargs)

	elif format_.lower() == 'networkx':
		from dataset.format_dataset import to_nxgraphs
		data = to_nxgraphs(data, ds_name.lower(), descriptor=descriptor, **kwargs)

	elif format_.lower() == 'rdkit':
		ds_dir = get_ds_dir(ds_name.lower())
		from dataset.format_dataset import to_rdkit_mols
		data = to_rdkit_mols(data, ds_name=ds_name.lower(),
					   descriptor=descriptor,
					   ds_dir=ds_dir, **kwargs)

	return data


def get_ds_dir(ds_name):
	# Get folder name.
	if ds_name in ['', 'polyacrylates200', 'poly200']:
		folder_name = 'Polyacrylates200'
	elif ds_name == 'sugarmono':
		folder_name = 'Sugarmono'

	cur_path = os.path.dirname(os.path.abspath(__file__))
	ds_dir = os.path.join(cur_path, '../datasets/' + folder_name + '/')

	return ds_dir


def load_thermophysical(path='../datasets/Thermophysical/'):
	fname = os.path.join(path, 'thermophysical.csv')

	if not os.path.isfile(fname):
		from fetch_dataset import fetch_dataset
		fetch_dataset('thermophysical', fname=fname)

	df = pd.read_csv(fname)

	return df


def load_polyacrylates200(path='../datasets/Polyacrylates200/',
						  temp_unit='K',
						  with_names=False):
	### Load raw data from file.
	fname = os.path.join(path, 'polyacrylates200.csv')
	df = pd.read_csv(fname)

	### Retrieve dataset from the raw data.
	dataset = {}
	dataset['X'] = [i.replace(' ', '') for i in df.iloc[:, 3].to_list()]
	dataset['targets'] = [float(i) for i in df.iloc[:, 2].to_list()]
	if temp_unit == 'C':
		dataset['targets'] = [t - 273.15 for t in dataset['targets']]
	if with_names:
		dataset['names'] = [i.strip() for i in df.iloc[:, 1].to_list()]

	return dataset


def load_sugarmono(path='../datasets/Sugarmono/',
				   temp_unit='K',
				   rm_replicate=False):
	### Load raw data from file.
	fname = os.path.join(path, 'oxygenated_polymers.csv')
	df = pd.read_csv(fname, header=None)

	### Retrieve dataset from the raw data.
	smiles = df.iloc[:, 2]
	targets = df.iloc[:, 6]
	families = df.iloc[:, 5]
	mononer2 = df.iloc[:, 4]

	# Save data to dataset while removing useless lines.
	import numpy as np
	dataset = {'X': [], 'targets': [], 'families': []}
	for idx, t in enumerate(targets):
		try:
			tf = float(t)
		except ValueError:
# 			raise
			pass
		else:
			if not np.isnan(tf) and isinstance(smiles[idx], str):
				# Remove the polymers (the combination of 2 monomers).
				mono2 = mononer2[idx]
				if pd.isnull(mono2) or mono2.strip() == '':
					dataset['X'].append(smiles[idx].strip())
					dataset['targets'].append(tf)
					# @todo: Make sure that family names are case sensitive.
					dataset['families'].append(families[idx].strip())

	# Convert temperature from C to K.
	if temp_unit == 'K':
		dataset['targets'] = [t + 273.15 for t in dataset['targets']]

	if rm_replicate:
		X, targets, families = [], [], []
		for idx, s in enumerate(dataset['X']):
			if s not in X:
				X.append(s)
				targets.append(dataset['targets'][idx])
				families.append(dataset['families'][idx])
		dataset['X'] = X
		dataset['targets'] = targets
		dataset['families'] = families

	return dataset


#%%

import numpy as np


def get_data(ds_name, descriptor='smiles', format_='smiles', **kwargs):

	def get_poly200(descriptor, format_):
		data = load_dataset('polyacrylates200', descriptor=descriptor, format_=format_, **kwargs)
		smiles = data['X']
		# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
		y = data['targets']
		if format_ == 'smiles':
			smiles = [s for i, s in enumerate(smiles) if i not in [6]]
			y = [y for i, y in enumerate(y) if i not in [6]]
		y = np.reshape(y, (len(y), 1))
		families = [ds_name] * len(smiles)
		return smiles, y, families


	def get_sugarmono(descriptor, format_):
		data = load_dataset('sugarmono', descriptor=descriptor, format_=format_, **kwargs)
		smiles = data['X']
		y = data['targets']
		families = data['families']
		if format_ == 'smiles':
			idx_rm = [17] # Generated atomic 3d coordinates are all 0s.
			smiles = [s for i, s in enumerate(smiles) if i not in idx_rm]
			y = [y for i, y in enumerate(y) if i not in idx_rm]
			families = [f for i, f in enumerate(families) if i not in idx_rm]
		y = np.reshape(y, (len(y), 1))
		return smiles, y, families


	if isinstance(descriptor, str) and descriptor.lower() in ['smiles+dis_stats_obabel', 'smiles+xyz_obabel']:
		format_ = 'rdkit'

	if ds_name.lower() == 'poly200':
		smiles, y, families = get_poly200(descriptor, format_)

	elif ds_name.lower() == 'thermo_exp':
		data = load_dataset('thermophysical', descriptor=descriptor, format_=format_, t_type='exp', **kwargs)
		smiles = data['X']
		y = data['targets']
		idx_rm = [168, 169, 170, 171, 172]
		smiles = [s for i, s in enumerate(smiles) if i not in idx_rm]
		y = [y for i, y in enumerate(y) if i not in idx_rm]
# 		import deepchem as dc
# 		featurizer = dc.feat.MolGraphConvFeaturizer()
# 		X_app = featurizer.featurize(smiles)
		y = np.reshape(y, (len(y), 1))
		families = [ds_name] * len(smiles)

	elif ds_name.lower() == 'thermo_cal':
		data = load_dataset('thermophysical', descriptor=descriptor, format_=format_, t_type='cal', **kwargs)
		smiles = data['X']
		y = data['targets']
		idx_rm = [151, 198, 199]
		smiles = [s for i, s in enumerate(smiles) if i not in idx_rm]
		y = [y for i, y in enumerate(y) if i not in idx_rm]
		y = np.reshape(y, (len(y), 1))
		families = [ds_name] * len(smiles)

	elif ds_name.lower() == 'sugarmono':
		smiles, y, families = get_sugarmono(descriptor, format_)

	elif ds_name.lower() == 'poly200+sugarmono':
		smiles1, y1, families1 = get_poly200(descriptor, format_)
		smiles2, y2, families2 = get_sugarmono(descriptor, format_)
		smiles = smiles1 + smiles2
		y = np.concatenate((y1, y2), axis=0)
		families = families1 + families2

	else:
		raise ValueError('Dataset name %s can not be recognized.' % ds_name)

	return smiles, y, families


#%%


if __name__ == '__main__':
# 	#%%
# 	dataset = load_dataset('thermophysical')
# 	dataset2 = load_dataset('polyacrylates200')

# 	# Check overlaps in the two datasets.
# 	dataset['X'] = list(set(dataset['X']))
# 	dataset2['X'] = list(set(dataset2['X']))
# 	overlaps = []
# 	exclusive_1 = []
# 	for s in dataset['X']:
# 		if s in dataset2['X']:
# 			overlaps.append(s)
# 		else:
# 			exclusive_1.append(s)
# 	exclusive_2 = []
# 	for s in dataset2['X']:
# 		if s not in dataset['X']:
# 			exclusive_2.append(s)


	#%%
	dataset = load_dataset('sugarmono', rm_replicate=False)