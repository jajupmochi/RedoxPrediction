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
		kwargs_data = ({'path': kwargs['path']} if 'path' in kwargs else {})
		data = load_polyacrylates200(**kwargs_data)

	elif ds_name.lower() == 'thermophysical':
		kwargs_data = ({'path': kwargs['path']} if 'path' in kwargs else {})
		data = load_thermophysical(**kwargs_data)

	elif ds_name.lower() == 'sugarmono':
		kwargs_data = ({'path': kwargs['path']} if 'path' in kwargs else {})
		data = load_sugarmono(**kwargs_data)

	# Format data.
	if format_.lower() == 'smiles':
		from dataset.format_dataset import to_smiles
		data = to_smiles(data, ds_name.lower(), **kwargs)

	elif format_.lower() == 'networkx':
		from dataset.format_dataset import to_nxgraphs
		data = to_nxgraphs(data, ds_name.lower(), **kwargs)

	return data


def load_thermophysical(path='../datasets/Thermophysical/'):
	fname = os.path.join(path, 'thermophysical.csv')

	if not os.path.isfile(fname):
		from fetch_dataset import fetch_dataset
		fetch_dataset('thermophysical', fname=fname)

	df = pd.read_csv(fname)

	return df


def load_polyacrylates200(path='../datasets/Polyacrylates200/'):
	fname = os.path.join(path, 'polyacrylates200.csv')

	df = pd.read_csv(fname)

	return df


def load_sugarmono(path='../datasets/Sugarmono/', rm_replicate=False):
	### Load raw data from file.
	fname = os.path.join(path, 'oxygenated_polymers.csv')
	df = pd.read_csv(fname, header=None)

	### Retrieve dataset from the raw data.
	smiles = df.iloc[:, 2]
	targets = df.iloc[:, 6]
	families = df.iloc[:, 5]

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
				dataset['X'].append(smiles[idx].strip())
				dataset['targets'].append(tf)
				dataset['families'].append(families[idx].strip())

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