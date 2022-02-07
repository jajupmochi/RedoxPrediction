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
from dataset.format_dataset import to_smiles


def load_dataset(ds_name, format_='smiles', **kwargs):
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

	# Format data.
	if format_ == 'smiles':
		data = to_smiles(data, ds_name.lower(), **kwargs)

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


if __name__ == '__main__':
	dataset = load_dataset('thermophysical')
	dataset2 = load_dataset('polyacrylates200')

	# Check overlaps in the two datasets.
	dataset['X'] = list(set(dataset['X']))
	dataset2['X'] = list(set(dataset2['X']))
	overlaps = []
	exclusive_1 = []
	for s in dataset['X']:
		if s in dataset2['X']:
			overlaps.append(s)
		else:
			exclusive_1.append(s)
	exclusive_2 = []
	for s in dataset2['X']:
		if s not in dataset['X']:
			exclusive_2.append(s)