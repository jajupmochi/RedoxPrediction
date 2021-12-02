#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:39:22 2021

@author: ljia
"""
import os
import pandas as pd


def load_dataset(ds_name, **kwargs):
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
		data = load_qm7(**kwargs)

	elif ds_name.lower() == 'thermophysical':
		data = load_thermophysical(**kwargs)

	return data


def load_thermophysical(path='../datasets/thermophysical/'):
	fname = os.path.join(path, 'thermophysical.csv')

	if not os.path.isfile(fname):
		from fetch_dataset import fetch_dataset
		fetch_dataset('thermophysical', fname=fname)

	df = pd.read_csv(fname)

	return df


if __name__ == '__main__':
	load_dataset('thermophysical')