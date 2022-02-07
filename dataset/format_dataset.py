#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:48:39 2021

@author: ljia
"""
import numpy as np


def to_smiles(data, ds_name, **kwargs):
	if ds_name == 'thermophysical':
		t_type = (kwargs['t_type'] if 't_type' in kwargs else 'cal')
		rm_replicate = (kwargs['rm_replicate'] if 'rm_replicate' in kwargs else True)
		return thermophysical_to_smiles(data, t_type=t_type, rm_replicate=rm_replicate)

	elif ds_name == 'polyacrylates200':
		return polyacrylates200_to_smiles(data)


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


def polyacrylates200_to_smiles(data):

	dataset = {'X': [], 'targets': []}

	dataset['X'] = [i.replace(' ', '') for i in data.iloc[:, 3].to_list()]
	dataset['targets'] = [float(i) for i in data.iloc[:, 2].to_list()]

	return dataset