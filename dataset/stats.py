#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:15:31 2021

@author: ljia
"""
import numpy as np

import sys
sys.path.insert(0, '../')



def get_atom_types(ds_name, **kwargs):
	from dataset.load_dataset import load_dataset
	dataset = load_dataset(ds_name, format_='networkx', **kwargs)

	atom_types = []
	for graph in dataset['X']:
		for n in graph.nodes(data=True):
			if n[1]['symbol'] not in atom_types:
				atom_types.append(n[1]['symbol'])
	print('Atom types: ', atom_types)

def get_rmse_thermophysical():
	"""Get rmse between experimental and computed Tg of the thermophysical dataset.


	Returns
	-------
	None.

	"""
	from load_dataset import load_thermophysical

	df = load_thermophysical()

	exp = df.iloc[:, 5]
	cal = df.iloc[:, 7]

	# Remove useless lines.
	exp_cln = []
	cal_cln = []
	for idx, e in enumerate(exp):
		try:
			ef = float(e)
			cf = float(cal[idx])
		except ValueError:
# 			raise
			pass
		else:
			if not np.isnan(ef) 	and not np.isnan(cf):
				exp_cln.append(ef)
				cal_cln.append(cf)


	# Compute rmse.
	rmse = np.sqrt(((np.array(exp_cln) - np.array(cal_cln)) ** 2).mean())
	from sklearn.metrics import mean_squared_error, r2_score
	rmse2 = mean_squared_error(exp_cln, cal_cln, squared=False)
	r2 = r2_score(exp_cln, cal_cln)
	print(rmse, ', ', rmse2)
	print(r2)

	perc = rmse / np.mean(exp_cln + cal_cln)

	return rmse, perc


if __name__ == '__main__':
# 	###
# 	rmse, perc = get_rmse_thermophysical()

	###
# 	ds_name = 'polyacrylates200'
	ds_name = 'sugarmono'
	get_atom_types(ds_name)

