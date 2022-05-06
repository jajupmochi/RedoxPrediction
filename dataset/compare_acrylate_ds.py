#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:34:16 2022

@author: ljia
"""
def load_acrylate_ds_from_pdf(fname):
	import fitz

	# Read pdf file.
	with fitz.open(fname) as f:
		text = ''
		for page in f:
			text += page.get_text()

	# Get the data part text.
	idx_start = text.find('\nTg (K)') + 7
	idx_end = text.find('\n2 Control group')
	if idx_end == -1:
		idx_end = text.find('\n2“Weighted” outputs')
	text = text[idx_start:idx_end].strip().split('\n')

	# Format data.
	data_tmp = {'X': [], 'targets': []}
	for i in range(0, len(text), 2):
		data_tmp['X'].append(text[i].strip())
		data_tmp['targets'].append(float(text[i + 1].strip()))

	# Remove duplicate.
	nb_duplicate = 0 # the same name and Tg
	nb_same_name = 0 # the same name but different Tg s
	data = {'X': [], 'targets': []}
	for i in range(0, len(data_tmp['X'])):
		if data_tmp['X'][i] in data['X']:
			idx_s = data['X'].index(data_tmp['X'][i])
			if data_tmp['targets'][i] == data['targets'][idx_s]:
				nb_duplicate += 1
				continue
			else:
				print('Molecules of #%d and #%d have the same name but different '
		  'temps: %s, %.1fK, %.1fK' % (idx_s, len(data['X']), data_tmp['X'][i],
							 data['targets'][idx_s], data_tmp['targets'][i]))
				nb_same_name += 1

		data['X'].append(data_tmp['X'][i])
		data['targets'].append(data_tmp['targets'][i])


	return data


def check_ds_consistence(ds1, ds2):
	if len(ds1['X']) != len(ds2['X']):
		print('The two datasets have different size!')
		return False

	for i in range(len(ds1['X'])):
		if ds1['X'][i] != ds2['X'][i] or ds1['targets'][i] != ds2['targets'][i]:
			print('The two datasets have different terms at %dth mol:' % i)
			print('ds1: X = %s, target = %f;' % (ds1['X'][i], ds1['targets'][i]))
			print('ds2: X = %s, target = %f;' % (ds2['X'][i], ds2['targets'][i]))
			return False

	print('The two datasets are the same!')
	return True


#%%


def compare_acrylate_ds():
	import os
	import sys
	sys.path.insert(1, '../')

	### Load datasets.
	fname_mapping = '../datasets/Acrylates/acrylates_mapping.pdf'
	data_mapping = load_acrylate_ds_from_pdf(fname_mapping)

	fname_localizing = '../datasets/Acrylates/acrylates_localizing.pdf'
	data_localizing = load_acrylate_ds_from_pdf(fname_localizing)

	if_same = check_ds_consistence(data_mapping, data_localizing)


#%%


def check_acrylate_in_poly200():
	import os
	import sys
	sys.path.insert(1, '../')

	### Load datasets.
	# acrylate
	fname_mapping = '../datasets/Acrylates/acrylates_mapping.pdf'
	data_mapping = load_acrylate_ds_from_pdf(fname_mapping)
	# poly200
	from dataset.load_dataset import load_dataset
	data_poly = load_dataset('polyacrylates200', format_='smiles')


	# Check if each mol in acrylate is also in poly200.
	nb_notin = 0
	for i, v in enumerate(data_mapping['X']):
		try:
			idx = data_poly['X'].index(v)
# 			continue
			if data_poly['targets'][idx] == data_mapping['targets'][i]:
				continue
		except ValueError:
			pass

		print('The molecule of #%d is not in poly200: %s.' % (i, v))
		nb_notin += 1

	return nb_notin


#%%


def load_localizing_from_xlsx(fname):
	import pandas as pd
	df = pd.read_excel(fname)
	X, targets = [], []
	for row in df.iterrows():
		X.append(row[1][7].replace(' ', ''))
		targets.append(row[1][6])
	data = {'X': X, 'targets': targets}
	return data


def compare_poly200_localizing():
	import os
	import sys
	sys.path.insert(1, '../')

	### Load datasets.
	# acrylate (localizing).
	fname_localizing = '../datasets/Acrylates/Tg_list_SMILES_v5_localizing.xlsx'
	data_localizing = load_localizing_from_xlsx(fname_localizing)
	# poly200
	from dataset.load_dataset import load_dataset
	data_poly = load_dataset('polyacrylates200', format_='smiles')

	if_same = check_ds_consistence(data_localizing, data_poly)

#%%


if __name__ == '__main__':
# 	check_acrylate_in_poly200()
 	compare_acrylate_ds()
# 	compare_poly200_localizing()