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
import warnings

DIR_LOAD_DATASET = os.path.dirname(os.path.abspath(__file__))


def load_dataset(ds_name, descriptor='smiles', format_='smiles', **kwargs):
	"""Load pre-defined datasets.

	Parameters
	----------
	ds_name : string
		The name of the dataset. Case insensitive.

	**kwargs : keyword arguments
		Auxiliary arguments.

	Returns
	-------
	data : uncertain (depends on the dataset)
		The loaded molecules.

	"""
	# Load data in SMILES format.
	if ds_name.lower() == '' or ds_name.lower() == 'polyacrylates200':

		kwargs_data = {key: kwargs[key] for key in kwargs.keys()
		               & set(
			inspect.signature(load_polyacrylates200).parameters
		)}
		data = load_polyacrylates200(**kwargs_data)

	elif ds_name.lower() == 'poly200r':

		kwargs_data = {key: kwargs[key] for key in kwargs.keys()
		               & set(
			inspect.signature(load_polyacrylates200).parameters
		)}
		data = load_poly200r(**kwargs_data)

	elif ds_name.lower() == 'benchmarktao96':

		kwargs_data = {key: kwargs[key] for key in kwargs.keys()
		               & set(
			inspect.signature(load_polyacrylates200).parameters
		)}
		data = load_benchmarktao96(**kwargs_data)

	elif ds_name.lower() == 'sugarmono':
		kwargs_data = {key: kwargs[key] for key in kwargs.keys()
		               & set(inspect.signature(load_sugarmono).parameters)}
		data = load_sugarmono(**kwargs_data)

	elif ds_name.lower() == 'thermophysical':
		kwargs_data = ({'path': kwargs['path']} if 'path' in kwargs else {})
		data = load_thermophysical(**kwargs_data)


	### Redox datasets:

	elif ds_name.lower() in ['brem_togn', 'bremond', 'tognetti']:
		data = load_redox_dataset(
			ds_name, descriptor=descriptor, format_=format_, **kwargs
		)
	# 		return data

	else:
		raise ValueError('Sorry I do not understand this ds_name :(')

	# Format data.
	if format_ is None:  # Return the output format of the featurizer.
		from dataset.format_dataset import to_featurizer_format
		data = to_featurizer_format(
			data, ds_name=ds_name.lower(),
			descriptor=descriptor,
			**kwargs
		)

	elif format_.lower() == 'smiles':
		from dataset.format_dataset import to_smiles
		data = to_smiles(data, ds_name.lower(), **kwargs)

	elif format_.lower() == 'networkx':
		from dataset.format_dataset import to_nxgraphs
		data = to_nxgraphs(
			data, ds_name.lower(), descriptor=descriptor,
			**kwargs
		)

	elif format_.lower() == 'rdkit':
		ds_dir = get_ds_dir(ds_name.lower())
		from dataset.format_dataset import to_rdkit_mols
		data = to_rdkit_mols(
			data, ds_name=ds_name.lower(),
			descriptor=descriptor,
			ds_dir=ds_dir, **kwargs
		)

	elif format_.lower() == 'vector':
		from dataset.format_dataset import to_vectors
		data = to_vectors(
			data, ds_name=ds_name.lower(),
			descriptor=descriptor, **kwargs
		)

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


def load_polyacrylates200(
		path='../datasets/Polyacrylates200/',
		temp_unit='K',
		with_names=False
):
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


def load_poly200r(
		path='../datasets/Poly200R/',
		temp_unit='K',
		with_names=False,
		version='latest'
):
	### Load raw data from file.
	fname = os.path.join(path, 'poly200r' + version + '.csv')
	df = pd.read_csv(
		fname
	)  # Note the first row is automatically removed as header.

	### Retrieve dataset from the raw data.
	dataset = {}
	dataset['X'] = [i.strip() for i in df.iloc[:, 2].to_list()]
	dataset['targets'] = [float(i) for i in df.iloc[:, 4].to_list()]
	if temp_unit == 'C':
		dataset['targets'] = [t - 273.15 for t in dataset['targets']]
	dataset['familes'] = [i.strip() for i in df.iloc[:, 6].to_list()]
	if with_names:
		dataset['names'] = [i.strip() for i in df.iloc[:, 1].to_list()]

	return dataset


def load_benchmarktao96(
		path='../datasets/BenchmarkTao96/',
		temp_unit='K',
		with_md_tg=False
):
	### Load raw data from file.
	fname = os.path.join(path, 'benchmark_tao_96.csv')
	df = pd.read_csv(
		fname
	)  # Note the first row is automatically removed as header.

	### Retrieve dataset from the raw data.
	dataset = {}
	dataset['X'] = [i.strip() for i in df.iloc[:, 0].to_list()]
	dataset['targets'] = [float(i) for i in df.iloc[:, 1].to_list()]
	if temp_unit == 'C':
		dataset['targets'] = [t - 273.15 for t in dataset['targets']]
	dataset['familes'] = ['noidea'] * len(dataset['X'])
	if with_md_tg:
		dataset['md_tg'] = [float(i) for i in df.iloc[:, 2].to_list()]

	return dataset


def load_sugarmono(
		path='../datasets/Sugarmono/',
		temp_unit='K',
		rm_replicate=False
):
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


# %%

def load_redox_dataset(
		ds_name, descriptor='smiles', format_='networkx', **kwargs
):
	"""Load a pre-defined Redox dataset.

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
	# ## Load data.
	if ds_name.lower() == 'brem_togn':
		dataset = load_bremond(descriptor=descriptor, **kwargs)
		dataset2 = load_tognetti(descriptor=descriptor, **kwargs)
		dataset['X'] += dataset2['X']
		dataset['targets'] += dataset2['targets']
		dataset['families'] += dataset2['families']

	elif ds_name.lower() == 'bremond':
		dataset = load_bremond(descriptor=descriptor, **kwargs)

	elif ds_name.lower() == 'tognetti':
		dataset = load_tognetti(descriptor=descriptor, **kwargs)

	elif ds_name.lower() == 'divergent':
		pass

	else:
		raise ValueError(
			'Dataset %s can not be recognized. The possible candidates include '
			'"brem_togn", "bremond", "tognetti", "divergent".'
		)

	# 	### Format data.
	# 	if descriptor == 'smiles':
	# 		if format_ == 'networkx':
	# 			sys.path.insert(0, '../dataset/')
	# 			from format_dataset import smiles_to_nxgraphs
	# 			dataset['X'] = smiles_to_nxgraphs(dataset['X'])
	# 		elif format_ == 'smiles':
	# 			pass
	# 	elif descriptor == 'xyz':
	# 		if format_ == 'networkx':
	# 			sys.path.insert(0, '../dataset/')
	# 			from format_dataset import xyz_to_nxgraphs
	# 			dataset['X'] = xyz_to_nxgraphs(dataset['X'])

	return dataset


def load_bremond(descriptor='smiles', **kwargs):
	subdir_db = '/../datasets/Redox/db_bremond/'
	kwargs_data = {}
	kwargs_data['path'] = kwargs.get('path', DIR_LOAD_DATASET + subdir_db)
	kwargs_data['fn_targets'] = kwargs.get(
		'fn_targets', DIR_LOAD_DATASET + subdir_db + '/pbe0_dG.csv'
	)
	kwargs_data['fn_families'] = kwargs.get(
		'fn_families', DIR_LOAD_DATASET + subdir_db + '/redox_family.csv'
	)
	kwargs_data['level'] = kwargs.get('level', 'pbe0')
	kwargs_data['target_name'] = kwargs.get('target', 'dGred')
	kwargs_data['sort_by_targets'] = kwargs.get('sort_by_targets', False)

	return _load_bremond_or_tognetti(descriptor=descriptor, **kwargs_data)


def load_tognetti(descriptor='smiles', **kwargs):
	subdir_db = '/../datasets/Redox/db_tognetti/'
	kwargs_data = {}
	kwargs_data['path'] = kwargs.get('path', DIR_LOAD_DATASET + subdir_db)
	kwargs_data['fn_targets'] = kwargs.get(
		'fn_targets', DIR_LOAD_DATASET + subdir_db + '/pbe0_dG.csv'
	)
	kwargs_data['fn_families'] = kwargs.get(
		'fn_families', DIR_LOAD_DATASET + subdir_db + '/pbe0_dG_family.csv'
	)
	kwargs_data['level'] = kwargs.get('level', 'pbe0')
	kwargs_data['target_name'] = kwargs.get('target', 'dGred')
	kwargs_data['sort_by_targets'] = kwargs.get('sort_by_targets', False)

	return _load_bremond_or_tognetti(descriptor=descriptor, **kwargs_data)


def _load_bremond_or_tognetti(
		path=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/',
		fn_targets=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/pbe0_dG.csv',
		fn_families=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/redox_family.csv',
		descriptor='smiles',
		level='pbe0',
		target_name='dGred',
		sort_by_targets=False
):
	### Return SMILES format in default.

	if descriptor == 'xyz':
		dataset = _load_bremond_or_tognetti_xyz(
			path, fn_targets, fn_families,
			descriptor,
			level,
			target_name,
			sort_by_targets
		)
	else:  ## if descriptor == 'smiles':
		dataset = _load_bremond_or_tognetti_smiles(
			path, fn_targets, fn_families,
			descriptor,
			level,
			target_name,
			sort_by_targets
		)

	return dataset


def _load_bremond_or_tognetti_smiles(
		path=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/',
		fn_targets=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/pbe0_dG.csv',
		fn_families=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/redox_family.csv',
		descriptor='smiles',
		level='pbe0',
		target_name='dGred',
		sort_by_targets=False
):
	dir_smiles = os.path.join(path, 'uff_from_pbe0/')  # path of SMILES files.

	### Sort data the by target file.
	if sort_by_targets:
		# Get molecule names from the target file.
		names, targets = _load_bremond_or_tognetti_target_file(
			fn_targets, target_name
		)
		# Get families df.
		if fn_families is not None:
			df_families = pd.read_csv(fn_families)
			family_list = []
			col_fam = \
				np.ravel(np.where(df_families.columns.values == 'family'))[0]
		else:
			family_list = [path.strip('/').split('/')[-1]] * len(names)

		# Iterate names.
		X = []

		for idx, nm in enumerate(names):
			# Retrieve SMILES from file.
			try:
				fn_smiles = os.path.join(dir_smiles, nm.strip() + '.smiles')
				with open(fn_smiles, 'r') as f_s:
					content = f_s.read().strip()
					smiles = content.split()[0].strip()
			except:
				print('%d: %s' % (idx, nm))
				raise

			X.append(smiles)

			# Get family from df.
			if fn_families is not None:
				row = df_families.loc[df_families['runs'] == nm]
				if row.shape[0] > 1:
					warnings.warn(
						'There is a duplicate for molecule "' + nm + '".'
					)
				fam = row.iloc[0, col_fam]
				family_list.append(fam)

	else:
		pass

	return {'X': X, 'targets': targets, 'families': family_list}


def _load_bremond_or_tognetti_xyz(
		path=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/',
		fn_targets=DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/pbe0_dG.csv',
		descriptor='smiles',
		level='pbe0',
		target_name='dGred',
		sort_by_targets=False
):
	def _read_xyz_file(fn_xyz):
		xyz = []
		with open(fn_xyz, 'r') as f_s:
			lines = f_s.readlines()
			for l in lines[2:]:
				l_spl = l.split()
				if len(l_spl) == 4:
					xyz.append(
						[l_spl[0]] + [float(l_spl[i]) for i in range(1, 4)]
					)

		return xyz


	if level == 'pbe0':

		dir_smiles = os.path.join(path, 'uff_from_pbe0/')  # path of .xyz files.

		### Sort data the by target file.
		if sort_by_targets:
			# Get molecule names from the target file.
			names, targets = _load_bremond_or_tognetti_target_file(
				fn_targets, target_name
			)

			# Iterate names.
			X = []

			for idx, nm in enumerate(names):
				# Retrieve xyz coordinates from the file.
				try:
					fn_xyz = os.path.join(dir_smiles, nm.strip() + '.xyz')
					xyz = _read_xyz_file(fn_xyz)
				except:
					print('%d: %s' % (idx, nm))
					raise

				X.append(xyz)

		else:
			pass

	elif level == 'obabel':

		dir_smiles = os.path.join(
			path,
			'xyz.obabel.cml/'
		)  # path of .xyz files.

		### Sort data the by target file.
		if sort_by_targets:
			# Get molecule names from the target file.
			names, targets = _load_bremond_or_tognetti_target_file(
				fn_targets, target_name
			)

			# Iterate names.
			X = []

			for idx, nm in enumerate(names):
				# Retrieve xyz coordinates from the file.
				try:
					fn_xyz = os.path.join(
						dir_smiles,
						'out' + str(idx + 1) + '.xyz'
					)
					xyz = _read_xyz_file(fn_xyz)
				except:
					print('%d: %s' % (idx, nm))
					raise

				X.append(xyz)

		else:
			pass

	return {'X': X, 'targets': targets}


def _load_bremond_or_tognetti_target_file(fn_targets, target_name):
	"""Get molecule names from the target file.
	"""
	df_t = pd.read_csv(
		fn_targets,
		header=None
	)  # The header is not used as it may not exist in the .csv file.)
	names = df_t.iloc[:, 0].tolist()  # names of mols.

	# Get targets.
	if target_name == 'dGred':
		targets_tmp = df_t.iloc[:, 3].tolist()
	elif target_name == 'dGox':
		targets_tmp = df_t.iloc[:, 4].tolist()
	else:
		raise ValueError(
			'Target name %s cannot be recognized, possible candidates include "dGred" and "dGox"' % target_name
		)

	# Remove headers if they exist.
	if df_t.iloc[0, 0].strip() == 'runs':
		names = names[1:]  # @todo: to check back
		targets_tmp = targets_tmp[1:]

	targets = [float(i) for i in targets_tmp]

	return names, targets


def family_by_target(ds_name):
	def get_data(target_path, fam_path, i_col_fam):

		import pandas as pd

		# Get molecule names from the target file.
		df_t = pd.read_csv(target_path)
		names = df_t.iloc[:, 0].tolist()

		return load_family(fam_path, mol_names=names, i_col_fam=i_col_fam)


	if ds_name == 'bremond':
		target_path = DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/pbe0_dG.csv'
		fam_path = DIR_LOAD_DATASET + '/../datasets/Redox/low_level/redox_family.csv'
		i_col_fam = 1
		return get_data(target_path, fam_path, i_col_fam)

	elif ds_name == 'tognetti':
		target_path = DIR_LOAD_DATASET + '/../datasets/Redox/db_tognetti/pbe0_dG.csv'
		fam_path = DIR_LOAD_DATASET + '/../datasets/Redox/db_tognetti/pbe0_dG_family.csv'
		i_col_fam = 5
		return get_data(target_path, fam_path, i_col_fam)

	elif ds_name == 'brem_togn':
		target_path = DIR_LOAD_DATASET + '/../datasets/Redox/db_bremond/pbe0_dG.csv'
		fam_path = DIR_LOAD_DATASET + '/../datasets/Redox/low_level/redox_family.csv'
		i_col_fam = 1
		dataset1 = get_data(target_path, fam_path, i_col_fam)

		target_path = DIR_LOAD_DATASET + '/../datasets/Redox/db_tognetti/pbe0_dG.csv'
		fam_path = DIR_LOAD_DATASET + '/../datasets/Redox/db_tognetti/pbe0_dG_family.csv'
		i_col_fam = 5
		dataset2 = get_data(target_path, fam_path, i_col_fam)

		return dataset1 + dataset2


def load_family(path, mol_names=None, i_col_fam=1):
	import pandas as pd

	# Read file.
	df = pd.read_csv(path, header=None, skiprows=[0])

	family_list = []

	if mol_names is None:
		# Iterate rows.
		for index, row in df.iterrows():
			fam = row[1]
			family_list.append(fam)

	else:

		# Iterate names.
		for nm in mol_names:
			row = df.loc[df[0] == nm]
			if row.shape[0] > 1:
				import warnings
				warnings.warn('There is a duplicate for molecule "' + nm + '".')
			fam = row.iloc[0, i_col_fam]
			family_list.append(fam)

	return family_list


# %%

import numpy as np


def get_data(ds_name, descriptor='smiles', format_='smiles', **kwargs):
	def get_poly200(descriptor, format_):
		data = load_dataset(
			'polyacrylates200', descriptor=descriptor,
			format_=format_, **kwargs
		)
		smiles = data['X']
		# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
		y = data['targets']
		if format_ == 'smiles':
			smiles = [s for i, s in enumerate(smiles) if i not in [6]]
			y = [y for i, y in enumerate(y) if i not in [6]]
			if 'names' in data:
				names = [y for i, y in enumerate(data['names']) if i not in [6]]
		y = np.reshape(y, (len(y), 1))
		families = [ds_name] * len(smiles)
		if 'names' in data:
			return (smiles, y, families, names)
		else:
			return (smiles, y, families)


	def get_poly200r(descriptor, format_):
		data = load_dataset(
			'poly200r', descriptor=descriptor, format_=format_,
			**kwargs
		)
		return tuple(data.values())


	def get_benchmarktao96(descriptor, format_):
		data = load_dataset(
			'benchmarktao96', descriptor=descriptor,
			format_=format_, **kwargs
		)
		return tuple(data.values())


	def get_sugarmono(descriptor, format_):
		data = load_dataset(
			'sugarmono', descriptor=descriptor, format_=format_,
			**kwargs
		)
		smiles = data['X']
		y = data['targets']
		families = data['families']
		if format_ == 'smiles':
			idx_rm = [17]  # Generated atomic 3d coordinates are all 0s.
			smiles = [s for i, s in enumerate(smiles) if i not in idx_rm]
			y = [y for i, y in enumerate(y) if i not in idx_rm]
			families = [f for i, f in enumerate(families) if i not in idx_rm]
		y = np.reshape(y, (len(y), 1))
		return smiles, y, families


	def get_acyclic(descriptor, format_):
		from gklearn.dataset import Dataset
		root = os.path.join(DIR_LOAD_DATASET, '../datasets/')
		dataset = Dataset('Acyclic', root=root)
		graphs = dataset.graphs
		y = dataset.targets
		families = ['Acyclic'] * len(graphs)
		return graphs, y, families


	def get_alkane(ds_name, descriptor, format_):
		from gklearn.dataset import Dataset
		root = os.path.join(DIR_LOAD_DATASET, '../datasets/')
		dataset = Dataset(ds_name, root=root)
		graphs = dataset.graphs
		y = dataset.targets
		families = [ds_name] * len(graphs)
		return graphs, y, families


	if isinstance(descriptor, str) and descriptor.lower() in [
		'smiles+dis_stats_obabel', 'smiles+xyz_obabel']:
		format_ = 'rdkit'

	if ds_name.lower() == 'poly200':
		data = get_poly200(descriptor, format_)
		return data

	elif ds_name.lower() == 'poly200r':
		data = get_poly200r(descriptor, format_)
		return data

	elif ds_name.lower() == 'benchmarktao96':
		data = get_benchmarktao96(descriptor, format_)
		return data

	elif ds_name.lower() == 'thermo_exp':
		data = load_dataset(
			'thermophysical', descriptor=descriptor,
			format_=format_, t_type='exp', **kwargs
		)
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
		data = load_dataset(
			'thermophysical', descriptor=descriptor,
			format_=format_, t_type='cal', **kwargs
		)
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


	### Redox datasets:

	elif ds_name.lower() in ['brem_togn', 'bremond', 'tognetti']:
		dataset = load_dataset(
			ds_name, descriptor=descriptor, format_=format_, **kwargs
		)
		smiles = dataset['X']
		y = dataset['targets']
		families = dataset['families']


	### Acyclic datasets:

	elif ds_name.lower() in ['acyclic']:
		data = get_acyclic(descriptor, format_)
		return data


	### Alkane datasets:
	elif ds_name.lower().startswith('alkane'):
		data = get_alkane(ds_name, descriptor, format_)
		return data

	else:
		raise ValueError('Dataset name %s can not be recognized.' % ds_name)

	return smiles, y, families


# %%


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

	# %%
	dataset = load_dataset('sugarmono', rm_replicate=False)
