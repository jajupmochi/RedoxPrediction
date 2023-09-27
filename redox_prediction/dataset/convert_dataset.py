#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:39:31 2022

@author: ljia
"""
import sys
import os
import numpy as np
sys.path.insert(0, '../')
from dataset.load_dataset import load_dataset
from dataset.load_dataset import get_data


#%%

def smiles_list_to_ctfiles(smiles_list, dir_ct=None, targets=None, save_file=True):
	'''
	Convert smiles to ChemDraw table files.
	'''
	if save_file:
		os.makedirs(dir_ct, exist_ok=True)

	# Convert using openbabel.
	from openbabel import openbabel
	conv = openbabel.OBConversion()
	conv.SetInAndOutFormats('smi', 'ct')
	mol = openbabel.OBMol()
# 	gen3d = openbabel.OBOp.FindType('gen3D')

	ct_all = []
	ds_str = '' # string containing all .ct file names

	for idx, smiles in enumerate(smiles_list):
		print(idx)

		conv.ReadString(mol, smiles)

		### ------------------------------------------------
# 		gen3d.Do(mol, '--best')
		ct_str = conv.WriteString(mol)
# 		print(xyz_str)
		ct_all.append(ct_str)

# 		### ------------------------------------------------
# 		for level in ['', '--fastest', 'fast', '--fast', 'fast', '--medium', 'medium', '--better', 'better', '--best', 'best']:
# 			print(level)
# 			gen3d.Do(mol, level)
# 			xyz_str = conv.WriteString(mol)
# 			print(xyz_str)

		# Save file.
		if save_file:
			fn_ct = os.path.join(dir_ct, str(idx) + '.ct')
			with open(fn_ct, 'w') as f:
				f.write(ct_str)

		if targets is None:
			ds_str += str(idx) + '.ct\n'
		else:
			ds_str += str(idx) + '.ct ' + str(float(targets[idx])) + '\n'


	if save_file:
		fn_ds = os.path.join(dir_ct, 'dataset.ds')
		with open(fn_ds, 'w') as f:
			f.write(ds_str)

	return ct_all, ds_str


def ctfiles_to_smiles_list(dir_ct):
	'''
	Convert ChemDraw table files to smiles.
	'''
	# Convert using openbabel.
	from openbabel import openbabel
	conv = openbabel.OBConversion()
	conv.SetInAndOutFormats('ct', 'smi')
	mol = openbabel.OBMol()
# 	gen3d = openbabel.OBOp.FindType('gen3D')

	smi_all = []

	for idx in range(0, 195):
		print(idx)

		# Get ct string from file.
		fn_ct = os.path.join(dir_ct, str(idx) + '.ct')
		with open(fn_ct, 'r') as f:
			ct_str = f.read()

		conv.ReadString(mol, ct_str)

		### ------------------------------------------------
# 		gen3d.Do(mol, '--best')
		smi = conv.WriteString(mol)
# 		print(xyz_str)
		smi_all.append(smi)

# 		### ------------------------------------------------
# 		for level in ['', '--fastest', 'fast', '--fast', 'fast', '--medium', 'medium', '--better', 'better', '--best', 'best']:
# 			print(level)
# 			gen3d.Do(mol, level)
# 			xyz_str = conv.WriteString(mol)
# 			print(xyz_str)


	return smi_all


#%%


def smiles_list_to_xyz(smiles_list, dir_xyz, save_file=True):
# 	str_smi = smiles_list_to_smi(smiles_list, fn_smi)

	if save_file:
		os.makedirs(dir_xyz, exist_ok=True)

	# Convert using openbabel.
	from openbabel import openbabel
	conv = openbabel.OBConversion()
	conv.SetInAndOutFormats('smi', 'xyz')
	mol = openbabel.OBMol()
	gen3d = openbabel.OBOp.FindType('gen3D')

	xyz_all = []

	for idx, smiles in enumerate(smiles_list):
		print(idx)

		conv.ReadString(mol, smiles)

		### ------------------------------------------------
		gen3d.Do(mol, '--best')
		xyz_str = conv.WriteString(mol)
# 		print(xyz_str)
		xyz_all.append(xyz_str)

# 		### ------------------------------------------------
# 		for level in ['', '--fastest', 'fast', '--fast', 'fast', '--medium', 'medium', '--better', 'better', '--best', 'best']:
# 			print(level)
# 			gen3d.Do(mol, level)
# 			xyz_str = conv.WriteString(mol)
# 			print(xyz_str)


		# Save file.
		if save_file:
			fn_xyz = os.path.join(dir_xyz, str(idx) + '.xyz')
			with open(fn_xyz, 'w') as f:
				f.write(xyz_str)

	return xyz_all


#%%


def smiles_list_to_smi(smiles_list, save_path):
	# Constrcut the string to save.
	str_save = ''
	for smiles in smiles_list:
		str_save += smiles + '\n'

	# Save to file.
	with open(save_path, 'w') as f:
		f.write(str_save)

	return str_save


#%%


def check_openbabel_convert_consistence():
	dir1 = '../outputs/gnn/quantum/poly200/'
	dir2 = '../outputs/gnn/quantum/poly200_trial1/'

	idx_diff = []
	for i_mol in range(1, 197):
		with open(dir1 + 'out' + str(i_mol) + '.xyz', 'r') as f:
			str_xyz1 = f.read()
		with open(dir2 + 'out' + str(i_mol) + '.xyz', 'r') as f:
			str_xyz2 = f.read()
		if str_xyz1.strip() != str_xyz2.strip():
			idx_diff.append(i_mol)

	return idx_diff


#%%


def sdf_to_gjf(
		str_sdf,
		method='m062x',
		out='wfx',
		i_mol=None
		):

	if method == 'pbeqidh' and out == 'wfx':
		import warnings
		warnings.warn('Are you sure you want to compute .wfx file using "pbeqidh"? It can be quite slow...')


	### useful string for .gjf files
	str_method = get_str_method(method)
	str_out = ('out=wfx' if out == 'wfx' else '')


	### Create strings to save.
	# prefix info.
	str_pre = '%mem=8000mb\n%nprocs=12\n%chk=Linlin.chk\n#p ' + str_method + ' ' + str_out + ' pop=npa\n\n' # @todo: pop=npa should this be set by arguments?
	str_pre += str_sdf[0:str_sdf.index('\n')] + '\n\n0 1\n'

	# suffix.
	if out == 'wfx':
		str_suf = '\nmolecule' + str(i_mol) + '.wfx\n\n\n'
	else:
		str_suf = '\n\n\n'


	### Get atom symbols and coordinates.
	try:
		idx_2D3D = str_sdf.index('3D')
	except ValueError:
		idx_2D3D = str_sdf.index('2D')
# 		print(i_mol + ': I am 2D.')

	idx_V2000 = str_sdf.index('V2000')
	nb_atoms = int(str_sdf[idx_2D3D + 3:idx_V2000].strip().split()[0].strip())
	atom_lines = str_sdf[idx_V2000:].split('\n')[1:1+nb_atoms]

	# For each atom:
	str_atoms = ''
	for line in atom_lines:
		ls = line.strip().split()
		str_atoms += ' '.join([ls[3]] + ls[0:3]) + '\n'

	# Create saving string.
	str_gjf = str_pre + str_atoms + str_suf

	return str_gjf


def get_str_method(method):
	if method == 'm062x':
		str_method = 'm062x 6-31++G(d,p)' # very high accuracy?
	elif method == 'm062x.6-31+G(d)':
		str_method = 'M062X 6-31+G(d)'
	elif method == 'svwn':
		str_method = 'svwn sto-3g' # fast and not accurate.
	elif method == 'svwn.6-31Gd':
		str_method = 'svwn 6-31G(d)' # may solve some nnacp problem caused by "svwn sto-3g".
	elif method == 'svwn.sto-3g.scf=qc': # more robust than svwn 6-31G(d), but much more time consuming.
		str_method = 'svwn sto-3g scf=qc'
	elif method == 'pbeqidh':
		str_method = 'pbeqidh def2tzvp' # more accurate than "svwn sto-3g".
	else:
		import warnings
		warnings.warn('The method cannot be recognized, better be sure it is correct! :)')
		str_method = method

	return str_method


#%%


def sdf_to_xyz(str_sdf):

	### Get atom symbols and coordinates.
	try:
		idx_2D3D = str_sdf.index('3D')
	except ValueError:
		idx_2D3D = str_sdf.index('2D')
# 		print(i_mol + ': I am 2D.')

	idx_V2000 = str_sdf.index('V2000')
	nb_atoms = int(str_sdf[idx_2D3D + 3:idx_V2000].strip().split()[0].strip())
	atom_lines = str_sdf[idx_V2000:].split('\n')[1:1+nb_atoms]

	# For each atom:
	coords = np.empty((nb_atoms, 3))
	for i_atom, line in enumerate(atom_lines):
		ls = line.strip().split()
		coords[i_atom, :] = ls[0:3]

	return coords


#%%

def smiles_to_rdkitmol(smiles, assign_stereo=True):
	from rdkit import Chem

	# MolFromSmiles(m, sanitize=True) should be equivalent to
	# MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
	molecule = Chem.MolFromSmiles(smiles, sanitize=False)

	# If sanitization is unsuccessful, catch the error, and try again without
	# the sanitization step that caused the error
	flag = Chem.SanitizeMol(molecule, catchErrors=True)
	if flag != Chem.SanitizeFlags.SANITIZE_NONE:
		Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

	if assign_stereo:
		Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)

	return molecule


#%%


def sdf_to_rdkitmol(sdf_str, add_Hs=False, assign_stereo=True):
	# Create the rdkit mol from the .sdf file.
	import io
	from rdkit import Chem

	# Add Hydrogens.
# 			with Chem.SDMolSupplier(fn_sdf, removeHs=False) as suppl:
# 				mol = suppl[0]
	suppl = Chem.ForwardSDMolSupplier(
		io.BytesIO(sdf_str.encode('utf-8')), removeHs=(not add_Hs))
	for molecule in suppl:
		break
# 				if mol is None:
# 					raise Exception('Molecule # %d can not be loaded as a rdkit mol.' % (i + 1))
# 			for a in mol.GetAtoms():
# 			    print(a.GetSymbol())

	# If sanitization is unsuccessful, catch the error, and try again without
	# the sanitization step that caused the error
	flag = Chem.SanitizeMol(molecule, catchErrors=True)
	if flag != Chem.SanitizeFlags.SANITIZE_NONE:
		Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

	if assign_stereo:
		Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)

	return molecule


if __name__ == '__main__':
	#%%

# 	DS_Name_List = ['poly200', 'thermo_exp', 'thermo_cal']
# 	# indices of the smiles that can not be recognized by the program Openbabel.
# 	idx_rm_list = [[6], [168, 169, 170, 171, 172], [151, 198, 199]]

# 	# The directory to save the output file.
# 	save_dir = '../outputs/gnn/quantum/'
# 	os.makedirs(save_dir, exist_ok=True)

# 	# Convert.
# 	for ds_name in DS_Name_List:
# 		if ds_name.lower() == 'poly200':
#  			data = load_dataset('polyacrylates200', format_='smiles')
# 		elif ds_name.lower() == 'thermo_exp':
#  			data = load_dataset('thermophysical', format_='smiles', t_type='exp')
# 		elif ds_name.lower() == 'thermo_cal':
#  			data = load_dataset('thermophysical', format_='smiles', t_type='cal')

# 		smiles_list = data['X']

# 		# Transfer smiles to .smi file.
# 		save_path = save_dir + 'smiles.' + ds_name + '.smi'
# 		str_save = smiles_list_to_smi(smiles_list, save_path)

# 		# Transform and save.
# 		dir_xyz = save_dir + '/' + ds_name + '/'
# # 		fn_smi = save_dir + 'smiles.' + ds_name + '.smi'
# 		xyz_all = smiles_list_to_xyz(smiles_list, dir_xyz, save_file=True)

# 		pass

	#%%
# 	idx_diff = check_openbabel_convert_consistence()


	#%%

	### Load dataset.
	for ds_name in ['poly200']: # , 'sugarmono']:
		smiles, y, families = get_data(ds_name, descriptor='smiles')
		ct_strs = smiles_list_to_ctfiles(smiles, dir_ct='../outputs/hoffman/poly200/ctfiles/', targets=y)

		# Convert the generated .ct files to smiles string to check.
		smi_list = ctfiles_to_smiles_list('../outputs/hoffman/poly200/ctfiles/')