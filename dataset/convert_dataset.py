#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:39:31 2022

@author: ljia
"""
import sys
import os
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


def smiles_list_to_smi(smiles_list, save_path):
	# Constrcut the string to save.
	str_save = ''
	for smiles in smiles_list:
		str_save += smiles + '\n'

	# Save to file.
	with open(save_path, 'w') as f:
		f.write(str_save)

	return str_save


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