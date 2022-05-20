#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:09:19 2022

@author: ljia
"""

import os
import sys
sys.path.insert(0, '../')
# import numpy as np

from dataset.load_dataset import load_dataset


def compute_coordinates(smiles, ds_dir='',
						mff_file='$HOME/Balloon/MMFF94.mff',
						with_GA=True,
						**kwargs):

	coords_dir = os.path.join(ds_dir, 'balloon.sdf' + ('.GA' if with_GA else '.noGA'))
	os.makedirs(coords_dir, exist_ok=True)
	# for each smiles, labelings starting from 1.
	for i, sm in enumerate(smiles):
		# Compute 3d coordinates and generate the .sdf file by OpenBabel
		# command line tools if the file does not exist.
# 			print(i)
		fn_sdf = os.path.join(coords_dir, 'out' + str(i + 1) + '.sdf')
		fn_mol2 = os.path.join(coords_dir, 'out' + str(i + 1) + '.mol2')
		if os.path.isfile(fn_sdf) and os.path.isfile(fn_mol2):
			continue
		command = 'export PATH="' + os.path.dirname(mff_file) + ':$PATH"\n'
		if not os.path.isfile(fn_sdf):
			command += 'balloon -f ' + mff_file + ' --nconfs 20 --nGenerations 300 --rebuildGeometry ' + ('--noGA' if with_GA else '') + ' "' + sm + '" ' + fn_sdf
		if not os.path.isfile(fn_mol2):
			command += '\nballoon -f ' + mff_file + ' --onlycharge ' + fn_sdf + ' ' +  fn_mol2
		os.system(command)


if __name__ == '__main__':
	data = load_dataset('polyacrylates200', descriptor='smiles', format_='smiles')
	smiles = data['X']
	ds_dir = '../datasets/Polyacrylates200/'
	compute_coordinates(smiles, ds_dir=ds_dir, mff_file='$HOME/Balloon/MMFF94.mff')