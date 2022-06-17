#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:06:36 2022

@author: ljia
"""
import sys
sys.path.insert(0, '../')
import numpy as np
from dataset.load_dataset import get_data
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem


def molecule_from_smiles(smiles):
	# MolFromSmiles(m, sanitize=True) should be equivalent to
	# MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
	molecule = Chem.MolFromSmiles(smiles, sanitize=False)

	# If sanitization is unsuccessful, catch the error, and try again without
	# the sanitization step that caused the error
	flag = Chem.SanitizeMol(molecule, catchErrors=True)
	if flag != Chem.SanitizeFlags.SANITIZE_NONE:
		Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

	Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
	return molecule


def plot_all_molecules(smiles, names, filename='mols_poly200_2'):
	from rdkit.Chem.Draw import MolsToGridImage
	molecules = [molecule_from_smiles(s) for s in smiles]
#	y_true = [y[index] for index in test_index]
#	y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)
#	y_pred = y_scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))
#	from sklearn.metrics import mean_absolute_error
#	MAE = mean_absolute_error(y_true, y_pred)

#	legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
	legends = [str(i) + ': ' + names[i] for i in range(len(molecules))]
	print(legends)
	img = MolsToGridImage(molecules[170:-5], molsPerRow=5, subImgSize=(300, 200),
					   legends=legends[170:-5], # [133:-5]
					   maxMols=300,
#					   useSVG=True,
					   returnPNG=False # To save, this argument must be set explicitly.
					   )
	if not filename.endswith('.png'):
		filename += '.png'
# 	img.show()
	img.save(filename)


if __name__ == '__main__':
	X, y, families, names = get_data('poly200', descriptor='smiles',
								  format_='smiles',
								  with_names=True)
	smiles = np.array(X)
#	y = np.array(y)
	plot_all_molecules(smiles, names)