#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:51:27 2022

@author: ljia
"""
import numpy as np

from typing import List, Union, Tuple

from deepchem.utils.typing import RDKitMol
from deepchem.utils.molecule_feature_utils import one_hot_encode


#################################################################
# atom (node) featurization
#################################################################


def get_atoms_3d_coordinates(datapoint: RDKitMol,
							 use_bohr: bool = False, #@todo: make sure what unit is used.
							 complete_coords: bool = False) -> np.ndarray:
	"""Get the 3d coordinates feature of each atom.

	Parameters
	---------
	datapoint: rdkit.Chem.rdchem.Mol
		RDKit Mol object

	use_bohr: bool, optional (default False)
		Whether to use bohr or angstrom as a coordinate unit.

	complete_coords: bool, optional (default False)
		Whether to generate missing coordinates automatically by `RDKit`.

	Returns
	-------
    np.ndarray
		A numpy array of atomic coordinates. The shape is `(n_atoms, 3)`.
	"""
	try:
		from rdkit import Chem
		from rdkit.Chem import AllChem
	except ModuleNotFoundError:
		raise ImportError('This class requires RDKit to be installed.')

	# Check whether num_confs >=1 or not
	num_confs = len(datapoint.GetConformers())
	if num_confs == 0:
		if complete_coords:
			datapoint = Chem.AddHs(datapoint)
			AllChem.EmbedMolecule(datapoint, AllChem.ETKDG())
			datapoint = Chem.RemoveHs(datapoint)
			import warnings
			warnings.warn('This molecule does not contain any 3d coordinates. '
				 'The 3d coordinates are generated automatically by `RDKit`.',
				 category=UserWarning)
# 			raise NotImplementedError('Completing 3d coordinates is not '
# 							 'implemented yet. Sorry~ ;)')
		else:
			raise ValueError('This molecule does not contain any 3d coordinates. '
					'Try to set `complete_coords=True`.')

	N = datapoint.GetNumAtoms()
	coords = np.zeros((N, 3))

	# RDKit stores atomic coordinates in Angstrom. Atomic unit of length is the
	# bohr (1 bohr = 0.529177 Angstrom). Converting units makes gradient calculation
	# consistent with most QM software packages.
	if use_bohr:
		coords_list = [
				datapoint.GetConformer(0).GetAtomPosition(i).__idiv__(0.52917721092)
				for i in range(N)
		]
	else:
		coords_list = [
				datapoint.GetConformer(0).GetAtomPosition(i) for i in range(N)
		]

	for atom in range(N):
		coords[atom, 0] = coords_list[atom].x
		coords[atom, 1] = coords_list[atom].y
		coords[atom, 2] = coords_list[atom].z

	return coords


def get_atoms_distance_stats(datapoint: RDKitMol,
							 use_bohr: bool = False, #@todo: make sure what unit is used.
							 complete_coords: bool = False) -> np.ndarray:
	"""Get the statistical features of distances between each atom and others,
	including: [min, max, mean].

	Parameters
	---------
	datapoint: rdkit.Chem.rdchem.Mol
		RDKit Mol object

	use_bohr: bool, optional (default False)
		Whether to use bohr or angstrom as a coordinate unit.

	complete_coords: bool, optional (default False)
		Whether to generate missing coordinates automatically by `RDKit`.

	Returns
	-------
    np.ndarray
		A numpy array of statistical features of distances. The shape is
		`(n_atoms, 3)`.
	"""
	try:
		from rdkit import Chem
		from rdkit.Chem import AllChem
	except ModuleNotFoundError:
		raise ImportError('This class requires RDKit to be installed.')

	# Check whether num_confs >=1 or not
	num_confs = len(datapoint.GetConformers())
	if num_confs == 0:
		if complete_coords:
			datapoint = Chem.AddHs(datapoint)
			AllChem.EmbedMolecule(datapoint, AllChem.ETKDG())
			datapoint = Chem.RemoveHs(datapoint)
			import warnings
			warnings.warn('This molecule does not contain any 3d coordinates. '
				 'The 3d coordinates are generated automatically by `RDKit`.',
				 category=UserWarning)
# 			raise NotImplementedError('Completing 3d coordinates is not '
# 							 'implemented yet. Sorry~ ;)')
		else:
			raise ValueError('This molecule does not contain any 3d coordinates. '
					'Try to set `complete_coords=True`.')

	N = datapoint.GetNumAtoms()
	features = np.zeros((N, 3))

	# RDKit stores atomic coordinates in Angstrom. Atomic unit of length is the
	# bohr (1 bohr = 0.529177 Angstrom). Converting units makes gradient calculation
	# consistent with most QM software packages.
	if use_bohr:
		coords_list = [
				datapoint.GetConformer(0).GetAtomPosition(i).__idiv__(0.52917721092)
				for i in range(N)
		]
	else:
		coords_list = [
				datapoint.GetConformer(0).GetAtomPosition(i) for i in range(N)
		]

	# Compute (Euclidean) distance matrix.
	dis_mat = np.zeros((N, N))
	for atom1 in range(N):
		for atom2 in range(atom1 + 1, N):
			coord1 = np.array([coords_list[atom1].x,
					  coords_list[atom1].y, coords_list[atom1].z])
			coord2 = np.array([coords_list[atom2].x,
					  coords_list[atom2].y, coords_list[atom2].z])
			dis_mat[atom1, atom2] = np.linalg.norm(coord1 - coord2, ord=2)
			dis_mat[atom2, atom1] = dis_mat[atom1, atom2]

	# Compute statistics of distances (min, max, mean).
	np.fill_diagonal(dis_mat, np.nan) # inplace operation
# 	mask = ~np.eye(dis_mat.shape[0], dtype=bool)
	features[:, 0] = np.nanmin(dis_mat, axis=0)
	features[:, 1] = np.nanmax(dis_mat, axis=0)
	features[:, 2] = np.nanmean(dis_mat, axis=0)

	return features