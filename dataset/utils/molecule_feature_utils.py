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
							 complete_coords: bool = False) -> List[List[float]]:
	"""Get the 3d coordinates feature of an atom.

	Parameters
	---------
	atom: rdkit.Chem.rdchem.Atom
		RDKit atom object
	allowable_set: List[str]
		The atom types to consider. The default set is
		`["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]`.
	include_unknown_set: bool, default True
		If true, the index of all atom not in `allowable_set` is `len(allowable_set)`.

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