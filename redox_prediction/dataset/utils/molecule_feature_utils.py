#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:51:27 2022

@author: ljia
"""
import logging
import os
import sys
import numpy as np

from typing import List, Union, Tuple

from deepchem.utils.typing import RDKitMol
# from deepchem.utils.molecule_feature_utils import one_hot_encode

sys.path.insert(0, '../')
from dataset.convert_dataset import sdf_to_xyz


logger = logging.getLogger(__name__)


def one_hot_encode(val: Union[int, str],
                   allowable_set: Union[List[str], List[int]],
                   include_unknown_set: bool = False) -> List[float]:
  """One hot encoder for elements of a provided set.

  Examples
  --------
  >>> one_hot_encode("a", ["a", "b", "c"])
  [1.0, 0.0, 0.0]
  >>> one_hot_encode(2, [0, 1, 2])
  [0.0, 0.0, 1.0]
  >>> one_hot_encode(3, [0, 1, 2])
  [0.0, 0.0, 0.0]
  >>> one_hot_encode(3, [0, 1, 2], True)
  [0.0, 0.0, 0.0, 1.0]

  Parameters
  ----------
  val: int or str
    The value must be present in `allowable_set`.
  allowable_set: List[int] or List[str]
    List of allowable quantities.
  include_unknown_set: bool, default False
    If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

  Returns
  -------
  List[float]
    An one-hot vector of val.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

  Raises
  ------
  ValueError
    If include_unknown_set is False and `val` is not in `allowable_set`.
  """
  if include_unknown_set is False:
    if val not in allowable_set:
      logger.info("input {0} not in allowable set {1}:".format(
          val, allowable_set))

  # init an one-hot vector
  if include_unknown_set is False:
    one_hot_legnth = len(allowable_set)
  else:
    one_hot_legnth = len(allowable_set) + 1
  one_hot = [0.0 for _ in range(one_hot_legnth)]

  try:
    one_hot[allowable_set.index(val)] = 1.0  # type: ignore
  except:
    if include_unknown_set:
      # If include_unknown_set is True, set the last index is 1.
      one_hot[-1] = 1.0
    else:
      pass
  return one_hot


#################################################################
# atom (node) featurization
#################################################################


def get_atoms_3d_coordinates(
		datapoint: Union[RDKitMol, str],
		in_type: str = 'rdkitmol',
		tool: str = 'rdkit',
		use_bohr: bool = False, #@todo: make sure what unit is used.
		complete_coords: bool = False,
		**kwargs) -> np.ndarray:
	"""Get the 3d coordinates feature of each atom.

	Parameters
	---------
	datapoint: rdkit.Chem.rdchem.Mol or string
		RDKit Mol object or SMILES string.

	use_bohr: bool, optional (default False)
		Whether to use bohr or angstrom as a coordinate unit.

	complete_coords: bool, optional (default False)
		Whether to generate missing coordinates automatically by `RDKit`.

	Returns
	-------
    np.ndarray
		A numpy array of atomic coordinates. The shape is `(n_atoms, 3)`.
	"""
	if in_type == 'rdkitmol':
		return get_atoms_3d_coords_from_rdkitmol(
			datapoint, tool=tool, use_bohr=use_bohr,
			complete_coords=complete_coords)

	elif in_type == 'smiles' and tool == 'balloon':
		# @todo: check if balloon is installed.
		return get_atoms_3d_coords_from_smiles_by_balloon(
			datapoint, use_bohr=use_bohr, complete_coords=complete_coords, **kwargs)

	else:
		raise ValueError('Argument"in_type" (%f) or "tool" (%f) can not be recoganized.')


def get_atoms_3d_coords_from_rdkitmol(
		datapoint: RDKitMol,
		tool: str = 'rdkit',
		use_bohr: bool = False, #@todo: make sure what unit is used.
		complete_coords: bool = False,
		**kwargs) -> np.ndarray:
	"""Get the 3d coordinates feature of each atom.

	Parameters
	---------
	datapoint: rdkit.Chem.rdchem.Mol
		RDKit Mol object.

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


def get_atoms_3d_coords_from_smiles_by_balloon(
		datapoint: str,
		use_bohr: bool = False, #@todo: make sure what unit is used.
		complete_coords: bool = False,
		**kwargs) -> Tuple[np.ndarray, str]:
	"""
	return_mode: "lowest_energy", ""
	"""
	# Load arguments.
	ds_dir = kwargs.get('ds_dir', '')
	mff_file = kwargs.get('mff_file', '$HOME/Balloon/MMFF94.mff')
	with_GA = kwargs.get('with_GA', 'true')
	i_mol = kwargs.get('i_mol', 0)
	return_mode = kwargs.get('return_mode', 'lowest_energy')

	# Set file name.
	coords_dir = os.path.join(ds_dir, 'balloon.sdf/' + ('GA' if with_GA == 'true' else 'noGA'))
	os.makedirs(coords_dir, exist_ok=True)
	fn_sdf = os.path.join(coords_dir, 'out' + str(i_mol) + '.sdf')

	# Check if the .sdf file exists.
	if not os.path.isfile(fn_sdf) or os.path.getsize(fn_sdf) <= 0:
		# Compute 3d coordinates and generate the .sdf file by Balloon.
		command = 'export PATH="' + os.path.dirname(mff_file) + ':$PATH"\n'
		command += 'balloon -f ' + mff_file + ' --nconfs 20 --nGenerations 300 --rebuildGeometry ' + ('' if with_GA else '--noGA') + ' "' + datapoint + '" ' + fn_sdf
		os.system(command)

	# Read .sdf.
	with open(fn_sdf, 'r') as f:
		str_sdf = [s.strip() for s in f.read().strip().split('$$$$')[0:-1]]
	energies = [float(s.split('<energy>\n')[-1].strip()) for s in str_sdf]

	# Return according to settings.
	if return_mode == 'lowest_energy':
		idx_lowest = np.argmin(energies)
		str_ = str_sdf[idx_lowest]
		coords = sdf_to_xyz(str_)
		return coords, str_


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


def get_atomic_charges(
		i_mol: int,
		ds_dir: str = '',
		with_GA: str = 'true',
		**kwargs) -> Tuple[np.ndarray, str]:
	"""
	return_mode: "lowest_energy", ""
	"""
	# Load arguments.
	return_mode = kwargs.get('atom_char_return_mode', 'lowest_energy')
	gaussian_method = kwargs.get('gaussian_method', 'm062x')

	# Set file name.
	out_dir = os.path.join(ds_dir, 'balloon.sdf/' + ('GA' if with_GA == 'true' else 'noGA') + '/gaussian/' + gaussian_method + '/')
	if return_mode == 'lowest_energy':
		fn_out = os.path.join(out_dir, 'out' + str(i_mol) + '.0.out')

	# Check if the .out file exists.
	if not os.path.isfile(fn_out) or os.path.getsize(fn_out) <= 0:
		raise FileNotFoundError('No such file: ' + fn_out)

	# Read atomic_charges from the s.out file.
	charges = []
	with open(fn_out, 'r') as f:
		str_out = f.read()
	str_out = str_out[str_out.index('Summary of Natural Population Analysis'):]
	str_out = str_out[str_out.index('Atom'):str_out.index('* Total *')]
	str_out = str_out.strip().split('\n')
	# for each atom:
	for i in range(2, len(str_out) - 1):
		line = str_out[i].strip().split()
		charges.append(float(line[2]))

	return np.array(charges).reshape(-1, 1)