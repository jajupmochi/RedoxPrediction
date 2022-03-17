#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:54:40 2022

@author: ljia
"""
from typing import List, Tuple
import numpy as np

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot

from dataset.utils.molecule_feature_utils import get_atoms_3d_coordinates


def _construct_atom_feature(
		atom: RDKitAtom,
		h_bond_infos: List[Tuple[int, str]],
		use_chirality: bool,
		use_partial_charge: bool) -> np.ndarray:
	"""Construct an atom feature from a RDKit atom object.
	Parameters
	----------
	atom: rdkit.Chem.rdchem.Atom
		RDKit atom object
	h_bond_infos: List[Tuple[int, str]]
		A list of tuple `(atom_index, hydrogen_bonding_type)`.
		Basically, it is expected that this value is the return value of
		`construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
		value is "Acceptor" or "Donor".
	use_chirality: bool
		Whether to use chirality information or not.
	use_partial_charge: bool
		Whether to use partial charge data or not.
	Returns
	-------
	np.ndarray
		A one-hot vector of the atom feature.
	"""
	atom_type = get_atom_type_one_hot(atom)
	formal_charge = get_atom_formal_charge(atom)
	hybridization = get_atom_hybridization_one_hot(atom)
	acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
	aromatic = get_atom_is_in_aromatic_one_hot(atom)
	degree = get_atom_total_degree_one_hot(atom)
	total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
	atom_feat = np.concatenate([
			atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree,
			total_num_Hs
	])

	if use_chirality:
		chirality = get_atom_chirality_one_hot(atom)
		atom_feat = np.concatenate([atom_feat, np.array(chirality)])

	if use_partial_charge:
		partial_charge = get_atom_partial_charge(atom)
		atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])
	return atom_feat


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
	"""Construct a bond feature from a RDKit bond object.
	Parameters
	---------
	bond: rdkit.Chem.rdchem.Bond
		RDKit bond object
	Returns
	-------
	np.ndarray
		A one-hot vector of the bond feature.
	"""
	bond_type = get_bond_type_one_hot(bond)
	same_ring = get_bond_is_in_same_ring_one_hot(bond)
	conjugated = get_bond_is_conjugated_one_hot(bond)
	stereo = get_bond_stereo_one_hot(bond)
	return np.concatenate([bond_type, same_ring, conjugated, stereo])


class DCMolGraphFeaturizer(MolecularFeaturizer):
	"""This class is a featurizer of general graph convolution networks for molecules.
	The default node(atom) and edge(bond) representations are based on
	`WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
	you could use this class as a guide to define your original Featurizer. In many cases, it's enough
	to modify return values of `construct_atom_feature` or `construct_bond_feature`.
	The default node representation are constructed by concatenating the following values,
	and the feature length is 30.
	- Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
	- Formal charge: Integer electronic charge.
	- Hybridization: A one-hot vector of "sp", "sp2", "sp3".
	- Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
	- Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
	- Degree: A one-hot vector of the degree (0-5) of this atom.
	- Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
	- Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
	- Partial charge: Calculated partial charge. (Optional)
	The default edge representation are constructed by concatenating the following values,
	and the feature length is 11.
	- Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
	- Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
	- Conjugated: A one-hot vector of whether this bond is conjugated or not.
	- Stereo: A one-hot vector of the stereo configuration of a bond.
	If you want to know more details about features, please check the paper [1]_ and
	utilities in deepchem.utils.molecule_feature_utils.py.
	Examples
	--------
	>>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
	>>> featurizer = MolGraphConvFeaturizer(use_edges=True)
	>>> out = featurizer.featurize(smiles)
	>>> type(out[0])
	<class 'deepchem.feat.graph_data.GraphData'>
	>>> out[0].num_node_features
	30
	>>> out[0].num_edge_features
	11
	References
	----------
	.. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
		Journal of computer-aided molecular design 30.8 (2016):595-608.
	Note
	----
	This class requires RDKit to be installed.
	"""

	def __init__(self,
			  use_edges: bool = False,
			  use_chirality: bool = False,
			  use_partial_charge: bool = False,
			  coords_scaling: str = 'auto'):
		"""
		Parameters
		----------
		use_edges: bool, default False
			Whether to use edge features or not.
		use_chirality: bool, default False
			Whether to use chirality information or not.
			If True, featurization becomes slow.
		use_partial_charge: bool, default False
			Whether to use partial charge data or not.
			If True, this featurizer computes gasteiger charges.
			Therefore, there is a possibility to fail to featurize for some molecules
			and featurization becomes slow.
		"""
		self.use_edges = use_edges
		self.use_partial_charge = use_partial_charge
		self.use_chirality = use_chirality
		self.coords_scaling = coords_scaling
		self.coords_scaler = None


	def featurize(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
		"""Calculate features for molecules.

		Parameters
		----------
		datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
			RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
			strings.
		log_every_n: int, default 1000
			Logging messages reported every `log_every_n` samples.

		Returns
		-------
		features: np.ndarray
			A numpy array containing a featurized representation of `datapoints`.
		"""
		features = super(DCMolGraphFeaturizer, self).featurize(datapoints,
														 log_every_n=log_every_n,
														 **kwargs)
		# Scale 3d coordinates.
		if self.coords_scaling == 'auto':
			# in "auto" mode, the scaler will be set to `MinMaxScaler()` if it
			# is not given in method argument.
			from sklearn.preprocessing import MinMaxScaler
			self.coords_scaler = kwargs.get('coords_scaler',
								   MinMaxScaler(feature_range=(0, 1)))
			self.scale_3d_coordinates(features, self.coords_scaler)
		else:
			raise ValueError('coords_scaling: "%s" can not be recognized. '
					'Possible candidates include "auto".' % self.coords_scaling)

		return features


	def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
		"""Calculate molecule graph features from RDKit mol object.
		Parameters
		----------
		datapoint: rdkit.Chem.rdchem.Mol
			RDKit mol object.
		Returns
		-------
		graph: GraphData
			A molecule graph with some features.
		"""
		assert datapoint.GetNumAtoms(
		) > 1, "More than one atom should be present in the molecule for this featurizer to work."
		if 'mol' in kwargs:
			datapoint = kwargs.get("mol")
			raise DeprecationWarning(
					'Mol is being phased out as a parameter, please pass "datapoint" instead.'
			)

		if self.use_partial_charge:
			try:
				datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
			except:
				# If partial charges were not computed
				try:
					from rdkit.Chem import AllChem
					AllChem.ComputeGasteigerCharges(datapoint)
				except ModuleNotFoundError:
					raise ImportError("This class requires RDKit to be installed.")

		# construct atom (node) feature
		h_bond_infos = construct_hydrogen_bonding_info(datapoint)
		atom_features = np.asarray(
				[
				 _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
																		self.use_partial_charge)
				 for atom in datapoint.GetAtoms()
				],
				dtype=float,
		)
		# Add 3d coordinates.
		# Keep the coordinates at the end of the feature vectors to keep the
		# method `scale_3d_coordinates` correct.
		coords = get_atoms_3d_coordinates(datapoint,
									use_bohr=False, complete_coords=False)
		atom_features = np.concatenate([atom_features, coords], axis=1)

		# construct edge (bond) index
		src, dest = [], []
		for bond in datapoint.GetBonds():
			# add edge list considering a directed graph
			start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
			src += [start, end]
			dest += [end, start]

		# construct edge (bond) feature
		bond_features = None	# deafult None
		if self.use_edges:
			features = []
			for bond in datapoint.GetBonds():
				features += 2 * [_construct_bond_feature(bond)]
			bond_features = np.asarray(features, dtype=float)

		return GraphData(
				node_features=atom_features,
				edge_index=np.asarray([src, dest], dtype=int),
				edge_features=bond_features)


	def scale_3d_coordinates(self, features, scaler):
		# Fit scaler if needed.
		from sklearn.utils.validation import check_is_fitted
		from sklearn.exceptions import NotFittedError
		try:
			check_is_fitted(scaler)
		except NotFittedError:
			# Retrieve all coordinates.
			coords = np.empty((0, 3))
			for gdata in features:
				coords = np.concatenate((coords, gdata.node_features[:, -3:]),
							axis=0)
			# Fit the scaler.
			scaler.fit(coords)

# 			# Find min and max values.
# 			min_coords = np.full((1, 3), np.inf)
# 			max_coords = np.full((1, 3), -np.inf)


# 			# Find min and max values.
# 			min_coords = np.full((1, 3), np.inf)
# 			max_coords = np.full((1, 3), -np.inf)
# 			for gdata in features:
# 				coords = gdata.node_features[:, -3:]
# 				cur_min = coords.max(axis=0)
# 				cur_max = coords.min(axis=0)

		# Transform data.
		for gdata in features:
			coords = gdata.node_features[:, -3:]
			coords_fitted = scaler.transform(coords)
			gdata.node_features[:, -3:] = coords_fitted


# 	@property
# 	def coords_scaler(self):
# # 		if hasattr(self, 'coords_scaling'):
# 		return self.coords_scaling
# # 		else:
# # 			raise ValueError('"coords_scaling" has not set yet. Initialize '
# # 					'`DCMolGraphFeaturizer` object with the argument '
# # 					'`coords_scaling` or set the argument `coords_scaler` in '
# # 					' the function `featurize`.')