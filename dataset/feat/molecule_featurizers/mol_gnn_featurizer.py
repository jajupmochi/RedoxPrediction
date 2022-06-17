#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:20:40 2022

@author: ljia
"""
import numpy as np

import tensorflow as tf

from rdkit import Chem

from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info

from dataset.utils import one_hot_encode
from dataset.utils.molecule_feature_utils import get_atoms_3d_coordinates
from dataset.utils.molecule_feature_utils import get_atoms_distance_stats


#%%
# =============================================================================
# Define featurizers.
# =============================================================================

class Featurizer:
	def __init__(self, allowable_sets):
		self.dim = 0
		self.allowable_sets = allowable_sets
		for k, s in allowable_sets.items():
 			self.dim += getattr(self, k)(None, allowable_set=s, return_dim=True)


	def encode(self, inputs, **kwargs):
		self.encode_args = kwargs
		output = []
		for name_feature, allowable_set in self.allowable_sets.items():
			feature = getattr(self, name_feature)(inputs, allowable_set=allowable_set)
			output += feature
# 			if feature not in feature_mapping:
# 				continue
# 			output[feature_mapping[feature]] = 1.0
		return np.array(output)


#################################################################
# atom (node) featurization
#################################################################


class AtomFeaturizer(Featurizer):
	def __init__(self, allowable_sets):
		super().__init__(allowable_sets)


	def atom_type(self, atom, allowable_set=None, include_unknown_set=True, return_dim=False):
		"""Get an one-hot feature of an atom type.

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
		List[float]
				An one-hot vector of atom types.
				If `include_unknown_set` is False, the length is `len(allowable_set)`.
				If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
		"""
		if return_dim:
			return len(allowable_set) + (1 if include_unknown_set else 0)
		return one_hot_encode(atom.GetSymbol(), allowable_set, include_unknown_set)


	def formal_charge(self, atom, allowable_set=None, return_dim=False):
		"""Get a formal charge of an atom.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object

		Returns
		-------
		List[float]
				A vector of the formal charge.
		"""
		if return_dim:
			return 1
		return [float(atom.GetFormalCharge())]


	def hybridization(self, atom, allowable_set=None, include_unknown_set=False, return_dim=False):
		"""Get an one-hot feature of hybridization type.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object
		allowable_set: List[str]
				The hybridization types to consider. The default set is `["SP", "SP2", "SP3"]`
		include_unknown_set: bool, default False
				If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

		Returns
		-------
		List[float]
				An one-hot vector of the hybridization type.
				If `include_unknown_set` is False, the length is `len(allowable_set)`.
				If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
		"""
		if return_dim:
			return len(allowable_set) + (1 if include_unknown_set else 0)
		return one_hot_encode(str(atom.GetHybridization()), allowable_set, include_unknown_set)
# 		return atom.GetHybridization().name.lower()


	def acceptor_donor(self, atom, allowable_set=None, return_dim=False):
		"""Get an one-hot feat about whether an atom accepts electrons or donates electrons.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object
		hydrogen_bonding: List[Tuple[int, str]]
				The return value of `construct_hydrogen_bonding_info`.
				The value is a list of tuple `(atom_index, hydrogen_bonding)` like (1, "Acceptor").

		Returns
		-------
		List[float]
				A one-hot vector of the ring size type. The first element
				indicates "Donor", and the second element indicates "Acceptor".
		"""
		if return_dim:
			return 2
		one_hot = [0.0, 0.0]
		atom_idx = atom.GetIdx()
		for hydrogen_bonding_tuple in self.encode_args['h_bond_infos']:
				if hydrogen_bonding_tuple[0] == atom_idx:
						if hydrogen_bonding_tuple[1] == "Donor":
								one_hot[0] = 1.0
						elif hydrogen_bonding_tuple[1] == "Acceptor":
								one_hot[1] = 1.0
		return one_hot


	def aromatic(self, atom, allowable_set=None, return_dim=False):
		"""Get ans one-hot feature about whether an atom is in aromatic system or not.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object

		Returns
		-------
		List[float]
				A vector of whether an atom is in aromatic system or not.
		"""
		if return_dim:
			return 1
		return [float(atom.GetIsAromatic())]


	def degree(self, atom, allowable_set=None, include_unknown_set=True, return_dim=False):
		"""Get an one-hot feature of the degree which an atom has.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object
		allowable_set: List[int]
				The degree to consider. The default set is `[0, 1, ..., 5]`
		include_unknown_set: bool, default True
				If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

		Returns
		-------
		List[float]
				A one-hot vector of the degree which an atom has.
				If `include_unknown_set` is False, the length is `len(allowable_set)`.
				If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
		"""
		if return_dim:
			return len(allowable_set) + (1 if include_unknown_set else 0)
		return one_hot_encode(atom.GetTotalDegree(), allowable_set, include_unknown_set)


	def n_valence(self, atom, allowable_set=None, include_unknown_set=True, return_dim=False):
		"""Get an one-hot feature of the total valence (explicit + implicit) of the atom.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object
		allowable_set: List[int]
				The degree to consider. The default set is `[0, 1, ..., 5, 6]`
		include_unknown_set: bool, default True
				If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

		Returns
		-------
		List[float]
				A one-hot vector of the degree which an atom has.
				If `include_unknown_set` is False, the length is `len(allowable_set)`.
				If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
		"""
		if return_dim:
			return len(allowable_set) + (1 if include_unknown_set else 0)
		return one_hot_encode(atom.GetTotalValence(), allowable_set, include_unknown_set)


	def total_num_Hs(self, atom, allowable_set=None, include_unknown_set=True, return_dim=False):
		"""Get an one-hot feature of the number of hydrogens which an atom has.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object
		allowable_set: List[int]
				The number of hydrogens to consider. The default set is `[0, 1, ..., 4]`
		include_unknown_set: bool, default True
				If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

		Returns
		-------
		List[float]
				A one-hot vector of the number of hydrogens which an atom has.
				If `include_unknown_set` is False, the length is `len(allowable_set)`.
				If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
		"""
		if return_dim:
			return len(allowable_set) + (1 if include_unknown_set else 0)
		return one_hot_encode(atom.GetTotalNumHs(), allowable_set, include_unknown_set)


	def chirality(self, atom, allowable_set=None, return_dim=False):
		"""Get an one-hot feature about an atom chirality type.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object

		Returns
		-------
		List[float]
				A one-hot vector of the chirality type. The first element
				indicates "R", and the second element indicates "S".
		"""
		if return_dim:
			return 2
		one_hot = [0.0, 0.0]
		try:
			chiral_type = atom.GetProp('_CIPCode')
			if chiral_type == "R":
					one_hot[0] = 1.0
			elif chiral_type == "S":
					one_hot[1] = 1.0
		except:
			pass
		return one_hot


	def atom_partial_charge(self, atom, allowable_set=None, return_dim=False):
		"""Get a partial charge of an atom.

		Parameters
		---------
		atom: rdkit.Chem.rdchem.Atom
				RDKit atom object

		Returns
		-------
		List[float]
				A vector of the parital charge.

		Notes
		-----
		Before using this function, you must calculate `GasteigerCharge`
		like `AllChem.ComputeGasteigerCharges(mol)`.
		"""
		gasteiger_charge = atom.GetProp('_GasteigerCharge')
		if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
				gasteiger_charge = 0.0
		return [float(gasteiger_charge)]


#################################################################
# bond (edge) featurization
#################################################################


class BondFeaturizer(Featurizer):
	def __init__(self, allowable_sets, add_self_loop=False):
		super().__init__(allowable_sets)
		self.add_self_loop = add_self_loop
		if add_self_loop:
			self.dim += 1

	def encode(self, bond):
		if bond is None:
			if not self.add_self_loop:
				raise Exception('add_self_loop is set to False; input bond can not be None.')
			output = np.zeros((self.dim,))
			output[-1] = 1.0
			return output
		else:
			output = super().encode(bond)
			if self.add_self_loop:
				output = np.append(output, 0)
			return output


	def bond_type(self, bond, allowable_set=None, include_unknown_set=False, return_dim=False):
		"""Get an one-hot feature of bond type.

		Parameters
		---------
		bond: rdkit.Chem.rdchem.Bond
				RDKit bond object
		allowable_set: List[str]
				The bond types to consider. The default set is `["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]`.
		include_unknown_set: bool, default False
				If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

		Returns
		-------
		List[float]
				A one-hot vector of the bond type.
				If `include_unknown_set` is False, the length is `len(allowable_set)`.
				If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
		"""
		if return_dim:
			return len(allowable_set) + (1 if include_unknown_set else 0)
		return one_hot_encode(
			str(bond.GetBondType()), allowable_set, include_unknown_set)


	def same_ring(self, bond, allowable_set=None, return_dim=False):
		"""Get an one-hot feature about whether atoms of a bond is in the same ring or not.

		Parameters
		---------
		bond: rdkit.Chem.rdchem.Bond
				RDKit bond object

		Returns
		-------
		List[float]
				A one-hot vector of whether a bond is in the same ring or not.
		"""
		if return_dim:
			return 1
		return [float(bond.IsInRing())]


	def conjugated(self, bond, allowable_set=None, return_dim=False):
		"""Get an one-hot feature about whether a bond is conjugated or not.

		Parameters
		---------
		bond: rdkit.Chem.rdchem.Bond
			RDKit bond object

		Returns
		-------
		List[float]
			A one-hot vector of whether a bond is conjugated or not.
		"""
		if return_dim:
			return 1
		return [float(bond.GetIsConjugated())]


	def stereo(self, bond, allowable_set=None, include_unknown_set=True, return_dim=False):
		"""Get an one-hot feature of the stereo configuration of a bond.

		Parameters
		---------
		bond: rdkit.Chem.rdchem.Bond
				RDKit bond object
		allowable_set: List[str]
				The stereo configuration types to consider.
				The default set is `["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]`.
		include_unknown_set: bool, default True
				If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

		Returns
		-------
		List[float]
				A one-hot vector of the stereo configuration of a bond.
				If `include_unknown_set` is False, the length is `len(allowable_set)`.
				If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
		"""
		if return_dim:
			return len(allowable_set) + (1 if include_unknown_set else 0)
		return one_hot_encode(
			str(bond.GetStereo()), allowable_set, include_unknown_set)


#%%

class MolGNNFeaturizer(object):
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
# 			  use_chirality: bool = False,
			  use_partial_charge: bool = False,
			  add_self_loop: bool = True,
			  af_allowable_sets: dict = None,
			  bf_allowable_sets: dict = None,
			  ):
# 			  use_distance_stats: bool = False,
# 			  use_xyz: bool = False,
# 			  return_int: bool = False,
# 			  feature_scaling: str = 'auto'):
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
# 		self.use_chirality = use_chirality
		self.use_partial_charge = use_partial_charge
		self.add_self_loop = add_self_loop

		if af_allowable_sets is None:
			af_allowable_sets = {
				'atom_type': ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
				'formal_charge': None, # [-2, -1, 0, 1, 2],
				'hybridization': ['SP', 'SP2', 'SP3'],
				'acceptor_donor': ['Donor', 'Acceptor'],
				'aromatic': [True, False],
				'degree': [0, 1, 2, 3, 4, 5],
				'total_num_Hs': [0, 1, 2, 3, 4],
				'chirality': ['R', 'S'],
				}
		self.atom_featurizer = AtomFeaturizer(allowable_sets=af_allowable_sets)

		if self.use_edges:
			if bf_allowable_sets is None:
				bf_allowable_sets = {
					'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
					'same_ring': [True, False],
					'conjugated': [True, False],
					'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'],
					}
			self.bond_featurizer = BondFeaturizer(allowable_sets=bf_allowable_sets,
											add_self_loop=add_self_loop)


# 		self.use_distance_stats = use_distance_stats
# 		self.use_xyz = use_xyz
# 		self.return_int = return_int
# 		self.feature_scaling = feature_scaling
# 		self.feature_scaler = None


	def featurize(self, smiles_list,
# 			   log_every_n=1000,
			   **kwargs) -> tuple:
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
		# Initialize graphs
		atom_features_list = []
		bond_features_list = []
		pair_indices_list = []

		for smiles in smiles_list:
			atom_features, bond_features, pair_indices = self._featurize(smiles)

			atom_features_list.append(atom_features)
			bond_features_list.append(bond_features)
			pair_indices_list.append(pair_indices)

		# Convert lists to ragged tensors for tf.data.Dataset later on
		features = (
			tf.ragged.constant(atom_features_list, dtype=tf.float32),
			tf.ragged.constant(bond_features_list, dtype=tf.float32),
			tf.ragged.constant(pair_indices_list, dtype=tf.int64),
		)

# 		features = super(DCMolGraphFeaturizer, self).featurize(datapoints,
# 														 log_every_n=log_every_n,
# 														 **kwargs)

# 		if self.return_int: # Format one-hot features to int.
# 			for feat in features:
# 				feat.node_features = feat.node_features.astype(int)
# 				feat.edge_features = feat.edge_features.astype(int)

# 		if self.use_distance_stats or self.use_xyz:
# 			# Scale real value features.
# 			if self.feature_scaling == 'auto':
# 				# in "auto" mode, the scaler will be set to `MinMaxScaler()` if it
# 				# is not given in method argument.
# 				from sklearn.preprocessing import MinMaxScaler
# 				self.feature_scaler = kwargs.get('feature_scaler',
# 									   MinMaxScaler(feature_range=(0, 1)))
# 				self.scale_real_value_features(features, self.feature_scaler)
# 			else:
# 				raise ValueError('feature_scaling: "%s" can not be recognized. '
# 						'Possible candidates include "auto".' % self.feature_scaling)

		return features


	def _featurize(self, smiles: str, **kwargs) -> tuple:
		"""Calculate molecule graph features from RDKit mol object.
		Parameters
		----------
		smiles: str
			RDKit mol object.
		Returns
		-------
		graph: tuple
			A molecule graph with some features.
		"""
		molecule = self.molecule_from_smiles(smiles)

		assert molecule.GetNumAtoms(
		) > 1, "More than one atom should be present in the molecule for this featurizer to work."

		atom_features, bond_features, pair_indices = self.graph_from_molecule(molecule)

		return atom_features, bond_features, pair_indices


# 		# Add statistical features of distances between the given atom and others.
# 		if self.use_distance_stats:
# 			# Keep the statistical features right before the 3d coordinate
# 			# features to keep the method `scale_real_value_features()` correct.
# 			stats = get_atoms_distance_stats(datapoint,
# 										use_bohr=False, complete_coords=False)
# 			atom_features = np.concatenate([atom_features, stats], axis=1)

# 		# Add 3d coordinates.
# 		if self.use_xyz:
# 			# Keep the coordinates at the end of the feature vectors to keep the
# 			# method `scale_real_value_features()` correct.
# 			coords = get_atoms_3d_coordinates(datapoint,
# 										use_bohr=False, complete_coords=False)
# 			atom_features = np.concatenate([atom_features, coords], axis=1)


	def molecule_from_smiles(self, smiles):
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


	def graph_from_molecule(self, molecule):
		# Initialize graph
		atom_features = []
		bond_features = []
		pair_indices = []


		if self.use_partial_charge:
			try:
				molecule.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
			except:
				# If partial charges were not computed
				try:
					from rdkit.Chem import AllChem
					AllChem.ComputeGasteigerCharges(molecule)
				except ModuleNotFoundError:
					raise ImportError("This class requires RDKit to be installed.")

		# construct atom (node) feature
		if 'acceptor_donor' in self.atom_featurizer.allowable_sets:
			h_bond_infos = construct_hydrogen_bonding_info(molecule)
		else:
			h_bond_infos = None


		for atom in molecule.GetAtoms():
			atom_features.append(self.atom_featurizer.encode(atom, h_bond_infos=h_bond_infos))

			# Add self-loop. Notice, this also helps against some edge cases where the
			# last node has no edges. Alternatively, if no self-loops are used, for these
			# edge cases, zero-padding on the output of the edge network is needed.
			if self.add_self_loop:
				pair_indices.append([atom.GetIdx(), atom.GetIdx()])
				if self.use_edges:
					bond_features.append(self.bond_featurizer.encode(None))

			atom_neighbors = atom.GetNeighbors()

			for neighbor in atom_neighbors:
				pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
				if self.use_edges:
					bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
					bond_features.append(self.bond_featurizer.encode(bond))

		return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


# 	def scale_real_value_features(self, features, scaler):
# 		nb_features = (6 if (self.use_distance_stats and self.use_xyz) else 3)

# 		# Fit scaler if needed.
# 		from sklearn.utils.validation import check_is_fitted
# 		from sklearn.exceptions import NotFittedError
# 		try:
# 			check_is_fitted(scaler)
# 		except NotFittedError:
# 			# Retrieve all real value features.
# 			feats_real = np.empty((0, nb_features))
# 			for gdata in features:
# 				feats_real = np.concatenate((feats_real, gdata.node_features[:, -nb_features:]),
# 							axis=0)
# 			# Fit the scaler.
# 			scaler.fit(feats_real)

# # 			# Find min and max values.
# # 			min_feats_real = np.full((1, 3), np.inf)
# # 			max_feats_real = np.full((1, 3), -np.inf)


# # 			# Find min and max values.
# # 			min_feats_real = np.full((1, 3), np.inf)
# # 			max_feats_real = np.full((1, 3), -np.inf)
# # 			for gdata in features:
# # 				feats_real = gdata.node_features[:, -3:]
# # 				cur_min = feats_real.max(axis=0)
# # 				cur_max = feats_real.min(axis=0)

# 		# Transform data.
# 		for gdata in features:
# 			feats_real = gdata.node_features[:, -nb_features:]
# 			feats_real_fitted = scaler.transform(feats_real)
# 			gdata.node_features[:, -nb_features:] = feats_real_fitted


# 	@property
# 	def coords_scaler(self):
# # 		if hasattr(self, 'coords_scaling'):
# 		return self.coords_scaling
# # 		else:
# # 			raise ValueError('"coords_scaling" has not set yet. Initialize '
# # 					'`DCMolGraphFeaturizer` object with the argument '
# # 					'`coords_scaling` or set the argument `coords_scaler` in '
# # 					' the function `featurize`.')