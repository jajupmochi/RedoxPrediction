"""
get_featurizer



@Author: linlin
@Date: 10.05.23
"""


def get_featurizer(descriptor='1hot', ds_dir='', use_atomic_charges='none'):
	from redox_prediction.dataset.feat import MolGNNFeaturizer
	af_allowable_sets = {
		'atom_type': (
			['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H'] if
			descriptor.endswith('+3d-dis') else
			['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
		),
		# 'atom_type': ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"],
		'formal_charge': None,  # [-2, -1, 0, 1, 2],
		'hybridization': ['SP', 'SP2', 'SP3'],
		# 	'hybridization': ['S', 'SP', 'SP2', 'SP3'],
		'acceptor_donor': ['Donor', 'Acceptor'],
		'aromatic': [True, False],
		'degree': [0, 1, 2, 3, 4, 5],
		# 'n_valence': [0, 1, 2, 3, 4, 5, 6],
		'total_num_Hs': [0, 1, 2, 3, 4],
		'chirality': ['R', 'S'],
	}
	add_3d_coords = descriptor.endswith('+3d-dis')
	if descriptor != 'af1hot+3d-dis':
		bf_allowable_sets = {
			'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
			'same_ring': [True, False],
			'conjugated': [True, False],
			'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'],
		}
		featurizer = MolGNNFeaturizer(
			use_edges=True,
			use_partial_charge=False,
			af_allowable_sets=af_allowable_sets,
			bf_allowable_sets=bf_allowable_sets,
			add_Hs=add_3d_coords,
			add_3d_coords=add_3d_coords,
			return_format='numpy',
		)
	else:
		featurizer = MolGNNFeaturizer(
			use_edges=False,
			use_partial_charge=False,
			use_atomic_charges=use_atomic_charges,
			af_allowable_sets=af_allowable_sets,
			bf_allowable_sets=None,
			add_Hs=add_3d_coords,
			add_3d_coords=add_3d_coords,
			coords_settings={  # Only works if add_3d_coords == True
				'in_type': 'smiles',
				'tool': 'balloon',
				'ds_dir': ds_dir,
				'with_GA': 'true',
				'return_mode': 'lowest_energy'
			},
			# @todo: atom_char_settings uses settings from coords_settings, so it
			# needs coords_settings to be set as well.
			atom_char_settings={  # Only works if use_atomic_charges == True
				'gaussian_method': 'm062x',
				'scaling': 'auto',
			},
			return_format='numpy',
		)
	return featurizer
