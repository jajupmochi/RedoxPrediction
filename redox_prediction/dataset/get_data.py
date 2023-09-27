"""
get_data



@Author: linlin
@Date: 14.05.23
"""


def get_get_data(ds_name, descriptor):
	if ds_name.startswith('brem_togn'):
		ds_name = ds_name.split('_')
		target = ds_name[2]
		ds_name = ds_name[0] + '_' + ds_name[1]

	from redox_prediction.dataset.load_dataset import get_data

	ds_kwargs = {
		'level': 'pbe0', 'target': target,
		'sort_by_targets': True,
		'fn_families': None,
	}
	ds_dir_feat = '../datasets/Redox/'

	if descriptor == 'atom_bond_types':
		graphs, y, _ = get_data(
			ds_name, descriptor='smiles', format_='networkx', **ds_kwargs
		)

	# 2. Get descriptors: one-hot descriptors on nodes and edges.
	elif descriptor in ['1hot', '1hot-dis', 'af1hot+3d-dis', '1hot+3d-dis']:
		from redox_prediction.dataset.get_featurizer import get_featurizer

		featurizer = get_featurizer(
			descriptor, ds_dir=ds_dir_feat,
			use_atomic_charges='none'
		)
		# The string is used for the GEDLIB module of the graphkit-learn library.
		ds_kwargs['feats_data_type'] = 'str'
		graphs, y, _ = get_data(
			ds_name, descriptor=featurizer, format_='networkx',
			coords_dis=(True if descriptor.endswith('+3d-dis') else False),
			**ds_kwargs
		)

	else:
		raise NotImplementedError(
			'`get_data` for Descriptor {} is not implemented.'.format(
				descriptor
			)
		)

	return graphs, y


def format_ds(ds, ds_name):
	if ds_name not in ['Letter-high', 'Letter-med', 'Letter-low']:
		ds.remove_labels(
			node_attrs=ds.node_attrs, edge_attrs=ds.edge_attrs
		)  # @todo: ged can not deal with sym and unsym labels at the same time.
	else:
		# For Shortest Path Kernel:
		ds.trim_dataset(edge_required=True)

	if ds_name == 'MAO':
		ds.remove_labels(edge_labels=['bond_stereo'])
		return ds

	return ds
