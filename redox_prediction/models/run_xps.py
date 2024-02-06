"""
run_xp



@Author: linlin
@Date: 26.09.23
"""

import sys

import networkx as nx
import numpy as np


def run_xp(
		ds_name, output_file, model_type, read_resu_from_file,
		# parallel,
		n_jobs_outer,
		# n_jobs_inner,
		n_jobs_params,
		**tasks
):
	from gklearn.dataset import Dataset
	from gklearn.experiments import DATASET_ROOT
	from learning import xp_main

	if ds_name.startswith('brem_togn'):
		from redox_prediction.dataset.get_data import get_get_data
		Gn, y_all = get_get_data(
			ds_name,
			tasks['descriptor'],
			model=tasks['model']
		)
		if tasks['model'].startswith('vc:'):
			# This is useless for vectorized models:
			node_labels = []
			edge_labels = []
		else:
			node_labels = list(Gn[0].nodes[list(Gn[0].nodes())[0]].keys())
			edge_labels = list(Gn[0].edges[list(Gn[0].edges())[0]].keys())
		node_attrs = []
		edge_attrs = []
	else:
		from redox_prediction.dataset.get_data import format_ds
		ds = Dataset(ds_name, root=DATASET_ROOT, verbose=True)
		ds = format_ds(ds, ds_name)
		# ds.cut_graphs(range(0, 20))  # debug: for debug only.
		Gn = ds.graphs
		y_all = ds.targets
		node_labels = ds.node_labels
		edge_labels = ds.edge_labels
		node_attrs = ds.node_attrs
		edge_attrs = ds.edge_attrs
	if not tasks['model'].startswith('vc:'):
		for i, G in enumerate(Gn):
			# Add id for each nx graph, for identification purposes in metric matrix:
			G.graph['id'] = i
			# Reorder nodes to prevent some bugs in some models:
			# todo: adjust the models
			nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

	resu = {}
	resu['task'] = task
	resu['dataset'] = ds_name
	unlabeled = (
			len(node_labels) == 0 and len(edge_labels) == 0
			and len(node_attrs) == 0 and len(edge_attrs) == 0
	)
	results = xp_main(
		Gn, y_all,
		model_type=model_type,
		unlabeled=unlabeled,
		node_labels=node_labels, edge_labels=edge_labels,
		node_attrs=node_attrs, edge_attrs=edge_attrs,
		ds_name=ds_name,
		output_file=output_file,
		read_resu_from_file=read_resu_from_file,
		# parallel=parallel,
		n_jobs_outer=n_jobs_outer,
		# n_jobs_inner=n_jobs_inner,
		n_jobs_params=n_jobs_params,
		**{
			**tasks,
			'output_dir': output_file[:-4] + '/'  # remove .pkl
		}
	)
	resu['results'] = results
	resu['unlabeled'] = unlabeled
	resu['model_type'] = model_type

	# Save results:
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	# Save to pkl:
	pickle.dump(resu, open(output_file, 'wb'))

	return resu, output_result


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()

	# ----- distinct tasks / experiments -----

	parser.add_argument('--dataset', type=str, help='the name of dataset')

	parser.add_argument(
		'--model', type=str, help='the model to use'
	)

	parser.add_argument(
		'--descriptor', type=str,
		choices=[
			'unlabeled', 'atom_bond_types', '1hot', 'af1hot+3d-dis',
			'1hot+3d-dis', 'mpnn', '1hot-dis'
		],
		help='the descriptor used for each graph as an input of ML models'
	)

	parser.add_argument(
		'--x_scaling', type=str,
		choices=['none', 'minmax', 'std'],
		help='the scaling method for the descriptors'
	)

	parser.add_argument(
		'--y_scaling', type=str,
		choices=['none', 'minmax', 'std'],
		help='the scaling method for the targets'
	)

	# ----- network settings -----

	parser.add_argument(
		'--epochs_per_eval', type=int, help='number of epochs per evaluation'
	)

	# parser.add_argument(
	# 	'--tune_n_epochs', type=str, choices=['true', 'false'],
	# 	help='whether to tune the number of epochs'
	# )

	# ----- other settings -----

	parser.add_argument(
		'-f', '--force_run', type=str, choices=['true', 'false'],
		help='Force to run the experiment, even if the results already exist.'
	)

	parser.add_argument(
		'--read_resu_from_file', type=int, choices=[0, 1, 2],
		help='Read the results from file, if it exists. 0: no; 1: yes, but only'
		     'the refitted model; 2: yes, and also the model before refitting.'
	)

	# parser.add_argument(
	# 	'--parallel', type=str, choices=['true', 'false'],
	# 	help='Whether to run the experiments in parallel.'
	# )

	parser.add_argument(
		'--n_jobs_outer', type=int,
		help='Number of jobs for outer loop.'
	)

	# parser.add_argument(
	# 	'--n_jobs_inner', type=int,
	# 	help='Number of jobs for inner loop.'
	# )

	parser.add_argument(
		'--n_jobs_params', type=int,
		help='Number of jobs for parameters tuning.'
	)

	# parser.add_argument(
	# 	'output_file', help='path to file which will contains the results'
	# )

	args = parser.parse_args()

	return args


if __name__ == "__main__":

	import pickle
	import os

	from redox_prediction.utils.utils import model_type_from_dataset

	# Read arguments.
	args = parse_args()

	# debug: change these as needed:

	# Basic settings.
	force_run = True if args.force_run is None else args.force_run == 'true'
	read_resu_from_file = (
		0 if args.read_resu_from_file is None else args.read_resu_from_file
	)
	# parallel = (False if args.parallel is None else args.parallel == 'true')
	n_jobs_outer = (1 if args.n_jobs_outer is None else args.n_jobs_outer)
	# n_jobs_inner = (1 if args.n_jobs_inner is None else args.n_jobs_inner)
	n_jobs_params = (1 if args.n_jobs_params is None else args.n_jobs_params)

	# Network settings.
	epochs_per_eval = 10 if args.epochs_per_eval is None else args.epochs_per_eval
	# if_tune_n_epochs = False if args.tune_n_epochs is None else args.tune_n_epochs == 'true'

	# Get task grid.
	from sklearn.model_selection import ParameterGrid

	Dataset_List = [
		# 'QM7', 'QM9',  # 0-1 Regression: big
		# 'Alkane_unlabeled', 'Acyclic',  # 2-3 Regression
		'brem_togn_dGred', 'brem_togn_dGox',  # 0-1 Regression: Redox
		# 'MAO', 'PAH', 'MUTAG', 'Monoterpens',  # 6-9 Jia thesis: mols
		# # 10-12 Jia thesis: letters
		# 'Letter-high', 'Letter-med', 'Letter-low',
		# 'PTC_FR',  # 13 Navarin2018 pre-training gnn with kernel paper
		# # 14-16 Fuchs2022 PR paper:
		# 'PTC_MR', 'IMDB-BINARY', 'COX2',
		# # 17-21 Jia thesis: bigger
		# 'ENZYMES', 'AIDS', 'NCI1', 'NCI109', 'DD',
		# 'Mutagenicity'  # 22 Fuchs2022 PR paper: bigger
		# 'HistoGraph',  # 23 Riba2018 paper
		# 'Chiral', 'Vitamin_D', 'Steroid'
	] if args.dataset is None else [args.dataset]
	Model_List = [
		# baselines:
		'vc:mean', 'vc:random',  # 0-1
		# "traditional" models:
		'vc:lr_s', 'vc:lr_c',  # 2-3
		'vc:gpr_s', 'vc:gpr_c',  # 4-5
		'vc:krr_s', 'vc:krr_c',  # 6-7
		'vc:svr_s', 'vc:svr_c',  # 8-9
		'vc:rf_s', 'vc:rf_c',  # 10-11
		'vc:xgb_s', 'vc:xgb_c',  # 12-13
		'vc:knn_s', 'vc:knn_c',  # 14-15
		# GEDs:
		'ged:bp_random', 'ged:bp_fitted',  # 16-17
		'ged:IPFP_random', 'ged:IPFP_fitted',  # 18-19
		# graph kernels, 20-24:
		'gk:sp', 'gk:structural_sp', 'gk:path', 'gk:treelet', 'gk:wlsubtree',
		# GNNs: 25-
		'nn:mpnn', 'nn:gcn', 'nn:gat', 'nn:dgcnn', 'nn:gin', 'nn:graphsage',
		'nn:egnn', 'nn:schnet', 'nn:diffpool', 'nn:transformer', 'nn:unimp',
	] if args.model is None else [args.model]
	Descriptor_List = [
		'unlabeled', 'atom_bond_types', '1hot',
		'af1hot+3d-dis', '1hot+3d-dis',
		'mpnn', '1hot-dis'
	] if args.descriptor is None else [args.descriptor]
	X_Scaling_List = [
		'none', 'minmax', 'std'
	] if args.x_scaling is None else [args.x_scaling]
	Y_Scaling_List = [
		'none', 'minmax', 'std'
	] if args.y_scaling is None else [args.y_scaling]

	if len(sys.argv) > 1:
		task_grid = ParameterGrid(
			{
				'dataset': Dataset_List[0:1],  # 'Redox'
				'model': Model_List[0:1],
				'descriptor': Descriptor_List[0:1],  # 'atom_bond_types'
				'x_scaling': X_Scaling_List[0:1],
				'y_scaling': Y_Scaling_List[0:1],
			}
		)
	else:  # This is mainly for testing purpose: debug
		task_grid = ParameterGrid(
			{
				'dataset': Dataset_List[0:1],  # 'Redox'
				'model': Model_List[9:10],
				'descriptor': Descriptor_List[1:2],  # 'atom_bond_types'
				'x_scaling': X_Scaling_List[0:1],
				'y_scaling': Y_Scaling_List[2:3],
			}
		)

	# Run.
	from redox_prediction.utils.utils import remove_useless_keys
	from contextlib import redirect_stdout
	from redox_prediction.utils.logging import StringAndStdoutWriter
	# from redox_prediction.utils.logging import PrintLogger

	for task in list(task_grid):

		task = remove_useless_keys(task)

		ab_path = os.path.dirname(os.path.abspath(__file__))
		output_result = ab_path + '/../outputs/results.' + '.'.join(
			list(task.values())
		) + '.pkl'

		# Redirect stdout to file:
		# logging_file = os.path.join(output_result[:-4], 'output.log')
		# os.makedirs(os.path.dirname(logging_file), exist_ok=True)
		# sys.stdout = PrintLogger(logging_file)

		# Redirect stdout to string:
		with redirect_stdout(StringAndStdoutWriter()) as op_str:

			print()
			print(task)

			model_type = model_type_from_dataset(task['dataset'])

			if not os.path.isfile(output_result) or force_run:
				resu, _ = run_xp(
					task['dataset'], output_result, model_type,
					epochs_per_eval=epochs_per_eval,
					read_resu_from_file=read_resu_from_file,
					# parallel=parallel,
					n_jobs_outer=n_jobs_outer,
					# n_jobs_inner=n_jobs_inner,
					n_jobs_params=n_jobs_params,
					# if_tune_n_epochs=if_tune_n_epochs,
					**{k: v for k, v in task.items() if k != 'dataset'}
				)
			else:
				resu = pickle.load(open(output_result, 'rb'))

			# Print results in latex format:
			from redox_prediction.dataset.compute_results import print_latex_results

			final_perf = print_latex_results(
				resu['results'][-1], model_type, rm_valid=True
			)
			final_perf['total_run_time'] = np.sum(
				[res['split_total_run_time'] for res in resu['results'][:-1]]
			)
			# Save to json:
			resu = {**{'final_perf': final_perf}, **resu}
			from redox_prediction.utils.logging import resu_to_serializable

			resu_json = resu_to_serializable(resu)
			import json

			with open(output_result[:-4] + '.json', 'w') as f:
				json.dump(resu_json, f, indent=4)
			print('\nResults are saved to {}.'.format(output_result))

			print("\nFini! Félicitations!")

		# Save the output string to file:
		op_str = op_str.getvalue()
		logging_file = os.path.join(output_result[:-4], 'output.log')
		os.makedirs(os.path.dirname(logging_file), exist_ok=True)
		with open(logging_file, 'w') as f:
			f.write(op_str)
