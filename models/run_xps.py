#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:03:26 2022

@author: ljia
"""


import sys
import os
# sys.path.append('../utils')
# os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(1, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, '../')
from dataset.load_dataset import load_dataset


def run_xp(smiles, y, output_result, mode, hyperparams, nb_epoch=100):
	from learning import xp_GCN

	resu = {}
	resu['dataset'] = hyperparams['ds_name']
# 	node_labels = list(list(Gn[0].nodes(data=True))[0][1].keys())
# 	edge_labels = list(list(Gn[0].edges(data=True))[0][2].keys())
# 	unlabeled = (len(node_labels) == 0 and len(edge_labels) == 0)

	### Get family names.
# 	if stratified:
# 		path_fam = 'datasets/redox_family.csv'
# 		stratified_y = family_by_target(path_fam, path_yn)
# 	else:
# 		stratified_y = None

	results = xp_GCN(smiles, y,
				  mode=mode,
				  nb_epoch=nb_epoch,
				  output_file=output_result,
# 				  ds_name=resu['dataset'],
				  **hyperparams)
# 	print('results: ', results)
	resu['results'] = results
	resu['mode'] = mode
	pickle.dump(resu, open(output_result, 'wb'))
	return output_result


def get_data(ds_name):
	if ds_name.lower() == 'poly200':
		data = load_dataset('polyacrylates200', format_='smiles')
		smiles = data['X']
		# smiles[192] = '(OCCCC)O.O=C=Nc1ccc(cc1)Cc1ccc(cc1)N=C=O'
		y = data['targets']
		smiles = [s for i, s in enumerate(smiles) if i not in [6]]
		y = [y for i, y in enumerate(y) if i not in [6]]
		y = np.reshape(y, (len(y), 1))
	elif ds_name.lower() == 'thermo_exp':
		data = load_dataset('thermophysical', format_='smiles', t_type='exp')
		smiles = data['X']
		y = data['targets']
		idx_rm = [168, 169, 170, 171, 172]
		smiles = [s for i, s in enumerate(smiles) if i not in idx_rm]
		y = [y for i, y in enumerate(y) if i not in idx_rm]
# 		import deepchem as dc
# 		featurizer = dc.feat.MolGraphConvFeaturizer()
# 		X_app = featurizer.featurize(smiles)
		y = np.reshape(y, (len(y), 1))
	elif ds_name.lower() == 'thermo_cal':
		data = load_dataset('thermophysical', format_='smiles', t_type='cal')
		smiles = data['X']
		y = data['targets']
		idx_rm = [151, 198, 199]
		smiles = [s for i, s in enumerate(smiles) if i not in idx_rm]
		y = [y for i, y in enumerate(y) if i not in idx_rm]
		y = np.reshape(y, (len(y), 1))
	else:
		raise ValueError('Dataset name %s can not be recognized.' % ds_name)

	return smiles, y


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
# 	parser.add_argument("dataset", help="path to / name of the dataset to predict")
	parser.add_argument('-g', '--path_G', type=str, default='datasets/redox_smiles.csv', help='path to the graphs to predict')

	parser.add_argument('-t', '--path_yn', type=str, help='path to the targets.')

	parser.add_argument('-o', '--output_file', type=str, help='path to file which will contains the results, the default is in the "outputs/" folder.')

	parser.add_argument("-u", "--unlabeled", help="Specify that the dataset is unlabeled graphs", action="store_true")

	parser.add_argument("-m", "--mode", type=str, choices=['reg', 'classif'],
						help="Specify if the dataset a classification or regression problem")

	parser.add_argument('-c', "--edit_cost", type=str, choices=['BIPARTITE', 'IPFP'], help='edit cost.')

	parser.add_argument("-y", "--y_distance", type=str, choices=['euclidean', 'manhattan', 'classif'], help="Specify the distance on y to fit the costs")

	parser.add_argument("-x", "--target", type=str, help="the name of targets/experiment")

	parser.add_argument("-s", "--stratified", type=str, help="whether to stratify the data or not.")

	parser.add_argument('-l', "--level", type=str, choices=['pbe', 'pbe0'], help='the level of chemical computational method.')

	# -------------------------------------------------------------------------

	parser.add_argument('-D', "--ds_name", type=str, help='the name of dataset')

	parser.add_argument("-f", "--feature_scaling", type=str, choices=['none', 'minmax', 'standard', 'minmax_x', 'standard_x', 'minmax_y', 'standard_y', 'minmax_xy', 'standard_xy'], help="the method to scale the input features")

	parser.add_argument('-M', "--model", type=str, help='the prediction model')

	parser.add_argument('-r', "--metric", type=str, help='the metric to measure the performance')

	parser.add_argument('-a', "--activation_fn", type=str, help='the activation function')

	parser.add_argument('-p', "--graph_pool", type=str, help='the graph pooling method')

	parser.add_argument('-C', "--cv", type=str, help='the CV protocol')

	args = parser.parse_args()

	return args

# 	dataset = args.dataset
# 	output_result = args.output_file
# 	unlabeled = args.unlabeled
# 	mode = args.mode

# 	print(args)
# 	y_distances = {
# 		'euclidean': euclid_d,
# 		'manhattan': man_d,
# 		'classif': classif_d
# 	}
# 	y_distance = y_distances['euclid']

# 	run_xp(dataset, output_result, unlabeled, mode, y_distance)
# 	print("Fini")


if __name__ == "__main__":

	import pickle
	import numpy as np

# 	from distances import euclid_d, man_d, classif_d
# 	y_distances = {
# 		'euclidean': euclid_d,
# 		'manhattan': man_d,
# 		'classif': classif_d
# 	}

	# Read arguments.
	args = parse_args()
# 	if len(sys.argv) > 1:
# 		run_from_args()
# 	else:
	from sklearn.model_selection import ParameterGrid

	### Load dataset target.
# 	path_yn = path = 'datasets/redox_deltaG.csv'
# 	path_yn = args.path_yn
# 	deltaG = load_deltaG(path_yn)

	# Get task grid.
# 	Level_List = (['low', 'pbe', 'pbe0'] if args.level is None else [args.level])
# 	Stratified_List = ([True, False] if args.stratified is None else [args.stratified == 'True'])
# 	Edit_Cost_List = (['BIPARTITE', 'IPFP'] if args.edit_cost is None else [args.edit_cost])
# 	Dis_List = (['euclidean', 'manhattan'] if args.y_distance is None else [args.y_distance])
# # 	Target_List = (list(deltaG.keys()) if args.target is None else [args.target])
# 	Target_List = (['dGred', 'dGox'] if args.target is None else [args.target])
	DS_Name_List = (['sugarmomo', 'poly200', 'thermo_exp', 'thermo_cal'] if args.ds_name is None else [args.ds_name])
	Feature_Scaling_List = (['standard_y', 'minmax_y', 'none'] if args.feature_scaling is None else [args.feature_scaling])
	Metric_List = (['MAE', 'RMSE', 'R2'] if args.metric is None else [args.metric])
	# network structural hyperparameters.
	Model_List = (['GATModelExt', 'GraphConvModelExt', 'GraphConvModel', 'GCNModel', 'GATModel'] if args.model is None else [args.model])
	Activation_Fn_List = (['relu', 'elu', 'leaky_relu', 'selu', 'gelu', 'linear',
# 					'exponetial',
					'tanh', 'softmax', 'sigmoid']#, 'normalize']
					   if args.activation_fn is None else [args.activation_fn])
	Graph_Pool_List = (['max', 'none'] if args.graph_pool is None else [args.graph_pool])
	# CV hyperparameters.
	CV_List = (['811', '622'] if args.cv is None else [args.cv])
	task_grid = ParameterGrid({
							'ds_name': DS_Name_List[1:2], # @todo: to change back.
							'feature_scaling': Feature_Scaling_List[0:1],
							'metric': Metric_List[0:1],
							# network structural hyperparameters.
							'model': Model_List[0:1],
							'activation_fn': Activation_Fn_List[0:],
							'graph_pool': Graph_Pool_List[0:1],
							'cv': CV_List[0:],
# 							'level': Level_List[0:],
# 							'stratified': Stratified_List[0:],
# 							'edit_cost': Edit_Cost_List[0:],
# 							'distance': Dis_List[0:],
# 							'target': Target_List
							})
# 	task_grid = [True]


# 	unlabeled = args.unlabeled # False # @todo: Not actually used. Automatically set in run_xp().
	mode = ('reg' if args.mode is None else args.mode)
	nb_epoch = 10000 # @todo: to change it back.
	# Run.

	for task in list(task_grid):
		print()
		print(task)

		### Load dataset.
		smiles, y = get_data(task['ds_name'])

		op_dir = '../outputs/gnn/'
		os.makedirs(op_dir, exist_ok=True)
# 		str_stratified = '.stratified' if task['stratified'] else ''
# 		output_result = op_dir + 'results.' + '.'.join([task['level'], task['edit_cost'], task['distance'], task['target']]) + str_stratified + '.shuffle.5folds.no_seed.pkl'
# 		output_result = op_dir + 'results.' + '.'.join([task['model']]) + '.shuffle.5folds.no_seed.pkl'
		output_result = op_dir + 'results.' + '.'.join([v for k, v in task.items()]) + '.random_cv.shuffle.5folds.no_seed.pkl'
# 		output_result = op_dir + 'results.shuffle.5folds.no_seed.pkl'
		print('output file: %s.' % output_result)

# 			if os.path.isfile(output_result):
		if os.path.isfile(output_result) and os.path.getsize(output_result) != 0: # @todo: to check it back
			# backup exsiting file.
			print('Backing up exsiting file...')
			from datetime import datetime
			result_bk = output_result + '.bk.' + datetime.now().strftime("%Y%m%d%H%M%S%f")
			from shutil import copyfile
			copyfile(output_result, result_bk)
# 		else:

		run_xp(smiles, y, output_result, mode, task, nb_epoch=nb_epoch)
