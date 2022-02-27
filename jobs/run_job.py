#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:11:41 2021

@author: ljia
"""
import os
import re


cur_path = os.path.dirname(os.path.abspath(__file__))


def get_job_script(args):
# 		ds_name = args['ds_name']
# 		kernel = args['kernel']
# 		feature_scaling = args['feature_scaling']
# 		remove_sig_errs = args['remove_sig_errs']
		id_str = '.'.join([v for k, v in args.items()])

		script = r"""
#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name="octo.""" + id_str + r""""
#SBATCH --partition=tcourt # @todo: to change it back
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/octo.""" + id_str + r""".o%J"
#SBATCH --error="errors/octo.""" + id_str + r""".e%J"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00 # @todo: to change it back
#SBATCH --mem-per-cpu=4000

module load -s python3-DL/3.8.5
module list

hostname
cd """ + cur_path + r"""/../models/
echo Working directory : $PWD
echo Local work dir : $LOCAL_WORK_DIR
python3 run_xps.py """ + ' '.join([r"""--""" + k + r""" """ + v for k, v in args.items()])
		script = script.strip()
		script = re.sub('\n\t+', '\n', script)
		script = re.sub('\n +', '\n', script)

		return script


if __name__ == '__main__':

	os.makedirs('outputs/', exist_ok=True)
	os.makedirs('errors/', exist_ok=True)

	from sklearn.model_selection import ParameterGrid

	# Get task grid.
	DS_Name_List = ['poly200', 'thermo_exp', 'thermo_cal']
	Feature_Scaling_List = ['standard_y', 'minmax_y', 'none']
	Metric_List = ['RMSE', 'MAE', 'R2']
	# network structural hyperparameters.
	Model_List = ['GATModelExt', 'GraphConvModelExt', 'GraphConvModel', 'GCNModel', 'GATModel']
	Activation_Fn_List = ['relu', 'elu', 'leaky_relu', 'selu', 'gelu', 'linear',
# 					'exponetial',
					'tanh', 'softmax', 'sigmoid']#, 'normalize']
	Graph_Pool_List = ['max', 'none']
# 	N_Layer_List = [1, 2, 3, 4]
# 	Loss_List = []
	# network non-structural hyperparameters.
# 	Learning_Rate_List = []
	Batch_Norm_List = [False, True]
# 	Dropout_List = []
	Residual_List = [True, False]
	CV_List = ['811', '622']
# 	Batch_Size_List = []
	task_grid = ParameterGrid({
							'ds_name': DS_Name_List[0:1], # @todo: to change back.
							'feature_scaling': Feature_Scaling_List[0:1],
							'metric': Metric_List[1:2],
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

	# Run.
	for task in list(task_grid):
		print()
		print(task)
		job_script = get_job_script(task)
		command = 'sbatch <<EOF\n' + job_script + '\nEOF'
# 		print(command)
		os.system(command)
# 		os.popen(command)
# 		output = stream.readlines()
