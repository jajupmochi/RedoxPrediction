#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:11:41 2021

@author: ljia
"""
import os
import re


cur_path = os.path.dirname(os.path.abspath(__file__))


def get_job_script(args, device='cpu'):
# 		ds_name = args['ds_name']
# 		kernel = args['kernel']
# 		feature_scaling = args['feature_scaling']
# 		remove_sig_errs = args['remove_sig_errs']
# 	str_stra = ('.stratified' if stratified == 'True' else '')
	id_str = '.'.join([v for k, v in args.items()])
	model = args['model']

	if device == 'gpu':
		script = get_job_script_gpu(args, id_str)
	elif device == 'cpu':
		script = get_job_script_cpu(args, id_str)
	elif device is None:
		script = ''

# 	script += r"""
# python3 run_xps.py """ + ' '.join([r"""--""" + k + r""" """ + v for k, v in args.items()]) + r""" --stratified """ + stratified
	script += r"""
python3 xp_""" + model + r""".py"""
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)

	return script


def get_job_script_gpu(args, id_str):

	script = r"""
#!/bin/bash

# Not shared resources
##SBATCH --exclusive
#SBATCH --job-name="octo.""" + id_str + r""""
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/octo.""" + id_str + r""".o%J"
#SBATCH --error="errors/octo.""" + id_str + r""".e%J"
#
# GPUs architecture and number
# ----------------------------
#SBATCH --partition=gpu_p100 # @todo: to change it back p100, v100
# GPUs per compute node
#   gpu:4 (maximum) for gpu_k80
#   gpu:2 (maximum) for gpu_p100
##SBATCH --gres gpu:4
#SBATCH --gres gpu:1
# ----------------------------
# Job time (hh:mm:ss)
#SBATCH --time=48:00:00 # @todo: to change it back
##SBATCH --ntasks=1
##SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000

# environments
# ---------------------------------
# module load cuda/9.0
#module load -s python3-DL/3.8.5
module load python3-DL/keras/2.4.3-cuda10.1
module list

hostname
cd """ + cur_path + r"""/../models/
echo Working directory : $PWD
echo Local work dir : $LOCAL_WORK_DIR
"""

	return script


def get_job_script_cpu(args, id_str):

	script = r"""
#!/bin/bash

##SBATCH --exclusive
#SBATCH --job-name="octo.""" + id_str + r""""
#SBATCH --partition=tcourt # @todo: to change it back
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/octo.""" + id_str + r""".o%J"
#SBATCH --error="errors/octo.""" + id_str + r""".e%J"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --time=24:00:00 # @todo: to change it back
#SBATCH --mem-per-cpu=4000

#module load -s python3-DL/3.8.5
module load python3-DL/keras/2.4.3-cuda10.1
module list

hostname
cd """ + cur_path + r"""/../models/
echo Working directory : $PWD
echo Local work dir : $LOCAL_WORK_DIR
"""

	return script


if __name__ == '__main__':

	os.makedirs('outputs/', exist_ok=True)
	os.makedirs('errors/', exist_ok=True)

# 	from sklearn.model_selection import ParameterGrid

# 	stratified = 'True'

# 	# Get task grid.
# 	DS_Name_List = ['poly200+sugarmono', 'sugarmono', 'poly200', 'thermo_exp', 'thermo_cal']
# 	Descriptor_List = ['smiles+dis_stats_obabel', 'smiles+xyz_obabel', 'smiles']
# 	Feature_Scaling_List = ['standard_y', 'minmax_y', 'none']
# 	Metric_List = ['MAE', 'RMSE', 'R2']
# 	# network structural hyperparameters.
# 	Model_List = ['GCNModelExt', 'GATModelExt', 'GraphConvModelExt', 'GraphConvModel', 'GCNModel', 'GATModel']
# 	Activation_Fn_List = ['relu', 'elu', 'leaky_relu', 'selu', 'gelu', 'linear',
# # 					'exponetial',
# 					'tanh', 'softmax', 'sigmoid']#, 'normalize']
# 	Graph_Pool_List = ['max', 'none']
# 	# CV hyperparameters.
# 	CV_List = ['811', '622']
# 	task_grid = ParameterGrid({
# 							'ds_name': DS_Name_List[2:3], # @todo: to change back.
# 							'descriptor': Descriptor_List[2:3],
# 							'feature_scaling': Feature_Scaling_List[0:1],
# 							'metric': Metric_List[0:1],
# 							# network structural hyperparameters.
# 							'model': Model_List[0:2],
# 							'activation_fn': Activation_Fn_List[0:],
# 							'graph_pool': Graph_Pool_List[0:1],
# 							'cv': CV_List[0:],
# # 							'level': Level_List[0:],
# # 							'stratified': Stratified_List[0:],
# # 							'edit_cost': Edit_Cost_List[0:],
# # 							'distance': Dis_List[0:],
# # 							'target': Target_List
# 							})

# 	# Run.
# 	for task in list(task_grid):
# 		print()
# 		print(task)
	job_script = get_job_script({'model': 'gat'}, device='cpu')  # @todo: to change it as needed.
	command = 'sbatch <<EOF\n' + job_script + '\nEOF'

# 		print(command)
	output = os.system(command)
# 		os.popen(command)
# 		output = stream.readlines()


# 		import subprocess
# 		try:
# 			subprocess.run([command], check=True)
# 		except subprocess.CalledProcessError:
# 			print('Error happened when using "sbatch". Use command line instead.')
# 			job_script = get_job_script(task, device=None)
# 			subprocess.run([job_script], check=True)
