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
	ds_name = args['ds_name']

	if device == 'gpu':
		script = get_job_script_gpu(args, id_str)
	elif device == 'cpu':
		script = get_job_script_cpu(args, id_str)
	elif device is None:
		script = ''

# 	script += r"""
# python3 run_xps.py """ + ' '.join([r"""--""" + k + r""" """ + v for k, v in args.items()]) + r""" --stratified """ + stratified
	script += r"""
python3 xp_""" + model + r""".py --ds_name """ + ds_name
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
#SBATCH --partition=court # @todo: to change it back
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/octo.""" + id_str + r""".o%J"
#SBATCH --error="errors/octo.""" + id_str + r""".e%J"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 # @todo: to change it
#SBATCH --time=48:00:00 # @todo: to change it back
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

# 	# Run.
# 	for task in list(task_grid):
# 		print()
# 		print(task)
	job_script = get_job_script(
		{'model': 'gat', 'ds_name': 'poly200r'}, device='cpu')  # @todo: to change it as needed.
	command = 'sbatch <<EOF\n' + job_script + '\nEOF'

# 		print(command)
	output = os.system(command)
# 		os.popen(command)
# 		output = stream.readlines()
