#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:00:31 2022

@author: ljia
"""

import os
import re

cur_path = os.path.dirname(os.path.abspath(__file__))

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
]
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
]
Descriptor_List = [
	'unlabeled', 'atom_bond_types', '1hot',
	'af1hot+3d-dis', '1hot+3d-dis',
	'mpnn', '1hot-dis'
]
X_Scaling_List = [
	'none', 'minmax', 'std'
]
Y_Scaling_List = [
	'none', 'minmax', 'std'
]

# prefix keyword # TODO
prefix_kw = 'redox_p'  # redox prediction

# The cluster used
infras = 'ubelix'  # 'criann'
"""Change the following script parts according to the cluster:
	- `--partition`: e.g., `tlong` for CRIANN, `epyc2` for UBELIX.
	- module loaded:
		- CRIANN/CPU: 
			module load python3-DL/keras/2.4.3-cuda10.1
			# CMake is needed for CRIANN as well.
		- UBELIX/CPU: 
			module load Python/3.8.6-GCCcore-10.2.0
			module load CMake
"""


def get_job_script(args, device='cpu'):
	id_str = '.'.join([v if isinstance(v, str) else str(v) for k, v in args.items()])
	# model = args['model']

	if device == 'gpu':
		script = get_job_script_gpu(id_str)


		def get_command(s):
			return 'sbatch <<EOF\n' + s + '\nEOF'

	elif device == 'cpu':
		script = get_job_script_cpu(id_str)


		def get_command(s):
			return 'sbatch <<EOF\n' + s + '\nEOF'

	elif device == 'cpu_local':
		print(args)
		script = get_job_script_cpu_local()
		import datetime
		now = datetime.datetime.now()
		fn_op = os.path.join(
			cur_path,
			'outputs/' + prefix_kw + '.' + id_str + '.o' + now.strftime(
				'%Y%m%d%H%M%S'
			)
		)


		def get_command(s):
			return s + ' > ' + fn_op

	elif device is None:
		script = ''

	# 	script += r"""
	# python3 run_xps.py """ + ' '.join([r"""--""" + k + r""" """ + v for k, v in args.items()]) + r""" --stratified """ + stratified
	script += r"""
python3 run_xps.py """ + ' '.join(
		['--' + k + ' ' + (v if isinstance(v, str) else str(v)) for k, v in args.items()]
	)
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)

	return get_command(script)


def get_job_script_gpu(id_str):
	# ubelix
	script = r"""
#!/bin/bash

# Not shared resources
##SBATCH --exclusive
#SBATCH --job-name=""" + '"' + prefix_kw + r""".""" + id_str + r""""
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/""" + prefix_kw + r""".""" + id_str + r""".o%J"
#SBATCH --error="errors/""" + prefix_kw + r""".""" + id_str + r""".e%J"
#
# GPUs architecture and number
# ----------------------------
#SBATCH --partition=gpu # @todo: to change it back p100, v100
## GPUs per compute node
##   gpu:4 (maximum) for gpu_k80
##   gpu:2 (maximum) for gpu_p100
#SBATCH --gres=gpu:gtx1080ti:1
# ----------------------------
# Job time (hh:mm:ss)
#SBATCH --time=24:00:00 # @todo: to change it back
##SBATCH --ntasks=1
##SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

# environments
# ---------------------------------
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Python/3.8.6-GCCcore-10.2.0
module load CMake
module list

hostname
cd """ + cur_path + r"""/../models/
echo Working directory : $PWD
echo Local work dir_file : $LOCAL_WORK_DIR
"""
	# 	script = r"""
	# #!/bin/bash

	# # Not shared resources
	# ##SBATCH --exclusive
	# #SBATCH --job-name=""" + '"' + prefix_kw + r""".""" + id_str + r""""
	# #SBATCH --mail-type=ALL
	# #SBATCH --mail-user=jajupmochi@gmail.com
	# #SBATCH --output="outputs/""" + prefix_kw + r""".""" + id_str + r""".o%J"
	# #SBATCH --error="errors/""" + prefix_kw + r""".""" + id_str + r""".e%J"
	# #
	# # GPUs architecture and number
	# # ----------------------------
	# #SBATCH --partition=gpu_p100 # @todo: to change it back p100, v100
	# # GPUs per compute node
	# #   gpu:4 (maximum) for gpu_k80
	# #   gpu:2 (maximum) for gpu_p100
	# ##SBATCH --gres gpu:4
	# #SBATCH --gres gpu:1
	# # ----------------------------
	# # Job time (hh:mm:ss)
	# #SBATCH --time=48:00:00 # @todo: to change it back
	# ##SBATCH --ntasks=1
	# ##SBATCH --nodes=1
	# #SBATCH --cpus-per-task=1
	# #SBATCH --mem-per-cpu=4G

	# # environments
	# # ---------------------------------
	# # module load cuda/9.0
	# #module load -s python3-DL/3.8.5
	# module load python3-DL/keras/2.4.3-cuda10.1
	# module list

	# hostname
	# cd """ + cur_path + r"""/../models/
	# echo Working directory : $PWD
	# echo Local work dir_file : $LOCAL_WORK_DIR
	# """

	return script


def get_job_script_cpu(id_str):
	script = r"""
#!/bin/bash

##SBATCH --exclusive
#SBATCH --job-name=""" + '"' + prefix_kw + r""".cpu.""" + id_str + r""""
#SBATCH --partition=epyc2 # @todo: to change it back
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/""" + prefix_kw + r""".cpu.""" + id_str + r""".o%J"
#SBATCH --error="errors/""" + prefix_kw + r""".cpu.""" + id_str + r""".e%J"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128 # @debug: to change it
#SBATCH --time=90:00:00 # @debug: to change it back
# Do not use values without a unit. In CRIANN, the default unit is MB; while in UBELIX, it is GB.
#SBATCH --mem-per-cpu=40G  # This value can not exceed 4GB on CRIANN.

module load Python/3.8.6-GCCcore-10.2.0
module load CMake
module list

hostname
cd """ + cur_path + r"""/../models/
echo Working directory : $PWD
echo Local work dir_file : $LOCAL_WORK_DIR
"""

	return script


def get_job_script_cpu_local():
	script = r"""
cd """ + cur_path + r"""/../models/
echo Working directory : $PWD
echo Local work dir_file : $LOCAL_WORK_DIR
"""
	return script


# TODO: this is not working correctly. Rewrite it and add it to `gklearn`.
def check_job_script(script, user='lj22u267'):
	"""
	Check the job name in the given script, to see if it is already submitted in
	the cluster by SLURM.
	"""
	import re
	pattern = re.compile(r"""--job-name="(.*)" """)
	match = pattern.search(script)
	if match is None:
		return False
	job_name = match.group(1)
	import subprocess
	cmd = 'squeue -u ' + user
	output = subprocess.check_output(cmd, shell=True)
	output = output.decode('utf-8')
	if job_name in output:
		return True
	else:
		return False


if __name__ == '__main__':
	os.makedirs('outputs/', exist_ok=True)
	os.makedirs('errors/', exist_ok=True)

	# debug: change these as needed.

	# general settings:
	device = 'cpu'
	settings = {
		'force_run': 'true',
		'read_resu_from_file': 1,
		'n_jobs_outer': 1,
		'n_jobs_params': 128,
	}

	# network settings:
	settings.update({
		'epochs_per_eval': 10,
		# 'tune_n_epochs': 'false',
	})

	# tasks:
	params_grid = {
		'dataset': Dataset_List[0:2],
		'model': Model_List[2:4],
		'descriptor': Descriptor_List[0:3],
		'x_scaling': X_Scaling_List[0:1],
		'y_scaling': Y_Scaling_List[0:1],
	}

	# # # This is for one by one running:
	# # params_grid = {
	# # }
	# # This is for auto grid running:  #  @TODO: to change accordingly.
	# params_grid = {
	# }
	# params_grid = [
	# 	# {
	# 	# 	**params_grid, **{
	# 	# 	'ds_name': ['brem_togn'],
	# 	# 	# for redox dataset only
	# 	# 	'tgt_name': ['dGox', 'dGred'],
	# 	# }
	# 	# },
	# 	{
	# 		**params_grid, **{
	# 		'ds_name': ['Acyclic']
	# 	}
	# 	},
	# ]

	# Run.
	from sklearn.model_selection import ParameterGrid

	cur_params = list(ParameterGrid(params_grid))
	print('Total number of jobs:', len(cur_params))

	from redox_prediction.utils.utils import remove_useless_keys

	for params in cur_params:
		params = remove_useless_keys(params)

		print()
		print('Current params: ', params)
	# 	cmd_args = {
	# 		'model': 'ged_knn',
	# 		# 'ds_version': 'v2',
	# 		# 'kfold_test': 'false',
	# 		# GED settings:
	# 		'ed_method': 'BIPARTITE',  # 'IPFP'
	# 		'optim_method': 'fitted',
	# 		# data descriptor:
	# 		# clustering settings
	# 		# for matching-graphs only
	# 		'mgs_model': 'original',
	# 		# ['none', 'original', 'random', 'iterative']
	# 		'mgs_mixing': 'add_dis',
	# 		# ['add_dis', 'augment_graphs', 'embed_sub', 'embed_ged']
	# 		# scaling
	# 	}
	# 	cmd_args.update(params)

		command = get_job_script({**params, **settings}, device=device)

		if check_job_script(command, user='lj22u267'):
			print('Job already submitted.')
		else:
			output = os.system(command)
