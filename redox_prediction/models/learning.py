import os
import sys
import pickle

import time

import numpy as np

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from redox_prediction.utils.logging import PrintLogger
from redox_prediction.utils.resource import get_computing_resource_info


def split_data(D, y, train_index, test_index):
	D_app = [D[i] for i in train_index]
	D_test = [D[i] for i in test_index]
	y_app = [y[i] for i in train_index]
	y_test = [y[i] for i in test_index]
	return D_app, D_test, y_app, y_test


def evaluate_models(
		G_train, y_train, G_valid, y_valid, G_test, y_test,
		model_type='reg', unlabeled=False,
		descriptor='atom_bond_types',
		read_resu_from_file=1,
		# fit_test=False,
		**kwargs
):
	if kwargs.get('model').startswith('nn'):
		from redox_prediction.models.model_selection.gnn import evaluate_gnn
		return evaluate_gnn(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	# graph kernels:
	elif kwargs.get('model').startswith('gk'):
		from redox_prediction.models.model_selection.kernel import \
			evaluate_graph_kernel
		return evaluate_graph_kernel(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	# GEDs:
	elif kwargs.get('model').startswith('ged:'):
		from redox_prediction.models.model_selection.ged import evaluate_ged
		return evaluate_ged(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	elif kwargs.get('model').startswith('vc:'):
		from redox_prediction.models.model_selection.vector_model import \
			evaluate_vector_model
		return evaluate_vector_model(
			G_train, y_train, G_valid, y_valid, G_test, y_test,
			model_type=model_type,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			**kwargs
		)
	else:
		raise ValueError(
			'Unknown model: {0}.'.format(kwargs.get('model'))
		)


def process_split(
		i, app_index, test_index, G_all, y_all, model_type, unlabeled,
		descriptor, read_resu_from_file,
		**kwargs
):
	"""
	Process a single split.
	"""
	from contextlib import redirect_stdout
	from redox_prediction.utils.logging import StringAndStdoutWriter

	with redirect_stdout(StringAndStdoutWriter()) as op_str:
		print()
		print('----- Split {0}/{1} -----'.format(i + 1, 10))

		print('\nComputing resource info:')
		print(get_computing_resource_info(return_json=True, return_joblib=True))

		start = time.time()

		# Get split data
		G_app, G_test, y_app, y_test = split_data(
			G_all, y_all, app_index, test_index
		)

		# Split evaluation set
		valid_size = 0.1 / (1 - 0.1)
		op_splits = train_test_split(
			G_app, y_app, app_index,
			test_size=valid_size,
			random_state=0,
			shuffle=True,
			stratify=None
		)
		G_train, G_valid, y_train, y_valid, _, _ = op_splits

		cur_results = evaluate_models(
			G_train, np.array(y_train),
			G_valid, np.array(y_valid),
			G_test, np.array(y_test),
			model_type=model_type, unlabeled=unlabeled,
			descriptor=descriptor,
			read_resu_from_file=read_resu_from_file,
			n_classes=len(np.unique(y_all)) if model_type == 'classif' else None,
			**{**kwargs, 'split_idx': i + 1}
		)

		run_time = time.time() - start
		cur_results = cur_results + (run_time, y_train, y_valid, y_test)

	# Save the output string to file:
	op_str = op_str.getvalue()
	logging_file = os.path.join(
		kwargs.get('output_dir'), 'split_%d.log' % (i + 1)
	)
	os.makedirs(os.path.dirname(logging_file), exist_ok=True)
	with open(logging_file, 'w') as f:
		f.write(op_str)

	return i, cur_results


def xp_main(
		Gn,
		y_all,
		model_type: str = 'reg',
		unlabeled: bool = False,
		descriptor='atom_bond_types',
		output_file: str = None,
		read_resu_from_file: int = 1,
		# parallel: bool = False,
		n_jobs_outer: int = 1,
		# n_jobs_inner: int = 1,
		n_jobs_params: int = 1,
		**kwargs
):
	"""
	Perform a knn regressor on given dataset
	"""
	n_splits = 10
	stratified = False
	if model_type == 'classif':
		stratified = False  # debug

	if stratified:
		rs = StratifiedShuffleSplit(
			n_splits=n_splits, test_size=.1, random_state=0
		)
	else:
		# 		rs = ShuffleSplit(n_splits=10, test_size=.1) #, random_state=0)
		rs = ShuffleSplit(n_splits=n_splits, test_size=.1, random_state=0)

	if stratified:
		split_scheme = rs.split(Gn, y_all)
	else:
		split_scheme = rs.split(Gn)

	# Load existing results if possible.
	if read_resu_from_file and output_file is not None and os.path.isfile(
			output_file
	) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results = pickle.load(file)['results']
	else:
		results = [None] * n_splits

	if n_jobs_outer > 1:
		# todo: skip existing results.

		# n_jobs = n_splits + 1

		print('\nDistributing the outer CV loop to %d workers:' % n_jobs_outer)

		# 1. Use joblib with Dask backends: this does not work for multiple nodes
		# on a cluster:
		#
		# from joblib import parallel_backend, Parallel, delayed
		# from dask.distributed import Client
		#
		# with Client(n_workers=n_jobs_outer) as client:
		# 	with parallel_backend('dask'):
		#
		# 		print('Dask Client: %s' % client)
		# 		print('Dask dashboard link: %s' % client.dashboard_link)
		#
		# 		results = Parallel()(
		# 			delayed(process_split)(
		# 				i, app_index, test_index, Gn, y_all, model_type, unlabeled,
		# 				descriptor, read_resu_from_file,
		# 				# n_jobs_inner=n_jobs_inner,
		# 				n_jobs_params=n_jobs_params,
		# 				**kwargs
		# 			) for i, (app_index, test_index) in enumerate(split_scheme)
		# 		)

		# 2. Use Dask directly. This creates multiple dask threads, but does not
		# work for multiple nodes on a cluster:
		#
		# import dask
		#
		# results = [dask.delayed(process_split)(
		# 		i, app_index, test_index, Gn, y_all, model_type, unlabeled,
		# 		descriptor, read_resu_from_file,
		# 		# n_jobs_inner=n_jobs_inner,
		# 		n_jobs_params=n_jobs_params,
		# 		**kwargs
		# 	) for i, (app_index, test_index) in enumerate(split_scheme)
		# ]
		# dask.compute(results)

		# 3. Use dask_jobqueue to submit jobs to SLURM. This will submit work on
		# each node as a separate job, and each job will have multiple threads.
		# This can be problematic when the number of nodes is large, where some
		# jobs will have to wait in a queue for a long time.
		#
		# if 'SLURM_JOB_ID' in os.environ:
		# 	from dask_jobqueue import SLURMCluster
		#
		# 	cluster = SLURMCluster(
		# 		queue='epyc2',
		# 		#	    project='linlin',
		# 		cores=min(n_jobs_params, 128),
		# 		memory='64GB',
		# 		walltime='01:00:00'
		# 	)
		# 	cluster.scale(jobs=n_jobs_outer)
		# else:
		# 	cluster = None
		#
		# import dask
		# from dask.distributed import Client
		#
		# with Client(cluster, n_workers=n_jobs_outer) as client:
		#
		# 	print('Dask Client: %s' % client)
		# 	print('Dask dashboard link: %s' % client.dashboard_link)
		#
		# 	results = [dask.delayed(process_split)(
		# 			i, app_index, test_index, Gn, y_all, model_type, unlabeled,
		# 			descriptor, read_resu_from_file,
		# 			# n_jobs_inner=n_jobs_inner,
		# 			n_jobs_params=n_jobs_params,
		# 			**kwargs
		# 		) for i, (app_index, test_index) in enumerate(split_scheme)
		# 	]
		# 	results = dask.compute(*results)
		#
		# 	# results = client.map(
		# 	# 	process_split,
		# 	# 	[
		# 	# 		(
		# 	# 			i, app_index, test_index, Gn, y_all, model_type,
		# 	# 			unlabeled,
		# 	# 			descriptor, read_resu_from_file,
		# 	# 			# n_jobs_inner=n_jobs_inner,
		# 	# 			n_jobs_params=n_jobs_params,
		# 	# 			**kwargs
		# 	# 		) for i, (app_index, test_index) in enumerate(split_scheme)
		# 	# 	]
		# 	# )
		# 	# results = client.gather(results)

		# 4. Use dask_mpi library to deploy Dask from within an existing MPI
		# environment:

		# # Check if `mpi4py` is installed:
		# import importlib
		# if importlib.util.find_spec('mpi4py') is not None:
		#
		# 	# Check if the current process is running under MPI:
		# 	# import mpi4py
		# 	from mpi4py import MPI
		#
		# 	comm = MPI.COMM_WORLD
		# 	rank = comm.Get_rank()
		# 	size = comm.Get_size()
		#
		# 	print('MPI rank: %d' % rank)
		# 	print('MPI size: %d' % size)
		#
		# 	if size > 1:
		# 		from dask_mpi import initialize
		#
		# 		initialize()

		# import dask
		# from dask.distributed import Client
		#
		# with Client(n_workers=n_jobs_outer) as client:
		#
		# 	print('Dask Client: %s' % client)
		# 	print('Dask dashboard link: %s' % client.dashboard_link)
		#
		# 	paral_results = [dask.delayed(process_split)(
		# 			i, app_index, test_index, Gn, y_all, model_type, unlabeled,
		# 			descriptor, read_resu_from_file,
		# 			# n_jobs_inner=n_jobs_inner,
		# 			n_jobs_params=n_jobs_params,
		# 			**kwargs
		# 		) for i, (app_index, test_index) in enumerate(split_scheme)
		# 	]
		# 	paral_results = dask.compute(*paral_results)

		# 5. Use MPI directly with joblib:

		# Check if we are using Slurm and `mpi4py` is installed. If so, we will
		# use MPI to parallelize the outer CV loop on the cluster over multiple
		# nodes:
		import importlib
		if 'SLURM_JOB_ID' in os.environ and importlib.util.find_spec(
				'mpi4py') is not None:

			# Check if the current process is running under MPI:
			# import mpi4py
			from mpi4py import MPI

			comm = MPI.COMM_WORLD
			rank = comm.Get_rank()
			size = comm.Get_size()

			print('MPI rank: %d' % rank)
			print('MPI size: %d' % size)

			# List of tasks:
			tasks = list(enumerate(split_scheme))

			# Determine the chunk of tasks for this process
			chunk_size = len(tasks) // size
			start_idx = rank * chunk_size
			end_idx = (rank + 1) * chunk_size if rank < size - 1 else len(tasks)

			# Each process handles a different chunk of tasks
			tasks = tasks[start_idx:end_idx]

			# Print the tasks each process will handle
			print('Process ranked {} will handle outer CV # {} to # {}.'.format(
				rank, start_idx, end_idx - 1
			))

			from joblib import parallel_backend, Parallel, delayed

			# Define a function to process a single task using the provided code
			def process_task(task):
				task_id, (app_index, test_index) = task
				with parallel_backend(
						'loky', inner_max_num_threads=n_jobs_params,
				):
					return process_split(
						app_index, test_index, Gn, y_all, model_type, unlabeled,
						descriptor, read_resu_from_file,
						n_jobs_params=n_jobs_params,
						**kwargs
					)

			# Parallelize the task processing using Joblib within the MPI process
			results = Parallel(n_jobs=-1)(
				delayed(process_task)(task) for task in tasks
			)

		# Otherwise we will use Joblib to parallelize the outer CV loop on a
		# single node:
		else:
			from joblib import parallel_backend, Parallel, delayed
			import multiprocessing

			# Compute the maximum number of threads to use for each task:
			inner_max_num_threads = min(n_jobs_params, multiprocessing.cpu_count() - 2)
			# Compute the maximum number of tasks to be parallelized:
			outer_max_num_threads = min(
				n_jobs_outer, (multiprocessing.cpu_count() - 1) // (inner_max_num_threads + 1)
			)

			with parallel_backend(
					'loky', inner_max_num_threads=inner_max_num_threads,
			):
				results = Parallel(n_jobs=outer_max_num_threads)(
					delayed(process_split)(
						i, app_index, test_index, Gn, y_all, model_type, unlabeled,
						descriptor, read_resu_from_file,
						n_jobs_params=n_jobs_params,
						**kwargs
					) for i, (app_index, test_index) in enumerate(split_scheme)
				)

		# results.visualize()

		# print(client.scheduler_info())
		# print(client.call_stack())
		# print(client.get_scheduler_logs())
		# print(client.get_task_stream())
		# print(client.get_versions(check=True))
		# print(client.get_worker_logs())
		# print(client.has_what())
		# print(client.list_datasets())
		# print(client.nbytes())
		# print(client.profile())

		# Sort results by split index:
		sorted_results = sorted(results, key=lambda x: x[0])

		# logging_file = os.path.join(kwargs['output_dir'], 'output.log')
		# sys.stdout = PrintLogger(logging_file)

		# Process the sorted results
		for i, cur_results in sorted_results:
			append_split_perf(results, i, cur_results)
			# perf_util_now(results)
			save_split_to_file(results, read_resu_from_file, output_file, i)

	else:
		print('\nRunning the outer CV loop sequentially:')

		for i, (app_index, test_index) in enumerate(split_scheme):
			# if i > 0:  # debug: for debug only.
			# 	break

			# Check if the split already computed.
			if results[i] is not None:
				print()
				print('----- Split {0}/{1} -----'.format(i + 1, 10))
				print('Skip.')
				continue

			i, cur_results = process_split(
				i, app_index, test_index, Gn, y_all, model_type, unlabeled,
				descriptor, read_resu_from_file,
				n_jobs_params=n_jobs_params,
				**kwargs
			)

			append_split_perf(results, i, cur_results)

			# # Show mean and std of performance on app and test set until now:
			perf_util_now(results)

			# Save the split:
			save_split_to_file(results, read_resu_from_file, output_file, i)

	# Print run time:
	total_run_time = np.sum(
		[res['split_total_run_time'] for res in results[:-1]]
	)
	print()
	print('Total running time of all CV trials: %.2f seconds.' % total_run_time)
	# print(
	# 	'Attention: the running time might be inaccurate when some parts of the '
	# 	'code are executed a priori. Please check the saved results for the '
	# 	'actual running time.'
	# )

	if len(results) == 10:
		# Calculate mean and standard deviation over splits:
		from redox_prediction.dataset.stats import calculate_stats, \
			save_results_to_csv
		stats = calculate_stats(results)
		results.append(stats)
		save_results_to_csv(results, kwargs['output_dir'])

	return results


# %%


def append_split_perf(results, split_idx, results_to_save):
	"""
	Save performance for split into results.
	"""
	cur_results = {}
	for i, key in enumerate(
			[
				'model',
				'perf_train', 'perf_valid', 'perf_test',
				'y_pred_train', 'y_pred_valid', 'y_pred_test',
				'best_history', 'history_train', 'history_valid',
				'history_test',
				'params_best',
				'split_total_run_time',
				'y_true_train', 'y_true_valid', 'y_true_test'
			]
	):
		cur_results[key] = results_to_save[i]
	results[split_idx] = cur_results


def perf_util_now(results):
	results = [res for res in results if res is not None]
	print()
	print(
		'performance until now:\t'
		'Train:\t{:0.3f} $\\pm$ {:0.3f}\t'
		'Valid:\t{:0.3f} $\\pm$ {:0.3f}\t'
		'Test:\t{:0.3f} $\\pm$ {:0.3f}'.format(
			np.mean([results[i]['perf_train'] for i in range(len(results))]),
			np.std([results[i]['perf_train'] for i in range(len(results))]),
			np.mean([results[i]['perf_valid'] for i in range(len(results))]),
			np.std([results[i]['perf_valid'] for i in range(len(results))]),
			np.mean([results[i]['perf_test'] for i in range(len(results))]),
			np.std([results[i]['perf_test'] for i in range(len(results))]),
		)
	)


def save_split_to_file(results, read_resu_from_file, output_file, i):
	# Check if the (possible) existing file was updated by another thread
	# during the computation of this split:
	if read_resu_from_file and output_file is not None and os.path.isfile(
			output_file
	) and os.path.getsize(output_file) > 0:
		with open(output_file, 'rb') as file:
			results_new = pickle.load(file)['results']
			if results_new[i] is not None:
				print('Skip.')
				return

	# If not, save the results.
	if output_file is not None:
		pickle.dump({'results': results}, open(output_file, 'wb'))
