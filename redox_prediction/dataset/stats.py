"""
stats



@Author: linlin
@Date: 16.08.23
"""
import numpy as np

from redox_prediction.utils.logging import AverageMeter


def calculate_stats(results, verbose=True):
	"""
	Calculate statistics from results.

	The purpose of this function is to calculate the mean and standard deviation
	over the splits of each element in the results.

	Parameters
	----------
	results: list
		List of results. Each result/split is a dictionary containing at least
		the following keys:
		- 'perf_app': the performance on the training set.
		- 'perf_test': the performance on the test set.
		- 'best_history': the history of the best model.
		- 'history_app': the history of the training set.
		- 'history_test': the history of the test set.

		'best_history', 'history_app', and 'history_test' are dictionaries with
		the following keys:
		- 'predict': the performance of the prediction procedure. It is a dictionary
			with the following keys:
			- 'batch_time': the running time of the model for each data.
			- 'data_time': the running time of the data loading and operating for
			each data.
		Additionally, 'best_history' and 'history_app' have the following keys:
		- 'fit_metric': the performance of the metric model while fitting,
			which is a dictionary with the following keys:
			- 'batch_time_train': the running time of the model during training
			for each data.
			- 'data_time_train': the running time of the data loading and operating
			during training for each data.
			- 'metric_time_compute_train': the running time of computing the metric
			during training for each data.
			- 'metric_time_load_train': the running time of loading the metric
			during training for each data.
			- 'metric_time_total_train': the total running time of the metric
			during training for each data.
			- 'epoch_time': the running time of each epoch.
		- 'fit_task': the performance of the task model while fitting:
			- 'batch_time_train': the running time of the model during training
			for each data.
			- 'data_time_train': the running time of the data loading and operating
			during training for each data.
			- 'epoch_time': the running time of each epoch.
		Additionally, 'best_history' has the following keys:
		- 'fit_metric': the performance of the metric model while fitting,
			which is a dictionary with the following keys:
			- 'batch_time_valid': the running time of the model during validation
			for each data.
			- 'data_time_valid': the running time of the data loading and operating
			during validation for each data.
			- 'metric_time_compute_valid': the running time of computing the metric
			during validation for each data.
			- 'metric_time_load_valid': the running time of loading the metric
			during validation for each data.
			- 'metric_time_total_valid': the total running time of the metric
			during validation for each data.
		- 'fit_task': the performance of the task model while fitting:
			- 'batch_time_valid': the running time of the model during validation
			for each data.
			- 'data_time_valid': the running time of the data loading and operating
			during validation for each data.
		All these values are stored in a `redox_prediction.model.nn.logging.AverageMeter` object
		except for 'perf_app' and 'perf_test', which are stored in a float value.

	Returns
	-------
	stats: dict
		Dictionary containing the statistics of the results. It has the same key
		architecture as the input results, but the values are the statistics of
		the corresponding values in the input results, i.e., a list of two values
		(mean and standard deviation) for each value in the input results.

	Notes
	-----
		The dicts may include other keys, but they are not used in this function.
	"""
	# Initialize the stats:
	stats = {}

	# Calculate the stats:
	for key in results[0].keys():
		if key in ['perf_app', 'perf_train', 'perf_valid', 'perf_test']:
			stats[key] = AverageMeter(keep_all=True)
		elif key in [
			'best_history', 'history_app', 'history_train',
			'history_valid', 'history_test'
		]:
			stats[key] = {}
			for key2 in results[0][key].keys():
				if isinstance(results[0][key][key2], dict):
					stats[key][key2] = {}
					for key3 in results[0][key][key2].keys():
						stats[key][key2][key3] = AverageMeter(keep_all=True)
				else:
					stats[key][key2] = AverageMeter(keep_all=True)

	# Calculate the stats:
	for result in results:
		for key in result.keys():
			if key in ['perf_app', 'perf_train', 'perf_valid', 'perf_test']:
				stats[key].update(result[key])
			elif key in [
				'best_history', 'history_app', 'history_train',
				'history_valid', 'history_test'
			]:
				for key2 in result[key].keys():
					if isinstance(result[key][key2], dict):
						for key3 in result[key][key2].keys():
							stats[key][key2][key3].update(
								result[key][key2][key3].avg
							)
					else:
						stats[key][key2].update(result[key][key2].avg)

	# Calculate the mean and standard deviation:
	for key in stats.keys():
		if key in ['perf_app', 'perf_train', 'perf_valid', 'perf_test']:
			stats[key] = [stats[key].avg, stats[key].std]
		elif key in [
			'best_history', 'history_app', 'history_train',
			'history_valid', 'history_test'
		]:
			for key2 in stats[key].keys():
				if isinstance(stats[key][key2], dict):
					for key3 in stats[key][key2].keys():
						avg, std = stats[key][key2][key3].avg, stats[key][key2][
							key3].std
						stats[key][key2][key3] = [avg, std]
				else:
					avg, std = stats[key][key2].avg, stats[key][key2].std
					stats[key][key2] = [avg, std]

	# Print the stats:
	if verbose:
		print('\nStatistics of the results:')
		print_stats(stats)

	return stats


def print_stats(stats):
	"""
	Print the stats.

	Parameters
	----------
	stats: dict
		Dictionary containing the statistics of the results. It has the same key
		architecture as the input results, but the values are the statistics of
		the corresponding values in the input results, i.e., a list of two values
		(mean and standard deviation) for each value in the input results.

	Notes
	-----
		The dicts may include other keys, but they are not used in this function.
	"""
	# Print the stats:
	print()
	for key in stats.keys():
		if key in ['perf_app', 'perf_train', 'perf_valid', 'perf_test']:
			print(
				'{}:\t{:.2f} +- {:.2f}'.format(
					key, stats[key][0], stats[key][1]
				)
			)
		elif key in [
			'best_history', 'history_app', 'history_train',
			'history_valid', 'history_test'
		]:
			print('{}:'.format(key))
			for key2 in stats[key].keys():
				if isinstance(stats[key][key2], dict):
					print('  {}:'.format(key2))
					for key3 in stats[key][key2].keys():
						if stats[key][key2][key3][0] != 0 or \
								stats[key][key2][key3][
									1] != 0:
							print(
								'    {}:\t{:.9f} +- {:.9f}'.format(
									key3, stats[key][key2][key3][0],
									stats[key][key2][key3][1]
								)
							)
				else:
					if stats[key][key2][0] != 0 or stats[key][key2][1] != 0:
						print(
							'  {}:\t{:.9f} +- {:.9f}'.format(
								key2, stats[key][key2][0], stats[key][key2][1]
							)
						)


def save_results_to_csv(results, output_dir):
	"""
	Save results to csv file as human-readable format.

	Parameters
	----------
	results: list
		The results to be saved. This list is a concatenation of the results of
		all splits and the statistics of the results, i.e., the inputs and outputs
		of `redox_prediction.dataset.stats.calculate_stats`.

	output_dir: str
		The directory to save the results.

	Notes
	-----
		The dicts may include other keys, but they are not used in this function.
	"""
	# Save the results including running times using pandas:
	import pandas as pd
	import os

	# Create the output directory:
	os.makedirs(output_dir, exist_ok=True)

	# Save the stats:
	stats = results[-1]
	# stats is a nested dict, each value in a dict in stats is a list of two
	# values (mean and standard deviation). Here we save the stats to a csv file
	# in the format of mean +- std. Include row names and column names. For
	# a nested dict, expand it to multiple columns and rows.
	stats_to_save = {}
	for key in stats.keys():
		if key in ['perf_app', 'perf_train', 'perf_valid', 'perf_test']:
			stats_to_save[key] = '{:.2f} +- {:.2f}'.format(
				stats[key][0], stats[key][1]
			)
		elif key in [
			'best_history', 'history_app', 'history_train',
			'history_valid', 'history_test'
		]:
			stats_to_save[key] = {}
			for key2 in stats[key].keys():
				if isinstance(stats[key][key2], dict):
					stats_to_save[key][key2] = {}
					for key3 in stats[key][key2].keys():
						stats_to_save[key][key2][
							key3] = '{:.9f} +- {:.9f}'.format(
							stats[key][key2][key3][0], stats[key][key2][key3][1]
						)
				else:
					stats_to_save[key][key2] = '{:.9f} +- {:.9f}'.format(
						stats[key][key2][0], stats[key][key2][1]
					)

	stats_to_save = pd.DataFrame(stats_to_save)
	stats_to_save.to_csv(
		os.path.join(output_dir, 'stats.csv'), index=True, header=True
	)
