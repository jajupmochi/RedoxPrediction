#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:00:02 2022

@author: ljia
"""
import os
import pickle
import numpy as np


path_kw = '/miccio/poly200+sugarmono/' # '/miccio/poly200', '/miccio/poly200+sugarmono/'
nb_trials = 30


def get_datawise_errs():
	# Get data.
	dir_ = '../outputs/' + path_kw
	nb_data = ((195 + 28) if 'poly200+sugarmono' in path_kw else 195)
	all_errs, set_masks = np.zeros((nb_data, nb_trials)), np.empty((nb_data, nb_trials))
	for trial in range(1, nb_trials + 1):
		# Load data.
		fn_idx = os.path.join(dir_, 'indices_splits.t' + str(trial) + '.pkl')
		indices = pickle.load(open(fn_idx, 'rb'))
		fn_vals = os.path.join(dir_, 'y_values.t' + str(trial) + '.pkl')
		values = pickle.load(open(fn_vals, 'rb'))

		# Compute errors.
		for i_set, set_nm in enumerate(['train', 'valid', 'test']):
			cur_idx = np.array(indices[set_nm + '_index'])
			vals_real = np.array(values['y_' + set_nm]).flatten()
			vals_pred = np.array(values['y_pred_' + set_nm]).flatten()
			abs_errs = np.abs(vals_real - vals_pred)
			all_errs[cur_idx, trial-1] = abs_errs
			set_masks[cur_idx, trial-1] = i_set


	mean_errs = np.mean(all_errs, axis=1)
	idx_sorted = np.argsort(mean_errs)
	errs_sorted = mean_errs[idx_sorted]
	mean_err = np.mean(mean_errs)
	mean_err_worst = np.mean(errs_sorted[:-int(len(errs_sorted)*0.1):-1])
	print('mean_err: %.4f, mean_err_worst: %.4f.' % (mean_err, mean_err_worst))
	# For mixed data.
	if 'poly200+sugarmono' in path_kw:
		# poly200
		mean_errs = np.mean(all_errs[0:195, :], axis=1)
		idx_sorted = np.argsort(mean_errs)
		errs_sorted = mean_errs[idx_sorted]
		mean_err = np.mean(mean_errs)
		mean_err_worst = np.mean(errs_sorted[-int(len(errs_sorted)*0.1):])
		print('poly200: mean_err: %.4f, mean_err_worst: %.4f.' % (mean_err, mean_err_worst))
		# sugarmono
		mean_errs = np.mean(all_errs[195:, :], axis=1)
		idx_sorted = np.argsort(mean_errs)
		errs_sorted = mean_errs[idx_sorted]
		mean_err = np.mean(mean_errs)
		mean_err_worst = np.mean(errs_sorted[-int(len(errs_sorted)*0.1):])
		print('sugarmono: mean_err: %.4f, mean_err_worst: %.4f.' % (mean_err, mean_err_worst))

	return mean_errs


def plot_datawise_errs():
	# Get data.
	dir_ = '../outputs/' + path_kw
	all_res = {'x_train': [], 'y_train': [],
			'x_valid': [], 'y_valid': [],
			'x_test': [], 'y_test': []}
	for trial in range(1, nb_trials + 1):
		# Load data.
		fn_idx = os.path.join(dir_, 'indices_splits.t' + str(trial) + '.pkl')
		indices = pickle.load(open(fn_idx, 'rb'))
		fn_vals = os.path.join(dir_, 'y_values.t' + str(trial) + '.pkl')
		values = pickle.load(open(fn_vals, 'rb'))

		# Compute errors.
		for set_nm in ['train', 'valid', 'test']:
			all_res['x_' + set_nm] += indices[set_nm + '_index'].tolist()
			vals_real = np.array(values['y_' + set_nm]).flatten()
			vals_pred = np.array(values['y_pred_' + set_nm]).flatten()
			abs_errs = np.abs(vals_real - vals_pred)
			all_res['y_' + set_nm] += abs_errs.tolist()


	# Plot.
	import matplotlib.pyplot as plt
	import seaborn as sns
	colors = sns.color_palette('husl')[0:]
# 		sns.axes_style('darkgrid')
	sns.set_theme()
	fig = plt.figure(figsize=(15, 4))
	ax = fig.add_subplot(111)    # The big subplot for common labels

	plt.plot(all_res['x_train'], all_res['y_train'], 'bs',
		  all_res['x_valid'], all_res['y_valid'], 'rs',
		  all_res['x_test'], all_res['y_test'], 'ys',
		  markersize=2, linewidth=0.2)
	# mean errors.
	mean_errs = get_datawise_errs()
	plt.plot(range(len(mean_errs)), mean_errs, 'co-',
		  markeredgewidth=0.5, markersize=2.5, markeredgecolor='c', markerfacecolor='w', linewidth=0.5)

	plt.axhline(y=10.3, color='b', linestyle='--', linewidth=1)
	plt.axhline(y=29.1, color='r', linestyle='--', linewidth=1)
	plt.axhline(y=39.7, color='y', linestyle='--', linewidth=1)

	# For mixed data.
	if 'poly200+sugarmono' in path_kw:
		plt.axvline(x=194.5, color='orange', linestyle='--', linewidth=2)

	ax.set_xlabel('index of data')
	ax.set_ylabel('Absolute error (K)')
	ax.set_title('')
# 	fig.subplots_adjust(bottom=0.3)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(['train', 'valid', 'test', 'mean']) # , loc='lower center', ncol=3, frameon=False)
	fn_fig_pref = '../figures/' + path_kw + '/datawise_abs_errs'
	plt.savefig(fn_fig_pref + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fn_fig_pref + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()


if __name__ == '__main__':
 	# plot_datawise_errs()
 	get_datawise_errs()