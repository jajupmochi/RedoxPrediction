#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:45:26 2022

@author: ljia
"""
import os
import pickle
import numpy as np

path_kw = '/miccio/poly200+sugarmono/' # '/miccio/poly200', '/miccio/poly200+sugarmono/'
nb_trials = 30


def plot_real_vs_pred():
	# Get data.
	dir_ = '../outputs/' + path_kw
	x_train, x_valid, x_test = [], [], []
	y_train, y_valid, y_test = [], [], []
	for trial in range(1, nb_trials + 1):
		fn_vals = os.path.join(dir_, 'y_values.t' + str(trial) + '.pkl')
		with open(fn_vals, 'rb') as f:
			vals = pickle.load(f)
		x_train += np.array(vals['y_pred_train']).flatten().tolist()
		x_valid += np.array(vals['y_pred_valid']).flatten().tolist()
		x_test += np.array(vals['y_pred_test']).flatten().tolist()
		y_train += np.array(vals['y_train']).flatten().tolist()
		y_valid += np.array(vals['y_valid']).flatten().tolist()
		y_test += np.array(vals['y_test']).flatten().tolist()


	# Plot.
# 	import matplotlib
# 	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
# 	plt.plot(X, targets_pred, '+', X, targets_pred, '--', markersize=0.1, linewidth=0.1)
# 	plt.xlabel(descriptor)
# 	plt.ylabel('Emol_ref')
# 	plt.title(descriptor + ' vs. Emol_ref')
	plt.plot(x_train, y_train, 'bo',
		  x_valid, y_valid, 'ro',
		  x_test, y_test, 'mo',
		  markersize=1, linewidth=0.2)
	plt.axline((1, 1), color='k', slope=1)
	plt.xlabel('Predicted (K)')
	plt.ylabel('Real (K)')
	axes_lim_min = np.min([np.min(x_train), np.min(y_train),
						np.min(x_valid), np.min(y_valid),
						np.min(x_test), np.min(y_test)]) - 10
	axes_lim_max = np.max([np.max(x_train), np.max(y_train),
						np.max(x_valid), np.max(y_valid),
						np.max(x_test), np.max(y_test)]) + 10
	plt.xlim([axes_lim_min, axes_lim_max])
	plt.ylim([axes_lim_min, axes_lim_max])
	plt.gca().set_aspect('equal', adjustable='box')
# 	plt.title(descriptor)
	plt.legend(['train', 'valid', 'test'])
	fn_fig_pref = '../figures/' + path_kw + '/correlation'
	plt.savefig(fn_fig_pref + '.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight')
	plt.savefig(fn_fig_pref + '.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()


if __name__ == '__main__':
	plot_real_vs_pred()
