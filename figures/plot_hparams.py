#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:05:10 2022

@author: ljia
"""
import os
import pickle
import numpy as np

def plot_hparams(fn):
	# Plot.
	import matplotlib.pyplot as plt
	import seaborn as sns
	colors = sns.color_palette('husl')[0:]
# 		sns.axes_style('darkgrid')
	sns.set_theme()
	fig = plt.figure(figsize=(15, 15))

	for idx, trial_index in enumerate(range(1, 10)):

		fn = '../outputs/' + path_kw + 'mae_all.t' + str(trial_index) + '.pkl'
		MAE_all = pickle.load(open(fn, 'rb'))

		# Organize data.
		alphas = sorted(list(set([item[0]['alpha'] for item in MAE_all])))
		heights = sorted(list(set([item[0]['height'] for item in MAE_all])))
# 		X, Y = np.meshgrid(np.log10(alphas), heights)
# 		Z = np.empty((len(heights), len(alphas)))
# 		for item in MAE_all:
# 			h, a = item[0]['height'], item[0]['alpha']
# 			i_h, i_a = heights.index(h), alphas.index(a)
# 			Z[i_h, i_a] = item[1]
		X, Y = np.meshgrid(heights, np.log10(alphas))
		Z = np.empty((len(alphas), len(heights)))
		for item in MAE_all:
			h, a = item[0]['height'], item[0]['alpha']
			i_h, i_a = heights.index(h), alphas.index(a)
			Z[i_a, i_h] = item[1]

		ax = fig.add_subplot(int(331 + idx), projection='3d')    # The big subplot for common labels

		ax.plot_surface(X, Y, Z, cmap='plasma')
# 		ax.plot_wireframe(X, Y, Z, cmap='plasma')
		ax.set_xlabel('heights')
		ax.set_ylabel('alphas (in log)')
		ax.set_zlabel('MAE (K)')
		ax.set_title('Hyperparams v.s. MAEs')
	# 	ax.set_xscale('log')
	# 	fig.subplots_adjust(bottom=0.3)
	# 	handles, labels = ax.get_legend_handles_labels()
	# 	ax.legend(['train', 'valid', 'test', 'mean']) # , loc='lower center', ncol=3, frameon=False)

	fn_fig_pref = '../figures/' + path_kw + '/hparams_vs_maes'
	os.makedirs(os.path.dirname(fn_fig_pref), exist_ok=True)
	plt.savefig(fn_fig_pref + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fn_fig_pref + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()


if __name__ == '__main__':
	path_kw = '/treelet_kernel_krr/1hot/'
	plot_hparams(path_kw)