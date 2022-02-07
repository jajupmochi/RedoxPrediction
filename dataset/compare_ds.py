#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:06:05 2022

@author: ljia
"""

def compare_ds():
	import os
	import sys
	sys.path.insert(1, '../')

	### Load datasets.
	from dataset.load_dataset import load_dataset
	# Load polyacrylates200 (from the paper.)
	ds_poly200 = load_dataset('polyacrylates200', format_='smiles')
	ds_poly200['X'] = [s for i, s in enumerate(ds_poly200['X']) if i not in [6]]
	ds_poly200['targets'] = [y for i, y in enumerate(ds_poly200['targets']) if i not in [6]]
	# Load thermophysical (from the website).
	ds_thermo_cal = load_dataset('thermophysical', format_='smiles', t_type='cal')
	ds_thermo_exp = load_dataset('thermophysical', format_='smiles', t_type='exp')


	### Sort datasets.
	smiles, y_poly200, y_thermo_cal, y_thermo_exp = [], [], [], []
	for idx, sm in enumerate(ds_poly200['X']):
		if not sm in ds_thermo_cal['X'] or not sm in ds_thermo_exp['X']:
			continue
		smiles.append(sm)
		y_poly200.append(ds_poly200['targets'][idx])
		idx_thermo_cal = ds_thermo_cal['X'].index(sm)
		y_thermo_cal.append(ds_thermo_cal['targets'][idx_thermo_cal])
		idx_thermo_exp = ds_thermo_exp['X'].index(sm)
		y_thermo_exp.append(ds_thermo_exp['targets'][idx_thermo_exp])


	### Get names.
	from dataset.load_dataset import load_polyacrylates200
	df_poly200 = load_polyacrylates200()
	ds_tmp = {'names': [], 'smiles': []}
	ds_tmp['smiles'] = [i.replace(' ', '') for i in df_poly200.iloc[:, 3].to_list()]
	ds_tmp['names'] = [i for i in df_poly200.iloc[:, 1].to_list()]
	names = []
	for idx, sm in enumerate(smiles):
		nm_idx = ds_tmp['smiles'].index(sm)
		names.append(ds_tmp['names'][nm_idx])


	### Compute results.
	import pandas as pd
	df = pd.DataFrame(columns=['name', 'smiles', 'Tg: from paper', 'Tg: preferred (exp)', 'Tg: preferred (cal)', 'Tg: paper-exp', 'Tg: paper-cal', 'Tg: exp-cal'])
	for idx, sm in enumerate(smiles):
		diff_paper_exp = y_poly200[idx] - y_thermo_exp[idx]
		diff_paper_cal = y_poly200[idx] - y_thermo_cal[idx]
		diff_exp_cal = y_thermo_exp[idx] - y_thermo_cal[idx]
		info = [names[idx],
		  sm, y_poly200[idx], y_thermo_exp[idx], y_thermo_cal[idx],
		  diff_paper_exp, diff_paper_cal, diff_exp_cal]
		df.loc[len(df)] = info
	# averages.
	mean = list(df.mean(axis=0))
	ave = ['', 'averages:'] + mean
	df.loc[len(df)] = ave


	### Save resluts to file.
	fname = '../datasets/compare_ds.csv'
	os.makedirs(os.path.dirname(fname), exist_ok=True)
	df.to_csv(fname)


	### Plot correlations.
# 	import numpy as np
# 	idx_diff = np.argsort(np.abs(np.subtract(y_poly200, y_thermo_exp)))
# 	y_poly200_cor = [y for i, y in enumerate(y_poly200) if i not in idx_diff[-3:]]
# 	y_thermo_exp_cor = [y for i, y in enumerate(y_thermo_exp) if i not in idx_diff[-3:]]
# 	get_correlations(y_poly200_cor, y_thermo_exp_cor, 'y_poly200_exp', 'y_thermo_exp')

	get_correlations(y_poly200, y_thermo_exp, 'y_poly200_exp', 'y_thermo_exp')
	get_correlations(y_poly200, y_thermo_cal, 'y_poly200_exp', 'y_thermo_cal')
	get_correlations(y_thermo_cal, y_thermo_exp, 'y_thermo_cal', 'y_thermo_exp')


def get_correlations(y1, y2, y1_name, y2_name):


	def set_figure(nb_rows):
		#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	#	plt.rc('axes', titlesize=15)     # fontsize of the axes title
	#	plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
	#	plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
	#	plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
	#	plt.rc('legend', fontsize=15)    # legend fontsize
	#	plt.rc('figure', titlesize=15)  # fontsize of the figure title

		#fig, _ = plt.subplots(2, 2, figsize=(13, 12))
		#ax1 = plt.subplot(221)
		#ax2 = plt.subplot(222)
		#ax3 = plt.subplot(223)
		#ax4 = plt.subplot(224)
		fig = plt.figure(figsize=(6, 3 * nb_rows + 0.56))
		ax = fig.add_subplot(111)    # The big subplot for common labels

		# Turn off axis lines and ticks of the big subplot
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.xaxis.set_ticks_position('none')
		ax.yaxis.set_ticks_position('none')
		# Set common labels
		#ax.set_xlabel('accuracy(%)')
# 		ax.yaxis.set_label_coords(-0.105, 0.5)
# 		ax.set_ylabel('RMSE')
# 		ax.yaxis.set_label_coords(-0.07, 0.5)

		return fig


	# Draw correlations.
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec

	fig = set_figure(1)
	gs = gridspec.GridSpec(1, 1)
	gs.update(hspace=0.5)


# 	plt.figure(figsize=(10, 7), dpi=80)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.set_title('Tg')
	ax1.set_xlabel('Tg ' + y1_name)
	ax1.set_ylabel('Tg ' + y2_name)
	ax1.plot(y1, y2, '+')

	fn_pre = '../figures/correlations.' + y1_name + '_' + y2_name
	plt.savefig(fn_pre + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fn_pre + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fn_pre + '.eps', format='eps', dpi=300, transparent=True, bbox_inches='tight')
	plt.show()


	# Compute R2.
	from sklearn.metrics import r2_score
	r2 = r2_score(y1, y2)
	print('R2(' + y1_name + ', ' + y2_name  + '): ' + str(r2))


if __name__ == '__main__':
	compare_ds()