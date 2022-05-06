#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:42:55 2020

@author: ljia
"""
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command


def rounder(x, decimals):
	x_strs = str(x).split('.')
	if len(x_strs) == 2:
		before = x_strs[0]
		after = x_strs[1]
		if len(after) > decimals:
			if int(after[decimals]) >= 5:
				after0s = ''
				for c in after:
					if c == '0':
						after0s += '0'
					elif c != '0':
						break
				after = after0s + str(int(after[0:decimals]) + 1)[-decimals:]
			else:
				after = after[0:decimals]
		elif len(after) < decimals:
			after += '0' * (decimals - len(after))
		return before + '.' + after

	elif len(x_strs) == 1:
		return x_strs[0]


def df_to_latex_table(df, replace_header=True, end_mid_line=7):
	ltx = df.to_latex(index=True, escape=False, multirow=True)

	# modify middle lines.
	end_mid_line = str(end_mid_line)
	ltx = ltx.replace('\\cline{1-' + end_mid_line + '}\n\\cline{2-' + end_mid_line + '}',	'\\toprule')
	ltx = ltx.replace('\\cline{2-' + end_mid_line + '}', '\\cmidrule(l){2-' + end_mid_line + '}')

	# Reset dataset name.
	ltx = ltx.replace('Alkane_unlabeled', 'Alkane')
	ltx = ltx.replace('Vitamin_D', 'Vitamin\_D')

	# modify header.
	if replace_header:
		i_start = ltx.find('\\begin{tabular}')
		i_end = ltx.find('\\\\\n\\midrule\n')
		replace = r"""\begin{tabular}{lll@{~~}c@{~~}c@{~~}c@{~~}c}
\toprule
\multirow{2}[2]{*}{\textbf{Dataset}} & \multirow{2}[2]{*}{\textbf{Distance}} & \multirow{2}[2]{*}{\textbf{Method}} & \multicolumn{2}{c}{\textbf{bipartite}} & \multicolumn{2}{c}{\textbf{IPFP}} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
&   &   &  \textbf{Train errors} & \textbf{Test errors} & \textbf{Train errors} & \textbf{Test errors} \\
\midrule
"""
		ltx = ltx.replace(ltx[i_start:i_end+12], replace, 1)
#
#	# add row numbers.
#	ltx = ltx.replace('lllllllll', 'lllllllll|@{\\makebox[2em][r]{\\textit{\\rownumber\\space}}}', 1)
#	ltx = replace_nth(ltx, '\\\\\n', '\\gdef\\rownumber{\\stepcounter{magicrownumbers}\\arabic{magicrownumbers}} \\\\\n', 1)

	return ltx


def beautify_df(df):
#	df = df.sort_values(by=['Datasets', 'Graph Kernels'])
#	df = df.set_index(['Datasets', 'Graph Kernels', 'Algorithms'])
# #	index = pd.MultiIndex.from_frame(df[['Datasets', 'Graph Kernels', 'Algorithms']])

	# bold the best results.
	for ds in df.index.get_level_values('Dataset').unique():
		for gk in df.loc[ds].index.get_level_values('Distance').unique():
			for label, col in df.loc[(ds, gk)].items():
				min_val = np.inf
				min_indices = []
				min_labels = []
				for index, row in col.items():
					value = row
					if value != '-':
						mean, interval = value.split('$\\pm$')
						mean = float(mean.strip('/same'))
						if mean < min_val:
							min_val = mean
							min_indices = [index]
							min_labels = [label]
						elif mean == min_val:
							min_indices.append(index)
							min_labels.append(label)
				for idx, index in enumerate(min_indices):
					df.loc[(ds, gk, index), min_labels[idx]] = '\\textbf{' + df.loc[(ds, gk, index), min_labels[idx]] + '}'

	# Rename indices.
	df.index.set_levels([r'Euclidean', r'Manhattan'], level=1, inplace=True)

	return df


def params_to_latex_table(results):
	import pandas as pd

	# Create df table.
	row_indices = pd.MultiIndex.from_product([Target_List, Edit_Cost_List, Dis_List], names=['Dataset', 'Edit cost', 'Distance'])
	df = pd.DataFrame(columns=['$c_{ni}$', '$c_{nr}$', '$c_{ns}$', '$c_{ei}$', '$c_{er}$', '$c_{es}$'], index=row_indices)

	# Set data.
	for idx_r, row in df.iterrows():
		for idx, (idx_c, col) in enumerate(row.items()):
			key = (idx_r[0], idx_r[2], idx_r[1])
			if key in results and results[key] is not None:
# 				if results[key][idx] != 0:
				df.loc[idx_r, idx_c] = results[key][idx]
# 				else:
# 					df.loc[idx_r, idx_c] = '-'
			else:
				df.loc[idx_r, idx_c] = '-'

# 	df = beautify_df(df)
	# Rename indices.
# 	df.index.set_levels([r'\texttt{bipartite}', r'\texttt{IPFP}'], level=1, inplace=True)
	df.index.set_levels([r'bipartite', r'IPFP'], level=1, inplace=True)
	df.index.set_levels([r'Euclidean', r'Manhattan'], level=2, inplace=True)

	ltx = df_to_latex_table(df, replace_header=False, end_mid_line=9)
	return ltx


def results_to_latex_table(results):
	import pandas as pd

	# Create df table.
	col_indices = pd.MultiIndex.from_product([Edit_Cost_List, ['Train errors', 'Test errors']])
	row_indices = pd.MultiIndex.from_product([Target_List, Dis_List, ['random', 'expert', 'fitted']], names=['Dataset', 'Distance', 'Method'])
	df = pd.DataFrame(columns=col_indices, index=row_indices)

	# Set data.
	for idx_r, row in df.iterrows():
		for idx_c, col in row.items():
			key = (idx_r[0], idx_r[1], idx_c[0])
			if key in results and results[key] is not None:
				mean = results[key][idx_r[2]]['mean']
				mean = mean[0] if idx_c[1] == 'Train errors' else mean[1]
				interval = results[key][idx_r[2]]['interval']
				interval = interval[0] if idx_c[1] == 'Train errors' else interval[1]
				df.loc[idx_r, idx_c] = rounder(mean, 2) + '$\pm$' + rounder(interval, 2)
			else:
				df.loc[idx_r, idx_c] = '-'

	df = beautify_df(df)
	ltx = df_to_latex_table(df)
	return ltx


def get_params(results):
	edit_costs = [[] for i in range(6)]
	for result in results['results']:
		ed = result['fitted']['edit_costs']
		for i, e in enumerate(ed):
			edit_costs[i].append(e)

	for i, ed in enumerate(edit_costs):
		mean, interval = mean_confidence_interval(ed)
		if mean == 0:
			edit_costs[i] = '-'
		else:
			edit_costs[i] = rounder(mean, 2) + '$\pm$' + rounder(interval, 2)

	return edit_costs


def print_curves(ax, p_all, title, y_label='RMSE', export_filename=None,
				 show_interval=False, renderer='seaborn', smooth=None):

	### Compute average scores.
	scores_ave = []
	for epoch in range(0, len(p_all[0])):
		ave = np.mean([i[epoch] for i in p_all])
		scores_ave.append(ave)


	### Use Seaborn style.
	if renderer == 'seaborn':
		import seaborn as sns
		colors = sns.color_palette('husl')[0:]
# 		sns.axes_style('darkgrid')
		sns.set_theme()


# 	### Smooth the data by average values.
# 	if smooth is not None:
# 		def fun_smooth(x):
# 			x_smt = np.empty((len(x),))
# 			for i in range(0, len(x)):
# 				x_smt[i] = np.mean(x[i - int(smooth / 2):i + int(smooth / 2) + 1])
# 			return x_smt


	### Plot figures.
	# if show confidence interval.
	if show_interval:
		for idx, (xp, y) in enumerate(y_err.items()):
			# Draw curves.
			if smooth is None:
				plt.plot(epochs, y, '.-', c=colors[idx])
				plt.fill_between(epochs,
						np.array(y) - np.array(y_int[xp]),
						np.array(y) + np.array(y_int[xp]),
						alpha=0.3, facecolor=colors[idx])
				txt_margin_ratio = 10
				max_y = np.max(y)
				smooth = 9
			else:
				y_smt, y_int_smt = fun_smooth(y), fun_smooth(y_int[xp])
				plt.plot(epochs, y_smt, '-', c=colors[idx])
				plt.fill_between(epochs,
						y_smt - y_int_smt,
						y_smt + y_int_smt,
						alpha=0.3, facecolor=colors[idx])
				txt_margin_ratio = 30
				max_y = np.max(y)
				y, y_int = y_smt, y_int_smt

			# Draw text.
			for index, value in enumerate(epochs):
				if value % 50 == 0:
					plt.text(value - epochs[-1] / 50,
					  (y[index] + max_y / txt_margin_ratio if y[index] > 0 else 0),
		 			'{:.1f}'.format(y[index]),
					 color=colors[idx], #palette(i),
					 size=6)
					plt.text(value - epochs[-1] / 50,
					  (y[index] - max_y / txt_margin_ratio if y[index] > 0 else 0),
		 			'{:.1f}'.format(np.mean(y[index-int(smooth / 2):index+int(smooth / 2)+1])),
					 color=colors[idx+1], #palette(i),
					 size=6)
	# if not show confidence interval.
	else:
		# Plot each trial.
		for idx, scores in enumerate(p_all):
			epochs = range(0, len(scores))
			ax.plot(epochs, scores, '-', c=colors[idx], label=('trial ' + str(idx)))

		# Plot the average score.
		ax.plot(epochs, scores_ave, '-', c=colors[idx + 1], label='average score')

# 		for xp, y in y_err.items():

# 			plt.plot(epochs, y, '.-')
# 			# Draw text.
# 			max_y = np.max(y)
# 			for index, value in enumerate(epochs):
# 				if value % 50 == 0:
# 		 			plt.text(value - epochs[-1] / 50, (y[index] + max_y / 50 if y[index] > 0 else 0),
# 		 			'{:.1f}'.format(y[index]),
# 					 color='Green', #palette(i),
# 					 size=6)

	### Set general layout.
# 	ax.set_xticks(r)
# 	ax.set_xticklabels(['train', 'test'] ) # ['train errors', 'test errors'])
	if renderer == 'matplotlib':
		ax.xaxis.set_ticks_position('none')
	ax.set_ylabel(y_label)
	#	ax.legend()
	ax.set_title(title)
	y_lower = 15
	y_upper = 60 # 50 # @todo: to change back
	ax.set_ylim([y_lower, y_upper])


	if (export_filename is not None):
		print(export_filename)
		plt.savefig(export_filename)


def print_table_results(results_by_xp):
	from tabulate import tabulate
	tab = []
	tab.append(["Method", "App","Test"])
	#setups = ["random","expert","fitted"]

	for i,setup in enumerate(results_by_xp.keys()):
		current_line = [setup]
		p = results_by_xp[setup]
		current_line.append(f"{p['mean'][0]:.2f} +- {p['interval'][0]:.2f}")
		current_line.append(f"{p['mean'][1]:.2f} +- {p['interval'][1]:.2f}")
		tab.append(current_line)

	print(tabulate(tab, headers="firstrow"))


def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
	return m, h


def compute_perf(results, app_or_test):
	return mean_confidence_interval(results[app_or_test])


def compute_displayable_results(results_by_xp):
	p = {}
	for xp in results_by_xp.keys():
		p[xp] = {}
		p[xp]["mean"] = [0] * 2
		p[xp]["interval"] = [0] * 2
		p[xp]["mean"][0], p[xp]["interval"][0] = compute_perf(results_by_xp[xp], 'app')
		p[xp]["mean"][1], p[xp]["interval"][1] = compute_perf(results_by_xp[xp], 'test')
	return p


def organize_results_by_xps(results, xps):
# 	all_results = results["results"]

	results_by_xp = {}
	for xp in xps:
		results_xp = {
			'app' :[],
			'test' : []
		}

		all_results = results[xp]['results']

		for i, split_res in enumerate(all_results):
			results_xp['app'].append(split_res['perf_app'])
			results_xp['test'].append(split_res['perf_test'])
		results_by_xp[xp] = results_xp
	return results_by_xp


def plot_a_task(ax, model_name, graph_pool, metric_name, title, y_label):
	p_all = {}

	DS_Name_List = ['poly200', 'thermo_exp', 'thermo_cal'][0:1]
	Feature_Scaling_List = ['standard_y', 'minmax_y', 'none'][0:1]
	feature_scaling = Feature_Scaling_List[0]

	# Load data.
	root_dir = '../outputs/gnn/'
# 	str_stra = ('.stratified' if stratified else '')
	results = []
	for ds_name in DS_Name_List:
		for trial in range(1, 6):
			task = {'ds_name': ds_name, 'feature_scaling': feature_scaling,
			  'metric': metric_name, 'model': model_name}
			fn = root_dir + 'results.' + '.'.join([v for k, v in task.items()]) + '.shuffle.5folds.no_seed/all_scores.trial' + str(trial) + '.pkl'
			if os.path.isfile(fn):
				with open(fn, 'rb') as file:
					result = pickle.load(file)
					results.append(result[score_key])
	# 			else:
	# 				results[ds_name] = {'results': [{'perf_app': np.nan, 'perf_test': np.nan}]}

	if results == []:
		return None, None

	# Compute data.
# 	xps = ["random", "expert", "fitted"]
# 	results_by_xp = organize_results_by_xps(results, DS_Name_List)
# 	p = compute_displayable_results(results_by_xp)
#		 print_bars(p,'KNN with CV and y_distance = {0}'.format(results['y_distance']),export_filename=export_filename)
	p_all = results

	print_curves(ax, p_all, title, y_label=y_label, export_filename=None)
	p, c = None, None # get_params(results)
	return p, c


def set_figure(nb_rows, nb_cols, y_label):
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
	fig = plt.figure(figsize=(8 * nb_cols, 2.12 * nb_rows + 2))
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
	ax.yaxis.set_label_coords(-0.105, 0.5)
	ax.set_xlabel('epochs')
	ax.set_ylabel(y_label) #, labelpad=-10)
	ax.yaxis.set_label_coords(-0.07, 0.5)

	return fig


def get_title(row_name):
# 	ed = 'bipartite' if edit_cost == 'BIPARTITE' else 'IPFP'
# # 	ed = r'\texttt{' + ed + r'}'
# 	dis = distance[0].upper() + distance[1:]
# 	return ed + ', ' + dis
	return row_name


def get_ylabel(col_name):
# 	if tg_name.startswith('dGox'):
# 		return '$\Delta G_{ox}^{PM7}$'
# 	elif tg_name.startswith('dGred'):
# 		return '$\Delta G_{red}^{PM7}$'
	return col_name


if __name__ == '__main__':
# 	from sklearn.model_selection import ParameterGrid
	import pickle

	stratified = False
	score_key = 'valid_scores'

	# Get task grid.
	Model_List = ['GraphConvModel', 'GCNModel', 'GATModel']
	Metric_List = ['RMSE', 'MAE', 'R2'][1:2]
	Graph_Pool_List = ['max', 'none'][0:1]

	# Set params of rows and cols.
#	row_grid = ParameterGrid({'edit_cost': Edit_Cost_List[0:],
#						 'distance': Dis_List[0:]})
	# show by edit costs then by distances.
	row_grid_list = Graph_Pool_List[0:]
# 	for i in Edit_Cost_List[0:]:
# 		for j in Dis_List[0:]:
# 		 row_grid_list.append({'edit_cost': i, 'distance': j})
	col_grid_list = Model_List[0:]
	fig_name_list = Metric_List[0:]

	for fig_name in fig_name_list:
		print('---------------------- %s ---------------------' % fig_name)

		# MAX_Y
		MAX_Y = 0

		# Compute and plot.
		fig = set_figure(len(col_grid_list), len(row_grid_list), fig_name)
		gs = gridspec.GridSpec(len(col_grid_list), len(row_grid_list))
		gs.update(hspace=0.3)

		results = {}
		params = {}
		for row, col_name in enumerate(col_grid_list):
			for col, row_name in enumerate(row_grid_list):
				print('---------------------- %s, %s ---------------------' % (col_name, row_name))

				ax = fig.add_subplot(gs[row, col])
				y_label = (get_ylabel(col_name) if col == 0 else '')

				title = get_title(row_name) if row == 0 else ''
				p, c = plot_a_task(ax, col_name, row_name, fig_name, title, y_label)
				results[(col_name, row_name)] = p
				params[(col_name, row_name)] = c
				if col == 0 and row == 0:
					handles, labels = ax.get_legend_handles_labels()

		# Show graphic
		size = fig.get_size_inches()
		fig.subplots_adjust(bottom=0.99 / size[1])
		fig.suptitle(score_key)
# 		labels[0] = 'poly200_exp'
		fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
		nm_stra = ('.stratified' if stratified else '')
		fn_prefix = 'accu_vs_epochs.' + fig_name + '.' + score_key + nm_stra
		plt.savefig(fn_prefix + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight')
		plt.savefig(fn_prefix + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
		plt.savefig(fn_prefix + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close()

# 		# Convert results to latex table.
# 		ltable_perf = results_to_latex_table(results)
# 		ltable_params = params_to_latex_table(params)
# 		print(ltable_perf)