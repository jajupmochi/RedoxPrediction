#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:36:54 2022

@author: ljia
"""

def plot_perfs_vs_epochs(all_scores, fig_name,
						 legends=None,
						 y_label='MAE',
						 y_label_loss='PMAE',
						 epoch_interval=1):
	import matplotlib.pyplot as plt
	import seaborn as sns
	colors = sns.color_palette('husl')[0:]
# 		sns.axes_style('darkgrid')
	sns.set_theme()
	fig = plt.figure()
	ax = fig.add_subplot(111)    # The big subplot for common labels
# 	ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

	for idx, (key, val) in enumerate(all_scores.items()):
		epochs = list(range(epoch_interval, (len(val) + 1) * epoch_interval,
					  epoch_interval))
# 		if 'loss' == key:
# 			ax2.plot(epochs, val, '-', c=colors[idx], label=key)
# 		else:
# 			ax.plot(epochs, val, '-', c=colors[idx], label=key)
		ax.plot(epochs, val, '-', c=colors[idx], label=key)


	ax.set_xlabel('epochs')
	ax.set_ylabel(y_label)
	ax.set_title('')
# 	if 'MAPE' in y_label:
# 		ax.set_yscale('log')
# 	ax2.set_ylabel('loss (PMAE)')
	fig.subplots_adjust(bottom=0.3)
	handles, labels = ax.get_legend_handles_labels()
	if 'R_squared' in labels[0]:
		ax.set_ylim(bottom=0, top=1)
# 	handles2, labels2 = ax2.get_legend_handles_labels()
# 	handles += handles2
# 	labels += labels2
	fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
# 	plt.savefig(fig_name + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fig_name + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fig_name + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()