"""
gnn_plot



@Author: linlin
@Date: 27.05.23
"""
import os

import numpy as np


def plot_perfs_vs_epochs(
		all_scores,
		fig_name,
		legends=None,
		y_labels=['loss'],
		epoch_interval=1,
		show_fig=True,
		title='',
):
	# import matplotlib
	# matplotlib.use('Agg')  # To be compatible with Dash.
	import matplotlib.pyplot as plt
	import seaborn as sns
	colors = sns.color_palette('husl')[0:]
	# 		sns.axes_style('darkgrid')
	sns.set_theme()
	fig = plt.figure()
	ax = fig.add_subplot(111)  # The big subplot for common labels
	if 'metric' in y_labels:
		ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

	for idx, (key, val) in enumerate(all_scores.items()):
		epochs = list(
			range(
				epoch_interval, (len(val) + 1) * epoch_interval,
				epoch_interval
			)
		)
		# 		if 'loss' == key:
		# 			ax2.plot(epochs, val, '-', c=colors[idx], label=key)
		# 		else:
		# 			ax.plot(epochs, val, '-', c=colors[idx], label=key)
		if isinstance(val, list):
			mask = np.isfinite(val)
		# When val is a redox_prediction.models.nn.logging.AverageMeter object:
		else:
			val = val.history
			mask = np.isfinite(val)
		if 'loss' in key:
			cur_ax = ax
		else:
			cur_ax = ax2
		cur_ax.plot(
			np.array(epochs)[mask], np.array(val)[mask], '-', c=colors[idx],
			label=key,  # Make suer that the curves appear on top of the grid.
		)

	# settings for the first y-axis:
	ax.set_xlabel('epochs')
	ax.set_ylabel(y_labels[0])
	ax.set_title(title)
	# ax.set_axisbelow(True)
	# 	if 'MAPE' in y_label:
	# 		ax.set_yscale('log')
	handles, labels = ax.get_legend_handles_labels()
	if len(labels) > 0 and 'R_squared' in labels[0]:
		ax.set_ylim(bottom=0, top=1)

	# settings for the second y-axis:
	if 'metric' in y_labels:
		ax2.set_ylabel(y_labels[1])
		# ax2.set_axisbelow(True)
		ax2.grid(False)
		handles2, labels2 = ax2.get_legend_handles_labels()

		handles += handles2
		labels += labels2

	fig.subplots_adjust(bottom=0.2)
	fig.legend(
		handles, labels, loc='lower center', ncol=3, frameon=False
	)  # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
	# 	plt.savefig(fig_name + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight')
	# 	plt.savefig(fig_name + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
	try:
		plt.savefig(
			fig_name + '.png', format='png', dpi=300, transparent=False,
			bbox_inches='tight'
		)
		if show_fig:
			plt.show()
		plt.clf()
		plt.close()
	except Exception as e:
		# This may happen when parallelizing the code.
		# Try to save the figure again:
		print('Warning: Exception when saving the figure. Trying again...')
		try:
			plt.savefig(
				fig_name + '.png', format='png', dpi=300, transparent=False,
				bbox_inches='tight'
			)
			if show_fig:
				plt.show()
			plt.clf()
			plt.close()
		except IndexError as e:
			# todo
			print('Warning: Exception when saving the figure. Skipping...')
			print('e: ', e)
		else:
			print('Succeeded in saving the figure.')



def plot_epoch_curves(
		history, figure_name, show_fig=True, loss_only=False, title=''
):
	os.makedirs(os.path.dirname(figure_name), exist_ok=True)

	# 1.
	# 	all_scores = {'loss': history['loss'],
	# 			   'train': history['mae'],
	# 			   'valid': history['val_mae'],
	# 			   'test': history['test_mae']}
	# 2.
	# 	all_scores = {}
	# 	for key, val in history.items():
	# 		if 'loss' in key or 'mae' in key:
	# 			all_scores[key] = val

	# 	plot_perfs_vs_epochs(all_scores, figure_name,
	# 						 y_label='MAE',
	# 						 y_label_loss='MAPE',
	# 						 epoch_interval=1)
	# 3.
	if loss_only:
		y_labels = ['loss']  # 'MAE', '$R^2$',
		metric_list = ['loss']
	else:
		y_labels = ['loss', 'metric']  # 'MAE', '$R^2$',
		metric_list = ['loss', 'metric']  # 'mae', 'R_squared',
	all_scores = {}
	for i_m, metric in enumerate(metric_list):
		for key, val in history.items():
			if metric in key:
				all_scores[key] = val
	# Plot all metrics in one figure:
	plot_perfs_vs_epochs(
		all_scores,
		figure_name,
		y_labels=y_labels,
		epoch_interval=1,
		show_fig=show_fig,
		title=title
	)


# all_scores = {}  # {'loss': history['loss']}
# for key, val in history.items():
# 	if metric in key:
# 		all_scores[key] = val
# plot_perfs_vs_epochs(
# 	all_scores,
# 	figure_name + '.' + metric,
# 	y_label=y_labels[i_m],
# 	y_label_loss='MAPE',
# 	epoch_interval=1,
# 	show_fig=show_fig,
# 	title=title
# )


def plot_training_loss(
		history: dict,
		**kwargs
):
	import datetime
	current_datetime = datetime.datetime.now()
	formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S%f")[:-3]
	name_kw = '{}.{}.{}.{}.{}.{}.s{}.p{}.f{}.{}'.format(
		kwargs.get('ds_name'),
		kwargs.get('deep_model'),
		kwargs.get('embedding'),
		kwargs.get('infer'),
		kwargs.get('loss'),
		kwargs.get('metric_target'),
		kwargs.get('split_idx'),
		kwargs.get('params_idx'),
		kwargs.get('fold_idx'),
		kwargs.get('fit_phase'),
	)

	figure_name = '../figures/redox_prediction/' + kwargs.get('ds_name') + '.' + kwargs.get(
		'model_name'
	) + '/' + 'perfs_vs_epochs.' + name_kw + '.' + formatted_datetime
	os.makedirs(os.path.dirname(figure_name), exist_ok=True)
	title = name_kw
	plot_epoch_curves(
		history, figure_name, show_fig=False, loss_only=False, title=None
	)


if __name__ == '__main__':
	pass
	# import numpy as np
	# print(
	# 	np.mean(
	# 		[
	# 			0.9473684210526315,
	# 			0.7368421052631579,
	# 			0.7368421052631579,
	# 			0.8947368421052632,
	# 			0.8421052631578947,
	# 			0.7368421052631579,
	# 			0.8421052631578947,
	# 			0.8947368421052632,
	# 			0.7777777777777778,
	# 			1.0
	# 		]
	# 	)
	# )
	# print(
	# 	np.std(
	# 		[
	# 			0.9473684210526315,
	# 			0.7368421052631579,
	# 			0.7368421052631579,
	# 			0.8947368421052632,
	# 			0.8421052631578947,
	# 			0.7368421052631579,
	# 			0.8421052631578947,
	# 			0.8947368421052632,
	# 			0.7777777777777778,
	# 			1.0
	# 		]
	# 	)
	# )
	# print(
	# 	np.mean(
	# 		[
	# 	0.9166666666666667,
	# 	0.6730769230769231,
	# 	0.717948717948718,
	# 	0.8782051282051282,
	# 	0.8397435897435898,
	# 	0.7023809523809523,
	# 	0.8452380952380952,
	# 	0.8571428571428572,
	# 	0.75,
	# 	1.0
	# ]
	# 	))
	# print(
	# 	np.std(
	# 		[
	# 	0.9166666666666667,
	# 	0.6730769230769231,
	# 	0.717948717948718,
	# 	0.8782051282051282,
	# 	0.8397435897435898,
	# 	0.7023809523809523,
	# 	0.8452380952380952,
	# 	0.8571428571428572,
	# 	0.75,
	# 	1.0
	# ]
	# 	))
