#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:48:59 2022

@author: ljia
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, '../')


# Get data.
def get_data(ds_name, edit_cost='random'):
	"""
	:return: data, labels, n_samples, n_features
	"""
	# Compute data as distance metrix.
	from dataset.load_dataset import load_dataset
	if ds_name == 'poly200+sugarmono':
		ds_poly = load_dataset('polyacrylates200', format_='networkx',
						 with_names=True)
		ds_sugar = load_dataset('sugarmono', format_='networkx')
		for i, n in enumerate(ds_poly['names']):
			ds_poly['names'][i] = 'poly200'

		mol_list = ds_poly['X'] + ds_sugar['X']
		y = ds_poly['targets'] + ds_sugar['targets']
		labels = ds_poly['names'] + ds_sugar['families']

	elif ds_name == 'sugarmono':
		ds_sugar = load_dataset('sugarmono', format_='networkx')
		mol_list = ds_sugar['X']
		y = ds_sugar['targets']
		labels = ds_sugar['families']

	kwargs = {'ds_name': ds_name,
		   'node_labels': list(list(mol_list[0].nodes(data=True))[0][1].keys()),
		   'edge_labels': list(list(mol_list[0].edges(data=True))[0][2].keys())
		   }
	if edit_cost == 'random':
		from models.ged import compute_D_random
		data, _ = compute_D_random(mol_list, **kwargs)
# 	elif edit_cost == 'expert':
# 		from models.ged import compute_D_expert
# 		data, _ = compute_D_expert(mol_list, **kwargs)
	elif edit_cost == 'fitted':
		from models.ged import compute_D_fitted
		data, _ = compute_D_fitted(mol_list, y, **kwargs)
	else:
		raise ValueError('edit_cost can not be recognized.')

	n_samples, n_features = len(labels), 1 # shape
	return data, labels, n_samples, n_features


# Plot.
def plot_embedding(data, label, title):
	"""
	:param data: data
	:param label: labels
	:param title: title
	:return: figure.
	"""
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)	 # Normalization
	fig = plt.figure()	  # Create figure.
	ax = plt.subplot(111)

	# for all samples.
	label_list = list(set(label))
	print(label_list)
# 	handles = [0] * len(label_list)
	for i in range(data.shape[0]):
		# Plot label.
		idx_label = label_list.index(label[i])
		h = plt.text(data[i, 0], data[i, 1], str(idx_label),
			   color=plt.cm.Set1(idx_label / len(label_list)),
			   fontdict={'weight': 'bold', 'size': 10})
# 		handles[idx_label] = h

	# Set legend.
	from matplotlib.lines import Line2D
	legend_elements = []
	for idx, label in enumerate(label_list):
		legend_elements.append(Line2D([0], [0], marker='o',
								color='w',
								label=(str(idx) + ': ' + label),
								markerfacecolor=plt.cm.Set1(idx / len(label_list)),
								markersize=5))
	size = fig.get_size_inches()
	fig.subplots_adjust(bottom=0.65 / size[1])
	fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=False)
	plt.xticks()
	plt.yticks()
	plt.title(title, fontsize=14)

	return fig


# main function.
def main(ds_name, edit_cost='expert'):
	# Get data.
	data, label, n_samples, n_features = get_data(ds_name, edit_cost)
	print('Starting compute t-SNE Embedding...')
	ts = TSNE(n_components=2, metric='precomputed') # , init='pca', random_state=0)
	# t-SNE.
	result = ts.fit_transform(data)
	# Plot.
	fig = plot_embedding(result, label, 't-SNE Embedding using GEDs (' + edit_cost + ' costs)')
	# Save and show fig.
	nm = '../figures/t-sne_geds.' + ds_name + '.' + edit_cost
	plt.savefig(nm + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(nm + '.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight')
	plt.savefig(nm + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()


if __name__ == '__main__':
	for edit_cost in ['random', 'fitted'][0:]:
		main('sugarmono', edit_cost)
# 		main('poly200+sugarmono', edit_cost)
