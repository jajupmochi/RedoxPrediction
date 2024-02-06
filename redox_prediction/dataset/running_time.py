"""
running_time



@Author: linlin
@Date: 27.10.23
"""


def get_ref_time_per_datapoint(descriptor, model, target):
	import json
	import os
	import numpy as np
	from sklearn.metrics import r2_score
	from sklearn.linear_model import LinearRegression

	# Get the reference time per datapoint:
	fn = '../outputs/UBELIX/results.brem_togn_{0}.{1}.{2}.none.std.json'.format(
		target, descriptor, model
	)
	if not os.path.exists(fn):
		return None

	with open(fn, 'r') as f:
		data = json.load(f)
	final_perf = data['final_perf']
	if 'history_test' in final_perf:
		if 'batch_time_pred' in final_perf['history_test']:
			ref_time = final_perf['history_test']['batch_time_pred']
		elif 'pred_time_total' in final_perf['history_test']:
			ref_time = final_perf['history_test']['pred_time_total']
		else:
			ref_time = None

		if ref_time is not None:
			# Separate the ref_time to mean and std by '\pm':
			ref_time = ref_time.split('$\\pm$')
			# ref_time = [float(ref_time[0]), float(ref_time[1])]
			ref_time = float(ref_time[0])
	else:
		ref_time = None
	return ref_time


def gather_all_ref_time_per_datapoint(target):
	all_ref_time = {}
	for descriptor in ['atom_bond_types', '1hot', 'af1hot+3d-dis']:
		ref_time_desc = {}
		for model in [
			# "traditional" models:
			# 'vc:lr_c',  # 2-3
			# 'vc:krr_c',  # 6-7
			# 'vc:svr_c',  # 8-9
			# 'vc:gpr_s', 'vc:gpr_c',  # 4-5
			# 'vc:rf_c',  # 10-11
			# 'vc:xgb_c',  # 12-13
			# 'vc:knn_c',  # 14-15
			# graph kernels, 20-24:
			'gk:sp', 'gk:structural_sp', 'gk:path', 'gk:treelet',
			'gk:wlsubtree',
			# GEDs:
			# 'ged:bp_random',
			'ged:bp_fitted',  # 16-17
			# 'ged:IPFP_random', 'ged:IPFP_fitted',  # 18-19
			# GNNs: 25-
			'nn:mpnn',
			'nn:gcn',
			'nn:dgcnn',
			'nn:gin',
			'nn:gat',
			# 'nn:graphsage',
			# 'nn:egnn', 'nn:schnet', 'nn:diffpool', 'nn:transformer',
			'nn:unimp',
		]:
			ref_time = get_ref_time_per_datapoint(descriptor, model, target)
			ref_time_desc[model] = ref_time
		all_ref_time[descriptor] = ref_time_desc
	return all_ref_time


def plot_heatmap(
		ref_times_red, ref_times_ox,
		output_file='../outputs/UBELIX/heatmap_ref_time.pdf',
):
	"""Plot heatmap of reference time per datapoint, with the corresponding
	values in the heatmap cells.

	Plot a heatmap with two parts, the upper part is for red, and the lower part
	is for ox, the two parts are separated by a solid line. For each part,
	plot the reference time per datapoint on the following axes:
	- x-axis: model
	- y-axis: descriptor
	- color: reference time
	If a value is None, then set the corresponding cell color to gray, and put
	'-' in the cell.
	Use seaborn.heatmap and its theme to beautify the plots. Then save the
	figure to a pdf file.
	"""
	import seaborn as sns
	import matplotlib.pyplot as plt
	import numpy as np
	from matplotlib.colors import LinearSegmentedColormap

	def plot_heatmap_part(
			ref_times, ax, title,
			show_x_label=True, show_y_label=True, show_x_ticklabels=True
	):
		# Define your models and descriptors
		descriptors = list(ref_times.keys())
		models = list(ref_times[descriptors[0]].keys())

		# Convert the dictionary into a list of lists
		ref_times_list = [
			[
				ref_times[desc][model] if (
						model in ref_times[desc] and ref_times[desc][model] is not None
				) else np.nan
				for model in models
			] for desc in descriptors
		]

		# Use log scale for the reference time:
		ref_times_list = np.log10(ref_times_list)

		# Define the colors for the colormap
		colors = [(1, 0, 0), (0, 1, 0)]  # Red to Green

		# Create the custom colormap
		cmap = LinearSegmentedColormap.from_list(
			'custom_red_green', colors, N=256
		)

		# Create a heatmap
		sns.heatmap(
			ref_times_list,
			annot=True,  # Show values in cells
			fmt=".2f",  # Format for displaying values
			cmap='coolwarm',  # Choose a color map
			cbar=False,  # Show color bar
			# cbar_kws={'label': 'Reference Time Per Data (log10)'},
			ax=ax,
			# cbar_kws={'missingno': 'gray'},
		)

		model_names = [
			'SP', 'SSP', 'Path', 'Treelet', 'WLSubtree',
			'GED',
			'MPNN', 'GCN', 'DGCNN', 'GIN', 'GAT', 'UniMP'
		]
		descs_names = ['AB Types', 'One-Hot', 'Dis']

		# Set axis labels
		# ax.set_xticks([])
		if show_x_ticklabels:
			ax.set_xticklabels(model_names)
			# Rotate the labels:
			ax.tick_params(axis='x', rotation=90)
		else:
			ax.set_xticklabels([])
		ax.set_yticklabels(descs_names)
		# Show labels horizontally:
		ax.tick_params(axis='y', rotation=0)
		if show_x_label:
			ax.set_xlabel('Models')
		if show_y_label:
			ax.set_ylabel('Descriptors')

		# Remove all ticks:
		ax.tick_params(
			axis='both', which='both', bottom=False, top=False,
			labelbottom=True, left=False, right=False, labelleft=True
		)

		# Set the title for the heatmap part
		ax.set_title(title)


	# Create a figure with two subplots
	fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.2))

	# Plot heatmap for ref_times_red
	plot_heatmap_part(
		ref_times_red, axes[0], title=r'$\Delta_r G^{0}_{Red}$',
		show_x_label=False, show_y_label=False, show_x_ticklabels=False
	)

	# Plot heatmap for ref_times_ox
	plot_heatmap_part(
		ref_times_ox, axes[1], title=r'$\Delta_r G^{0}_{Ox}$',
		show_x_label=True, show_y_label=False, show_x_ticklabels=True
	)

	# # Add a solid line to separate the two heatmaps
	# plt.axhline(len(ref_times_red) - 0.5, color='black', linewidth=2)

	# Add y label for the whole figure:
	fig.text(
		0, 0.5, 'Descriptors', va='center', rotation='vertical'
	)

	# Add y label for the whole figure on the right:
	fig.text(
		0.978, 0.5, 'Reference time per data (in log10, seconds)',
		va='center', rotation='vertical'
	)

	# Extend the figure to the right to make room for the y label:
	plt.subplots_adjust(right=0.7)

	# Use Seaborn's theme for styling
	sns.set_theme()

	# Adjust layout
	plt.tight_layout()

	# Save the figure to a PDF file
	plt.savefig(output_file)
	plt.close()





def main():
	# Gather all reference time per datapoint:
	ref_times_red = gather_all_ref_time_per_datapoint('dGred')
	ref_times_ox = gather_all_ref_time_per_datapoint('dGox')
	plot_heatmap(ref_times_red, ref_times_ox)


if __name__ == '__main__':
	main()
