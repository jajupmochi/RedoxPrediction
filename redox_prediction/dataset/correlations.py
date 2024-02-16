"""
correlations



@Author: linlin
@Date: 27.10.23
"""
import numpy as np


def get_y_data(fn_name):
	import json
	with open(fn_name, 'r') as f:
		data = json.load(f)
	# y_true = []
	y_pred = []
	results = data['results']
	for key, val in results.items():
		if 'y_pred_test' in val:
			# y_true += np.array(val['y_true_test']).ravel().tolist()
			y_pred += np.array(val['y_pred_test']).ravel().tolist()

	# todo: Remove this when new results are available.
	if 'dGred' in fn_name:
		fn_y_true = '../outputs/results.brem_togn_dGred.1hot.vc:lr_s.none.std.json'
	elif 'dGox' in fn_name:
		fn_y_true = '../outputs/results.brem_togn_dGox.1hot.vc:lr_s.none.std.json'
	else:
		raise ValueError('Unknown fn_name: ', fn_name)
	with open(fn_y_true, 'r') as f:
		data = json.load(f)
		y_true = []
		results = data['results']
		for key, val in results.items():
			if 'y_true_test' in val:
				y_true += np.array(val['y_true_test']).ravel().tolist()

	return y_true, y_pred


def plot_correlations(
		y_true, y_pred, y_true2, y_pred2,
		output_file='../outputs/UBELIX/best_correlations.pdf'
):
	"""Plot correlations between y_true and y_pred.

	Plot two scatter figures side by side, left for dGred, right for dGox.
	Use seaborn.scatterplot and its theme to beautify the plots. Then save the
	figure to a pdf file.
	"""
	import seaborn as sns
	import matplotlib.pyplot as plt
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import r2_score

	# # Enable LaTeX rendering
	# plt.rc('text', usetex=True)

	# Create a figure with two subplots
	fig, axes = plt.subplots(1, 2, figsize=(6, 3))

	# Plot dGred scatterplot
	sns.scatterplot(
		x=y_true, y=y_pred, ax=axes[0], alpha=0.7, s=30, label='Scatter',
		zorder=2
	)
	axes[0].set_xlabel('DFT (kcal/mol)')
	axes[0].set_ylabel('Predicted (kcal/mol)')
	# axes[0].set_title(r'$\Delta_r \text{G}^{0}_{\text{Red}}$')
	axes[0].set_title(r'$\Delta_r G^{0}_{Red}$')

	# Compute and plot the linear regression line for dGred
	model_dGred = LinearRegression()
	model_dGred.fit(np.array(y_true).reshape(-1, 1), y_pred)
	y_pred_dGred = model_dGred.predict(np.array(y_true).reshape(-1, 1))
	r2_dGred = r2_score(y_pred, y_pred_dGred)
	h1 = axes[0].plot(
		y_true, y_pred_dGred, color='gray',
		linestyle='--', linewidth=1,
		zorder=1
	)

	# Add a text of R2 score to the upper left corner of the subplot with a box:
	axes[0].text(
		0.05, 0.95, f'$R^2 = $ {r2_dGred:.3f}', transform=axes[0].transAxes,
		# fontsize=11,
		verticalalignment='top',
		bbox=dict(
			boxstyle='round', facecolor='white', alpha=0.2)
	)

	# Do not show the legend for scatterplot, show the legend for the linear
	# regression line (the handle h1) in the bottom right corner of the subplot:
	axes[0].legend(
		loc='lower right', labels=[f'Ideal fit'],
		handles=h1
	)



	# Plot dGox scatterplot with green dots
	sns.scatterplot(
		x=y_true2, y=y_pred2, ax=axes[1], color='green', alpha=0.7, s=30,
		label='Scatter',
		zorder=2
	)
	axes[1].set_xlabel('DFT (kcal/mol)')
	axes[1].set_ylabel('Predicted (kcal/mol)')
	axes[1].set_title(r'$\Delta_r G^{0}_{Ox}$')


	# Compute and plot the linear regression line for dGox
	model_dGox = LinearRegression()
	model_dGox.fit(np.array(y_true2).reshape(-1, 1), y_pred2)
	y_pred_dGox = model_dGox.predict(np.array(y_true2).reshape(-1, 1))
	r2_dGox = r2_score(y_pred2, y_pred_dGox)
	h2 = axes[1].plot(
		y_true2, y_pred_dGox, color='gray',
		linestyle='--', linewidth=1,
		zorder=1
	)

	# Add a text of R2 score to the upper left corner of the subplot with a box:
	axes[1].text(
		0.05, 0.95, f'$R^2 = $ {r2_dGox:.3f}', transform=axes[1].transAxes,
		# fontsize=11,
		verticalalignment='top',
		bbox=dict(
			boxstyle='round', facecolor='white', alpha=0.2)
	)

	# Do not show the legend for scatterplot, show the legend for the linear
	# regression line (the handle h1) in the bottom right corner of the subplot:
	axes[1].legend(
		loc='lower right', labels=[f'Ideal fit'],
		handles=h2
	)



	# Use Seaborn's theme for styling
	sns.set_theme()

	# Add legends for both subfigures
	# axes[0].legend(loc='upper left', labels=[f'R^2 Score = {r2_dGred:.2f}'])
	# axes[1].legend(loc='lower right', labels=[f'Ideal fit (R^2={r2_dGox:.2f})'])

	# Save the figure to a PDF file
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()


def main(fn_name1, fn_name2):
	# ----- dGred: -----
	y_true1, y_pred1 = get_y_data(fn_name1)
	# Compute R2 score:
	from sklearn.metrics import r2_score
	r2 = r2_score(y_true1, y_pred1)
	print('R2 score for dGred: ', r2)

	# ----- dGox: -----
	y_true2, y_pred2 = get_y_data(fn_name2)
	# Compute R2 score:
	from sklearn.metrics import r2_score
	r2 = r2_score(y_true2, y_pred2)
	print('R2 score for dGox: ', r2)

	# Plot correlations:
	plot_correlations(y_true1, y_pred1, y_true2, y_pred2)


if __name__ == '__main__':
	fn_name1 = '../outputs/UBELIX/results.brem_togn_dGred.af1hot+3d-dis.nn:mpnn.none.std.json'
	fn_name2 = '../outputs/UBELIX/results.brem_togn_dGox.1hot.nn:mpnn.none.std.json'
	# Remove '/UBELIX/' for experiments on local machine.
	main(fn_name1, fn_name2)
