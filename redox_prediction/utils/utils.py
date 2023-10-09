import numpy as np


def vec2sym_mat(v):
	"""
	Convert a vector encoding a symmetric matrix into a matrix
	See Golub and Van Loan, Matrix Computations, 3rd edition, p21
	"""
	n = int((-1 + np.sqrt(1 + 8 * len(v))) / 2)  # second order resolution
	M = np.zeros((n, n))
	for i in range(n):
		for j in range(i, n):
			# Golub van Loan, Matrix Computations, Eq. 1.2.2, p21
			M[i, j] = M[j, i] = v[i * n - (i + 1) * (i) // 2 + j]
	return M


def model_type_from_dataset(ds_name):
	if ds_name in [
		'Alkane_unlabeled', 'Acyclic',  # Regression
		'QM7', 'QM9',  # Regression: big
		'brem_togn_dGred', 'brem_togn_dGox',  # Regression: Redox
	]:
		return 'reg'
	elif ds_name in [
		'MAO', 'PAH', 'MUTAG', 'Monoterpens',  # Jia thesis: mols
		'Letter-high', 'Letter-med', 'Letter-low',  # Jia thesis: letters
		'ENZYMES', 'AIDS', 'NCI1', 'NCI109', 'DD',  # Jia thesis: bigger
		'Mutagenicity', 'IMDB-BINARY', 'COX2', 'PTC_MR',  # Fuchs2022 PR paper
		'PTC_FR',  # Navarin2018 pre-training gnn with kernel paper
	]:
		return 'classif'
	else:
		raise ValueError(f'Unknown dataset {ds_name}.')


def remove_useless_keys(task):
	return task


# task_new = task.copy()
# if task_new['deep_model'] == 'none':
# 	del task_new['metric_target']
# 	del task_new['tricks']
# 	# if task_new['infer']
# 	del task_new['infer']
# return task_new


def format_bytes(num):
	for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
		if abs(num) < 1024.0:
			return f"{num:.2f} {unit}"
		num /= 1024.0
	return f"{num:.2f} {unit}"
