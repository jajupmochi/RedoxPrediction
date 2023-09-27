"""
utils



@Author: linlin
@Date: 04.08.23
"""
import torch
import torch_geometric


def transform_dataset_task(
		dataset: torch_geometric.data.Dataset,
		infer_mode: str,
		**kwargs
):
	"""
	Transforms a dataset to a task dataset.

	Parameters
	----------
	dataset : torch.utils.data.Dataset
		The dataset to transform.
	infer_mode : str
		The model_type of the dataset.
	kwargs
		Additional arguments.

	Returns
	-------
	torch.utils.data.Dataset
		The transformed dataset.
	"""
	if infer_mode == 'pretrain+refine':
		return torch_geometric.loader.DataLoader(
			dataset,
			batch_size=kwargs['batch_size'],
			shuffle=False
		)
	else:
		raise ValueError(f"Unknown infer model_type: {infer_mode}.")
