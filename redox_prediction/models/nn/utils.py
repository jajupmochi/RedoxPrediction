"""
utils



@Author: linlin
@Date: 25.05.23
"""
import torch


def nn_to_dict(
		nn: torch.nn.Module,
		with_state_dict: bool = False,
) -> dict:
	"""Convert a neural network to a dictionary.

	Parameters
	----------
	nn : torch.nn.Module
		The neural network to convert.

	with_state_dict : bool, optional (default=False)
		Whether to include the state dictionary of the neural network.

	Returns
	-------
	dict
		The dictionary of the neural network.
	"""
	dict_nn = {
		'model_type': str(type(nn)),
		'model_architecture': str(nn),
	}
	if with_state_dict:
		dict_nn['model_state_dict'] = nn.state_dict()
	return dict_nn


def format_targets(
		targets: torch.Tensor,
		criterion: str = 'NLLLoss',
) -> torch.Tensor:
	"""Format the targets for the loss function.

	Parameters
	----------
	targets : torch.Tensor
		The targets.

	criterion : str, optional (default='NLLLoss')
		The loss function criterion.

	Returns
	-------
	torch.Tensor
		The formatted targets.
	"""
	if criterion == 'NLLLoss':
		targets = torch.flatten(targets.long())
	elif criterion == 'BCELoss':
		targets = torch.unsqueeze(targets, dim=1)
	elif criterion == 'MSELoss':
		targets = torch.unsqueeze(targets, dim=1)
	else:
		raise ValueError('Unknown criterion: {}.'.format(criterion))
	return targets
