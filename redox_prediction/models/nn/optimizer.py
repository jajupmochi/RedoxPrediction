"""
optimizer



@Author: linlin
@Date: 03.08.23
"""
from typing import Iterator, Union

import torch
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, \
	CosineAnnealingLR


def get_optimizer(
		params: Iterator[Parameter],
		optimizer: str,
		learning_rate: float,
		weight_decay: float,
		**kwargs
) -> torch.optim.Optimizer:
	"""
	Returns an optimizer.

	Parameters
	----------
	optimizer: str
		The optimizer to use.
	params: list of dict
		The parameters to optimize.
	learning_rate: float
		The evaluation rate.
	weight_decay: float
		The weight decay.
	kwargs
		Additional keyword arguments.

	Returns
	-------
	torch.optim.Optimizer
		The optimizer.
	"""
	if optimizer == 'adam':
		return torch.optim.Adam(
			params,
			lr=learning_rate,
			weight_decay=weight_decay,
		)
	elif optimizer == 'sgd':
		return torch.optim.SGD(
			params,
			lr=learning_rate,
			weight_decay=weight_decay,
			**kwargs
		)
	elif optimizer == 'adagrad':
		return torch.optim.Adagrad(
			params,
			lr=learning_rate,
			weight_decay=weight_decay,
			**kwargs
		)
	elif optimizer == 'adadelta':
		return torch.optim.Adadelta(
			params,
			lr=learning_rate,
			weight_decay=weight_decay,
			**kwargs
		)
	elif optimizer == 'rmsprop':
		return torch.optim.RMSprop(
			params,
			lr=learning_rate,
			weight_decay=weight_decay,
			**kwargs
		)
	else:
		raise ValueError(
			'Unknown optimizer: {}'.format(optimizer)
		)


# def adjust_learning_rate(
# 		optimizer: torch.optim.Optimizer,
# 		epoch: int,
# 		learning_rate: float,
# 		learning_rate_decay: float,
# 		learning_rate_decay_steps: int
# ) -> None:
# 	"""
# 	Adjusts the evaluation rate of an optimizer.
#
# 	Parameters
# 	----------
# 	optimizer: torch.optim.Optimizer
# 		The optimizer.
# 	epoch: int
# 		The current epoch.
# 	learning_rate: float
# 		The evaluation rate.
# 	learning_rate_decay: float
# 		The evaluation rate decay.
# 	learning_rate_decay_steps: int
# 		The number of epochs before the evaluation rate is decayed.
# 	"""
# 	if learning_rate_decay_steps > 0:
# 		learning_rate = learning_rate * (
# 				learning_rate_decay ** (epoch // learning_rate_decay_steps)
# 		)
# 		for param_group in optimizer.param_groups:
# 			param_group['lr'] = learning_rate


def get_learning_rate_scheduler(
		optimizer: torch.optim.Optimizer,
		learning_rate_scheduler: str = 'step',
		learning_rate_decay: float = 0.5,
		learning_rate_decay_steps: int = 50,
		**kwargs
) -> Union[StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR]:
	"""
	Returns a evaluation rate scheduler.

	Parameters
	----------
	optimizer: torch.optim.Optimizer
		The optimizer.
	learning_rate_scheduler: str
		The evaluation rate scheduler to use.
	learning_rate_decay: float
		The evaluation rate decay.
	learning_rate_decay_steps: int
		The number of epochs before the evaluation rate is decayed.
	kwargs
		Additional keyword arguments.

	Returns
	-------
	Union[StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR]
		The evaluation rate scheduler.
	"""
	if learning_rate_scheduler == 'step':
		return torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=learning_rate_decay_steps,
			gamma=learning_rate_decay,
			**kwargs
		)
	elif learning_rate_scheduler == 'exp':
		return torch.optim.lr_scheduler.ExponentialLR(
			optimizer,
			gamma=learning_rate_decay,
			**kwargs
		)
	elif learning_rate_scheduler == 'plateau':
		return torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			factor=learning_rate_decay,
			patience=learning_rate_decay_steps,
			**kwargs
		)
	elif learning_rate_scheduler == 'cosine':
		return torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=learning_rate_decay_steps,
			eta_min=0,
			**kwargs
		)
	else:
		raise ValueError(
			'Unknown evaluation rate scheduler: {}'.format(
				learning_rate_scheduler
			)
)