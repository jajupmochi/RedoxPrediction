"""
loss



@Author: linlin
@Date: 31.07.23
"""

import torch
import torch.nn as nn


class DistanceLoss(nn.Module):
	"""
	Computes the mean squared error between the distance of two vectors and a
	target value.

	Authors
	-------
	* Linlin Jia
	* ChatGPT-4, 2023.07.31
	"""


	def __init__(self):
		super(DistanceLoss, self).__init__()
		self.mse_loss = nn.MSELoss()


	def forward(self, output1, output2, target):
		diff = output1 - output2
		diff_norm = diff.norm(dim=1)  # Computes the norm of the vectors
		# Computes the mean squared error:
		loss = self.mse_loss(diff_norm, target)
		return loss


class WeightedSumLoss(nn.Module):
	"""
	Computes the combined loss between two losses by the following formula:
	loss = alpha * loss1 + (1 - alpha) * loss2, where
	loss1 = func1(output1, target1), and
	loss2 = func2(output2, target2).

	Authors
	-------
	* Linlin Jia
	* ChatGPT-4, 2023.09.21
	"""


	def __init__(
			self,
			func1: nn.Module = nn.MSELoss(),
			func2: nn.Module = nn.NLLLoss(),
			alpha: float = 0.5,
			train_alpha: bool = True,
	):
		super(WeightedSumLoss, self).__init__()
		self.func1 = func1
		self.func2 = func2
		if train_alpha:
			self._alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
		else:
			self._alpha = alpha

	def forward(self, output1, target1, output2, target2):
		loss1 = self.func1(output1, target1)
		loss2 = self.func2(output2, target2)
		loss = self._alpha * loss1 + (1 - self._alpha) * loss2
		return loss, loss1, loss2


	@property
	def alpha(self):
		if isinstance(self._alpha, nn.Parameter):
			return self._alpha.detach().item()
		else:
			return self._alpha


class MetricTargetLoss(nn.Module):
	"""
	Computes the combined loss by the following formula:
	loss = alpha * loss1 + (1 - alpha) * (loss2 + loss3), where
	loss1 = func1(||embedding1 - embedding2||, metric_score),
	loss2 = func2(output1, target1), and
	loss3 = func3(output2, target2).

	Authors
	-------
	* Linlin Jia
	* ChatGPT-4, 2023.09.21
	"""


	def __init__(
			self,
			func1: nn.Module = nn.MSELoss(),
			func2: nn.Module = nn.NLLLoss(),
			alpha: float = 0.5,
			train_alpha: bool = True,
	):
		super(MetricTargetLoss, self).__init__()
		self.func1 = func1
		self.func2 = func2
		if train_alpha:
			self._alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
		else:
			self._alpha = alpha

	def forward(
			self,
			embedding1, embedding2, metric_score,
			output1, target1,
			output2, target2,
	):
		diff_m = embedding1 - embedding2
		diff_m = diff_m.norm(dim=1)  # Computes the norm of the vectors
		loss0 = self.func1(diff_m, metric_score)
		loss1 = self.func2(output1, target1)
		loss2 = self.func2(output2, target2)
		loss = self._alpha * loss0 + (1 - self._alpha) * (loss1 + loss2)
		return loss, loss0, loss1, loss2


	@property
	def alpha(self):
		if isinstance(self._alpha, nn.Parameter):
			return self._alpha.detach().item()
		else:
			return self._alpha


# def get_loss_func(
# 		loss_func: str = 'mse',
# 		**kwargs
# ) -> nn.Module:
# 	"""
# 	Returns a loss function.
#
# 	Parameters
# 	----------
# 	loss_func: str, optional
# 		The loss function to use.
#
# 	Returns
# 	-------
# 	nn.Module
# 		The loss function.
# 	"""
# 	if loss_func == 'mse':
# 		return nn.MSELoss()
# 	elif loss_func == 'mae':
# 		return nn.L1Loss()
# 	elif loss_func == 'distance':
# 		return DistanceLoss()
# 	else:
# 		raise ValueError('Unknown loss function: {}'.format(loss_func))


def get_loss_func(
		return_embeddings: bool = False,
		loss_name: str = None,
		model_type: str = None,
		infer: str = None,
		predictor_clf_activation: str = 'None',
		**kwargs
):
	if return_embeddings:
		loss = DistanceLoss()
		return loss

	if loss_name is not None:
		if loss_name == 'mse':
			loss = nn.MSELoss()
		elif loss_name == 'mae':
			loss = nn.L1Loss()
		elif loss_name == 'distance':
			loss = DistanceLoss()
		elif loss_name == 'weighted_sum':
			loss = WeightedSumLoss()
		elif loss_name == 'metric_target':
			loss = MetricTargetLoss()
		else:
			raise ValueError('Unknown loss function: {}'.format(loss_name))
		return loss

	if model_type == 'reg':
		loss = torch.nn.MSELoss()
	elif model_type == 'classif':
		if predictor_clf_activation == 'sigmoid':
			loss = torch.nn.BCELoss()
		elif predictor_clf_activation == 'log_softmax':
			loss = torch.nn.NLLLoss()
		else:
			raise ValueError('"predictor_clf_activation" must be either "sigmoid" or "softmax".')
	# loss = torch.nn.CrossEntropyLoss() # @ todo
	# loss = torch.nn.BCEWithLogitsLoss()  # @ todo: see
	# https://discuss.pytorch.org/t/bce-loss-giving-negative-values/62309/2
	# (In general, you will prefer BCEWithLogitsLoss over BCELoss.)
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')

	if infer is not None:
		if infer == 'e2e:losses_m+t':
			loss = MetricTargetLoss(
				func1=nn.MSELoss(),
				func2=loss,
				alpha=0.5,
				train_alpha=True,
			)
		else:
			pass

	return loss
