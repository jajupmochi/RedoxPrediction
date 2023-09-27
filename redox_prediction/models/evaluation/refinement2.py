"""
finetuning



@Author: linlin
@Date: 02.08.23
"""
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch_geometric

from redox_prediction.models.nn.loss import get_loss_func
from redox_prediction.models.nn.early_stopping import get_early_stopping_metric
from redox_prediction.models.nn.optimizer import get_optimizer
from redox_prediction.utils.logging import logging

logger = logging.get_logger(__name__)


def refine_model(
		model: torch.nn.Module,
		train_data: torch_geometric.data.Data,
		validation_data: torch_geometric.data.Data = None,
		epochs: int = 100,
		learning_rate: float = 0.001,
		weight_decay: float = 0.001,
		optimizer: str = 'adam',
		loss_func: str = 'mse',
		device: str = 'cuda',
		early_stopping: bool = True,
		early_stopping_patience: int = 10,
		early_stopping_min_delta: float = 0.0,
		early_stopping_metric: str = 'loss',
		early_stopping_mode: str = 'min',
		verbose: bool = True,
		**kwargs
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
	"""
	Trains a model on a dataset.

	Parameters
	----------
	model: torch.nn.Module
		A model to train.
	train_data: torch_geometric.data.Data
		The training data.
	validation_data: torch_geometric.data.Data, optional
		The validation data.
	epochs: int, optional
		The number of epochs to train the model for.
	learning_rate: float, optional
		The evaluation rate.
	weight_decay: float, optional
		The weight decay.
	optimizer: str, optional
		The optimizer to use. Can be 'adam', 'sgd', or 'adagrad'.
	loss_func: str, optional
		The loss function to use. Can be 'mse', 'bce', or 'nll'.
	device: str, optional
		The device to use. Can be 'cpu' or 'cuda'.
	early_stopping: bool, optional
		Whether to use early stopping.
	early_stopping_patience: int, optional
		The number of epochs to wait before stopping training if the validation
		loss/metric does not improve.
	early_stopping_min_delta: float, optional
		The minimum change in validation loss/metric to be considered an
		improvement.
	early_stopping_metric: str, optional
		The metric to use for early stopping. Can be 'loss', 'accuracy', or
		'f1'.
	early_stopping_mode: str, optional
		The model_type to use for early stopping. Can be 'min' or 'max'.
	verbose: bool, optional
		Whether to print information about training.
	kwargs: dict, optional
		Additional keyword arguments to pass to the optimizer.

	Returns
	-------
	tuple
		A tuple containing the trained model and a dictionary containing
		information about training.
	"""
	# Move the model to the specified device
	model = model.to(device)

	# Initialize the optimizer
	optimizer = get_optimizer(
		model.parameters(),
		optimizer,
		learning_rate,
		weight_decay,
		**kwargs
	)

	# Initialize the loss function
	loss_func = get_loss_func(
		loss_func,
		device,
		**kwargs
	)

	# Initialize the early stopping metric
	early_stopping_metric = get_early_stopping_metric(
		early_stopping_metric,
		early_stopping_mode,
		**kwargs
	)

	# Initialize the early stopping counter
	early_stopping_counter = 0

	# Initialize the best validation metric
	best_validation_metric = np.inf if early_stopping_mode == 'min' else -np.inf

	# Initialize the training history
	history = {
		'training_loss': [],
		'validation_loss': [],
		'training_metric': [],
		'validation_metric': []
	}

	# Initialize the epoch iterator
	epochs_iter = range(1, epochs + 1)

	# Iterate over epochs
	for epoch in epochs_iter:
		# Initialize the epoch start time
		epoch_start_time = time.time()

		# Train the model
		train_loss, train_metric = train_epoch_refine(
			model,
			train_data,
			optimizer,
			loss_func,
			device,
			**kwargs
		)

		# Evaluate the model on the validation data
		if validation_data is not None:
			validation_loss, validation_metric = evaluate_model_refine(
				model,
				validation_data,
				loss_func,
				device,
				**kwargs
			)
		else:
			validation_loss, validation_metric = None, None

		# Update the training history
		history['training_loss'].append(train_loss)
		history['validation_loss'].append(validation_loss)
		history['training_metric'].append(train_metric)
		history['validation_metric'].append(validation_metric)

		# Update the epoch iterator
		epochs_iter.set_postfix(
			epoch=epoch,
			loss=train_loss,
			metric=train_metric,
			val_loss=validation_loss,
			val_metric=validation_metric,
			time=time.time() - epoch_start_time
		)

		# Check if early stopping should be used
		if early_stopping:
			# Check if the validation metric improved
			if early_stopping_metric(validation_metric, best_validation_metric):
				# Update the best validation metric
				best_validation_metric = validation_metric

				# Reset the early stopping counter
				early_stopping_counter = 0
			else:
				# Increment the early stopping counter
				early_stopping_counter += 1

				# Check if training should be stopped
				if early_stopping_counter >= early_stopping_patience:
					# Print a message
					logger.info('Stopping early.')

					# Stop training
					break

	# Print a message
	logger.info('Finished training.')

	# Return the trained model and the training history
	return model, history


def train_epoch_refine(
		model: torch.nn.Module,
		data: torch_geometric.data.Data,
		optimizer: torch.optim.Optimizer,
		loss_func: torch.nn.modules.loss._Loss,
		device: str = 'cuda',
		**kwargs
) -> Tuple[float, float]:
	"""
	Trains a model for one epoch.

	Parameters
	----------
	model: torch.nn.Module
		A model to train.
	data: torch_geometric.data.Data
		The training data.
	optimizer: torch.optim.Optimizer
		The optimizer to use.
	loss_func: torch.nn.modules.loss._Loss
		The loss function to use.
	device: str, optional
		The device to use. Can be 'cpu' or 'cuda'.
	kwargs: dict, optional
		Additional keyword arguments to pass to the loss function.

	Returns
	-------
	tuple
		A tuple containing the loss and metric for the epoch.
	"""
	# Set the model to training model_type
	model.train()

	# Initialize the loss and metric
	loss, metric = 0.0, 0.0

	# Iterate over the data
	for batch in data:
		# Move the batch to the specified device
		batch = batch.to(device)

		# Zero the gradients
		optimizer.zero_grad()

		# Forward pass
		pred = model(batch)

		# Compute the loss
		batch_loss = loss_func(pred, batch, **kwargs)

		# Backward pass
		batch_loss.backward()

		# Update the parameters
		optimizer.step()

		# Update the loss and metric
		loss += batch_loss.item()
		metric += utils.compute_metric(pred, batch)

	# Return the loss and metric
	return loss / len(data), metric / len(data)


def evaluate_model_refine(
		model: torch.nn.Module,
		data: torch_geometric.data.Data,
		loss_func: torch.nn.modules.loss._Loss,
		device: str = 'cuda',
		**kwargs
) -> Tuple[float, float]:
	"""
	Evaluates a model on a dataset.

	Parameters
	----------
	model: torch.nn.Module
		A model to evaluate.
	data: torch_geometric.data.Data
		The dataset to evaluate on.
	loss_func: torch.nn.modules.loss._Loss
		The loss function to use.
	device: str, optional
		The device to use. Can be 'cpu' or 'cuda'.
	kwargs: dict, optional
		Additional keyword arguments to pass to the loss function.

	Returns
	-------
	tuple
		A tuple containing the loss and metric for the dataset.
	"""
	# Set the model to evaluation model_type
	model.eval()

	# Initialize the loss and metric
	loss, metric = 0.0, 0.0

	# Iterate over the data
	for batch in data:
		# Move the batch to the specified device
		batch = batch.to(device)

		# Forward pass
		pred = model(batch)

		# Compute the loss
		batch_loss = loss_func(pred, batch, **kwargs)

		# Update the loss and metric
		loss += batch_loss.item()
		metric += utils.compute_metric(pred, batch)

	# Return the loss and metric
	return loss / len(data), metric / len(data)


def train_model_refine(
		model: torch.nn.Module,
		train_data: torch_geometric.data.Data,
		validation_data: torch_geometric.data.Data = None,
		optimizer: torch.optim.Optimizer = None,
		loss_func: torch.nn.modules.loss._Loss = None,
		epochs: int = 100,
		early_stopping: bool = False,
		early_stopping_patience: int = 10,
		early_stopping_mode: str = 'min',
		device: str = 'cuda',
		**kwargs
) -> Tuple[torch.nn.Module, dict]:
	"""
	Trains a model.

	Parameters
	----------
	model: torch.nn.Module
		A model to train.
	train_data: torch_geometric.data.Data
		The training data.
	validation_data: torch_geometric.data.Data, optional
		The validation data.
	optimizer: torch.optim.Optimizer, optional
		The optimizer to use.
	loss_func: torch.nn.modules.loss._Loss, optional
		The loss function to use.
	epochs: int, optional
		The number of epochs to train for.
	early_stopping: bool, optional
		Whether to use early stopping.
	early_stopping_patience: int, optional
		The number of epochs to wait before stopping early.
	early_stopping_mode: str, optional
		The model_type to use for early stopping. Can be 'min' or 'max'.
	device: str, optional
		The device to use. Can be 'cpu' or 'cuda'.
	kwargs: dict, optional
		Additional keyword arguments to pass to the loss function.

	Returns
	-------
	tuple
		A tuple containing the trained model and the training history.
	"""
	# Check if the optimizer is None
	if optimizer is None:
		# Initialize the optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# Check if the loss function is None
	if loss_func is None:
		# Initialize the loss function
		loss_func = torch.nn.MSELoss()

	# Initialize the early stopping counter
	early_stopping_counter = 0

	# Initialize the best validation metric
	best_validation_metric = np.inf if early_stopping_mode == 'min' else -np.inf

	# Initialize the training history
	history = {
		'training_loss': [],
		'training_metric': [],
		'validation_loss': [],
		'validation_metric': []
	}

	# Iterate over the epochs
	for epoch in range(epochs):
		# Train the model for one epoch
		train_loss, train_metric = train_epoch_refine(
			model, train_data, optimizer, loss_func, device, **kwargs
		)

		# Evaluate the model on the validation data
		if validation_data is not None:
			validation_loss, validation_metric = evaluate_model_refine(
				model, validation_data, loss_func, device, **kwargs
			)

		# Print the epoch results
		print(
			f'Epoch {epoch + 1} | Training Loss: {train_loss:.4f} | Training Metric: {train_metric:.4f} | Validation Loss: {validation_loss:.4f} | Validation Metric: {validation_metric:.4f}'
		)

		# Append the epoch results to the training history
		history['training_loss'].append(train_loss)
		history['training_metric'].append(train_metric)
		history['validation_loss'].append(validation_loss)
		history['validation_metric'].append(validation_metric)

		# Check if early stopping is enabled
		if early_stopping:
			# Check if the validation metric has improved
			if (
					early_stopping_mode == 'min' and validation_metric < best_validation_metric) or (
					early_stopping_mode == 'max' and validation_metric > best_validation_metric):
				# Update the best validation metric
				best_validation_metric = validation_metric

				# Reset the early stopping counter
				early_stopping_counter = 0
			else:
				# Increment the early stopping counter
				early_stopping_counter += 1

				# Check if early stopping should be performed
				if early_stopping_counter == early_stopping_patience:
					# Print the early stopping message
					print(f'Early stopping at epoch {epoch + 1}')

					# Stop training
					break

	# Return the trained model and the training history
	return model, history
