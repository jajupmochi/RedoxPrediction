"""
paiwise



@Author: linlin
@Date: 03.08.23
"""
import os
import time
import pickle
import copy
from typing import Tuple

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from redox_prediction.models.nn.loss import get_loss_func
from redox_prediction.models.nn.early_stopping import get_early_stopping_metric
from redox_prediction.models.nn.optimizer import get_optimizer, get_learning_rate_scheduler
from redox_prediction.utils.logging import AverageMeter
from redox_prediction.models.embed.embed import get_metrics_embed


def evaluate_epoch_pairwise(
		val_loader: DataLoader,
		model: torch.nn.Module,
		criterion: callable,
		device: str = 'cuda',
		print_interval: int = 10,
		verbose: bool = False,
		**kwargs
) -> Tuple[float, dict]:
	"""
	Evaluates a model for one epoch.

	Parameters
	----------
	val_loader: torch_geometric.data.Data
		The validation data.
	model: torch.nn.Module
		The model to evaluate.
	criterion: callable
		The loss function.
	device: str, optional
		The device to use.
	print_interval: int, optional
		The interval at which to print the loss.
	verbose: bool, optional
		Whether to print the loss.
	kwargs
		Additional keyword arguments.

	Returns
	-------
	float
		The average loss.
	models.nn.logging.AverageMeter
	"""
	ave_meters = {key: AverageMeter(keep_all=False) for key in [
		'loss', 'batch_time', 'data_time',
		'metric_time_total', 'metric_time_load', 'metric_time_compute'
	]}

	model.eval()

	with torch.no_grad():
		for i, (data1, data2) in enumerate(val_loader):
			start_time = time.time()

			# Prepare input data. Send the following tensors to the device and
			# exclude the others: 'x', 'edge_index', 'edge_attr', 'y'.
			data1 = data1.to(device)
			data2 = data2.to(device)

			# Compute metrics between embeddings of graphs:
			metrics, metric_meters = get_metrics_embed(
				data1.graph, data2.graph, **kwargs
			)
			# Convert metrics to tensor:
			metrics = torch.tensor(metrics, dtype=torch.float, device=device)

			# Log metric meters:
			total_metric_time = metric_meters['total'].sum
			ave_meters['data_time'].update(
				(time.time() - start_time - total_metric_time) / metrics.size(0),
				metrics.size(0)
			)
			for key, meter in metric_meters.items():
				ave_meters['metric_time_' + key].update(meter)

			start_time = time.time()

			# Compute features:
			output1 = model(data1, return_embeddings=True)
			output2 = model(data2, return_embeddings=True)

			# Log batch time:
			ave_meters['batch_time'].update(
				(time.time() - start_time) / metrics.size(0), metrics.size(0)
			)

			# Compute loss:
			loss = criterion(output1, output2, metrics)

			# Update average meters:
			ave_meters['loss'].update(loss.item(), metrics.size(0))

			if verbose and (i + 1) % print_interval == 0:
				print(
					f'Valid: [{i + 1}/{len(val_loader)}]\t'
					f'Batch Time {ave_meters["batch_time"].avg:.3f} '
					f'({ave_meters["batch_time"].sum:.3f})\t'
					f'Data Time {ave_meters["data_time"].avg:.3f} '
					f'({ave_meters["data_time"].sum:.3f})\t'
					f'Metric Time {ave_meters["metric_time_total"].avg:.3f} '
					f'({ave_meters["metric_time_total"].sum:.3f})\t'
					f'Loss {ave_meters["loss"].avg:.4f} ({ave_meters["loss"].sum:.4f})'
				)

	return ave_meters['loss'].avg, ave_meters


def train_epoch_pairwise(
		train_loader: DataLoader,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		criterion: callable,
		device: str = 'cuda',
		print_interval: int = 10,
		verbose: bool = False,
		**kwargs
) -> Tuple[float, dict]:
	"""
	Trains a model for one epoch.

	Parameters
	----------
	train_loader: torch_geometric.data.Data
		The training data.
	model: torch.nn.Module
		The model to train.
	optimizer: torch.optim.Optimizer
		The optimizer to use.
	criterion: callable
		The loss function.
	device: str, optional
		The device to use.
	print_interval: int, optional
		The interval at which to print the loss.
	verbose: bool, optional
		Whether to print the loss.
	kwargs
		Additional keyword arguments.

	Returns
	-------
	float
		The average loss.
	models.nn.logging.AverageMeter
		The average metric meters.
	"""
	ave_meters = {key: AverageMeter(keep_all=False) for key in [
		'loss', 'batch_time', 'data_time',
		'metric_time_total', 'metric_time_load', 'metric_time_compute'
	]}

	model.train()

	for i, (data1, data2) in enumerate(train_loader):
		start_time = time.time()

		# Prepare input data:
		data1 = data1.to(device)  # todo: do not send to device if not needed (id, graph)
		data2 = data2.to(device)

		# Compute metrics between embeddings of graphs:
		metrics, metric_meters = get_metrics_embed(
			data1.graph, data2.graph, **kwargs
		)
		# Convert metrics to tensor:
		metrics = torch.tensor(metrics, dtype=torch.float, device=device)

		# Log metric meters:
		total_metric_time = metric_meters['total'].sum
		ave_meters['data_time'].update(
			(time.time() - start_time - total_metric_time) / metrics.size(0),
			metrics.size(0)
		)
		for key, meter in metric_meters.items():
			ave_meters['metric_time_' + key].update(meter)

		start_time = time.time()

		optimizer.zero_grad()

		# Compute features:
		output1 = model(data1, return_embeddings=True)
		output2 = model(data2, return_embeddings=True)

		# Compute loss:
		loss = criterion(output1, output2, metrics)
		# print(loss.item())

		# Backward pass:
		loss.backward()
		optimizer.step()

		# Log batch time:
		ave_meters['batch_time'].update(
			(time.time() - start_time) / metrics.size(0), metrics.size(0)
		)

		# Log loss:
		ave_meters['loss'].update(loss.item(), metrics.size(0))

		# Print loss:
		if verbose and i % print_interval == 0:
			print(
				f'Train: [{i}/{len(train_loader)}]\t'
				f'Batch Time {ave_meters["batch_time"].val:.3f} '
				f'({ave_meters["batch_time"].avg:.3f})\t'
				f'Data Time {ave_meters["data_time"].val:.3f} '
				f'({ave_meters["data_time"].avg:.3f})\t'
				f'Metric Time {ave_meters["metric_time_total"].val:.3f} '
				f'({ave_meters["metric_time_total"].avg:.3f})\t'
				f'Loss {ave_meters["loss"].val:.4f} ({ave_meters["loss"].avg:.4f})'
			)

	return ave_meters['loss'].avg, ave_meters


def fit_model_pairwise(
		model: torch.nn.Module,
		train_loader: DataLoader,
		valid_loader: DataLoader = None,
		n_epochs: int = 500,  # 20
		learning_rate: float = 0.01,
		optimizer: str = 'adam',
		early_stopping: bool = True,
		device: str = 'cuda',
		load_state_dict_from: str = 'metric_model_state.pt',
		save_state_time_interval: int = 600,  # todo: change back to 600
		plot_loss: bool = True,  # todo: change back to False
		print_interval: int = 10,
		verbose: bool = True,
		**kwargs
) -> Tuple[torch.nn.Module, dict]:
	"""
	Trains a model on a dataset.

	Parameters
	----------
	model: torch.nn.Module
		A model to train.
	train_loader: torch_geometric.data.Data
		The training data.
	valid_loader: torch_geometric.data.Data, optional
		The validation data.
	n_epochs: int, optional
		The number of epochs to train the model for.
	learning_rate: float, optional
		The evaluation rate.
	optimizer: str, optional
		The optimizer to use. Can be 'adam', 'sgd', or 'adagrad'.
	device: str, optional
		The device to use. Can be 'cpu' or 'cuda'.
	early_stopping: bool, optional
		Whether to use early stopping.
	verbose: bool, optional
		Whether to print information about training.
	kwargs: dict, optional
		Additional keyword arguments. See below.

	Keyword Arguments
	-----------------
	weight_decay: float, optional
		The weight decay.
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
	learning_rate_scheduler: str, optional
		The evaluation rate scheduler to use. Can be 'step', 'exp',
		'plateau', or 'cosine'.
	learning_rate_scheduler_step_size: int, optional
		The step size for the evaluation rate scheduler.
	learning_rate_scheduler_gamma: float, optional
		The gamma for the evaluation rate scheduler.
	learning_rate_scheduler_patience: int, optional
		The patience for the evaluation rate scheduler.

	Returns
	-------
	tuple
		A tuple containing the trained model and a dictionary containing
		information about training.
	"""
	# Move the model to the specified device
	model = model.to(device)

	# Check if training has already been completed:
	train_completed, model, history, best_n_epochs = _check_if_train_completed(
		model, load_state_dict_from
	)
	if train_completed:
		return model, history, best_n_epochs

	# Initialize the optimizer
	optimizer = get_optimizer(
		model.parameters(),
		optimizer,
		learning_rate,
		kwargs.get('weight_decay', 5e-4)
	)

	# Initialize the evaluation rate scheduler:
	learning_rate_scheduler = get_learning_rate_scheduler(
		optimizer,
		learning_rate_scheduler=kwargs.get('learning_rate_scheduler', 'step'),
		learning_rate_decay=kwargs.get('learning_rate_scheduler_gamma', 0.5),
		learning_rate_decay_steps=kwargs.get(
			'learning_rate_scheduler_step_size', 50
		),
	)

	# Initialize the loss function
	loss_func = get_loss_func(True)

	# Initialize the early stopping metric
	early_stopping_mode = kwargs.get('early_stopping_mode', 'min')
	early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 0.0)
	early_stopping_patience = kwargs.get('early_stopping_patience', 100)

	early_stopping_metric = get_early_stopping_metric(
		kwargs.get('early_stopping_metric', 'loss'),
		early_stopping_mode
	)

	# Load the model state if specified:
	if load_state_dict_from is not None and os.path.exists(
			load_state_dict_from
	):
		checkpoint = torch.load(load_state_dict_from)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		learning_rate_scheduler.load_state_dict(
			checkpoint['learning_rate_scheduler_state_dict']
		)

		# Load the training history:
		history = checkpoint['history']
		early_stopping_counter = checkpoint['early_stopping_counter']
		best_valid_metric = checkpoint['best_validation_metric']
		best_model_state = checkpoint['best_model_state_dict']
		start_epoch = checkpoint['epoch'] + 1
		best_n_epochs = checkpoint['best_n_epochs']

		# Log the loaded information:
		print(f'Loaded model state from {load_state_dict_from}.')
	else:
		# Initialize the training history:
		history = {key: AverageMeter(keep_all=True) for key in [  # todo: true?
			'train_loss', 'valid_loss',
			# 'train_metric' 'valid_metric'
			'epoch_time',
			'batch_time_train', 'data_time_train',
			'batch_time_valid', 'data_time_valid',
			'metric_time_total_train', 'metric_time_load_train',
			'metric_time_compute_train',
			'metric_time_total_valid', 'metric_time_load_valid',
			'metric_time_compute_valid',
		]}

		# Initialize the early stopping counter
		early_stopping_counter = 0

		# Initialize the best validation metric
		best_valid_metric = np.inf if early_stopping_mode == 'min' else -np.inf

		# Initialize the start epoch
		start_epoch = 0

		# Initialize the best number of epochs
		best_n_epochs = 0

		# Log the start of training:
		print('\nStarting training:')

	# Initialize the start time for state saving:
	start_time = time.time()

	# Train the model
	for epoch in range(start_epoch, n_epochs):
		# Initialize the epoch start time:
		epoch_start_time = time.time()

		# Train the model
		train_loss, train_meters = train_epoch_pairwise(
			train_loader,
			model,
			optimizer,
			loss_func,
			device,
			print_interval,
			**kwargs
		)

		# Evaluate the model on the validation data
		if valid_loader is not None:
			valid_loss, valid_meters = evaluate_epoch_pairwise(
				valid_loader,
				model,
				loss_func,
				device,
				print_interval,
				**kwargs
			)
		else:
			valid_loss = np.inf

		# Update the evaluation rate scheduler
		learning_rate_scheduler.step()

		# Update the training history
		epoch_time = time.time() - epoch_start_time
		history['epoch_time'].update(epoch_time)
		history['train_loss'].update(train_loss)
		# history['train_metric'].append(train_metric)
		if valid_loader is not None:
			history['valid_loss'].update(valid_loss)
		# history['valid_metric'].append(valid_metric)
		for key, value in history.items():
			if key.endswith('_train'):
				value.update(train_meters[key[:-6]])
			elif key.endswith('_valid') and valid_loader is not None:
				value.update(valid_meters[key[:-6]])

		# Print information about the epoch:
		if verbose and (epoch < 10 or epoch % print_interval == 0):
			if valid_loader is not None:
				print(
					'Epoch: {:04d} \t '
					'Train Loss: {:.4f} \t Valid Loss: {:.4f} \t '
					'Time: {:.4f} \t '
					'# Metrics Computed {:d} - Loaded {:d} '
					'- (Valid) C {:d} - L {:d} \t '
					'LR: {:.5f}'.format(
						epoch,
						train_loss,
						valid_loss,
						epoch_time,
						train_meters['metric_time_compute'].count,
						train_meters['metric_time_load'].count,
						valid_meters['metric_time_compute'].count,
						valid_meters['metric_time_load'].count,
						optimizer.param_groups[0]['lr']
					)
				)
			else:
				print(
					'Epoch: {:04d} \t '
					'Train Loss: {:.4f} \t '
					'Time: {:.4f} \t '
					'# Metrics Computed {:d} - Loaded {:d} \t '
					'LR: {:.5f}'.format(
						epoch,
						train_loss,
						epoch_time,
						train_meters['metric_time_compute'].count,
						train_meters['metric_time_load'].count,
						optimizer.param_groups[0]['lr']
					)
				)

		# Check if early stopping should be used (only if validation data is
		# available):
		if early_stopping and valid_loader is not None:
			# Check if the validation metric improved
			if early_stopping_metric(valid_loss, best_valid_metric):
				# Update the best validation metric
				best_valid_metric = valid_loss
				# Save the model state:
				best_model_state = copy.deepcopy(model.state_dict())
				# Save the best number of epochs:
				best_n_epochs = epoch

				# Reset the early stopping counter
				early_stopping_counter = 0
			else:
				# Increment the early stopping counter
				early_stopping_counter += 1

				# Check if training should be stopped
				if early_stopping_counter >= early_stopping_patience:
					# Print a message
					print('Stopping early.')

					_save_model_states(
						best_model_state, best_valid_metric,
						early_stopping_counter,
						epoch, history, learning_rate_scheduler,
						load_state_dict_from, model, optimizer,
						best_n_epochs,
						train_completed=True,
						metric_matrix=kwargs.get('metric_matrix'),
						train_loss=train_loss,
						valid_loss=valid_loss,
					)

					# Stop training
					break

		# Save the model state if specified:
		if save_state_time_interval is not None:
			time_elapsed = time.time() - start_time
			if time_elapsed > save_state_time_interval:
				# Save the model state
				best_model_state = copy.deepcopy(model.state_dict())
				_save_model_states(
					best_model_state, best_valid_metric, early_stopping_counter,
					epoch, history, learning_rate_scheduler,
					load_state_dict_from, model, optimizer,
					best_n_epochs,
					train_completed=False,
					metric_matrix=kwargs.get('metric_matrix'),
					train_loss=train_loss,
					valid_loss=valid_loss,
				)

				# Reset the start time
				start_time = time.time()

	# Set the model state to the best model state:
	if early_stopping and valid_loader is not None:
		model.load_state_dict(best_model_state)
	else:
		best_model_state = copy.deepcopy(model.state_dict())
		best_n_epochs = epoch

	# Save the model state:
	_save_model_states(
		best_model_state, best_valid_metric, early_stopping_counter,
		epoch, history, learning_rate_scheduler,
		load_state_dict_from, model, optimizer,
		best_n_epochs,
		train_completed=True,
		metric_matrix=kwargs.get('metric_matrix'),
		train_loss=train_loss,
		valid_loss=valid_loss,
	)

	# Plot the training loss:
	if plot_loss:
		from redox_prediction.figures.gnn_plot import plot_training_loss
		plot_training_loss(history, fit_phase='metric', **kwargs)

	# Print a message
	print('Finished training metric evaluation model.')

	# Return the trained model and the training history
	return model, history, best_n_epochs


def _save_model_states(
		best_model_state, best_valid_metric, early_stopping_counter, epoch,
		history, learning_rate_scheduler, load_state_dict_from, model,
		optimizer,
		best_n_epochs,
		train_completed,
		metric_matrix,
		train_loss,
		valid_loss=np.inf,
):
	# Save metric matrix first in case the program stops before saving the
	# model state:
	param_str = os.path.basename(load_state_dict_from).split('.')[1]
	fn_mat = os.path.join(
		os.path.dirname(load_state_dict_from),
		'metric_matrix.' + param_str + '.pkl'
		)
	pickle.dump(metric_matrix, open(fn_mat, 'wb'))

	torch.save(
		{
			'epoch': epoch,
			'best_n_epochs': best_n_epochs,
			'model_state_dict': model.state_dict(),
			'best_model_state_dict': best_model_state,
			'optimizer_state_dict': optimizer.state_dict(),
			'learning_rate_scheduler_state_dict':
				learning_rate_scheduler.state_dict(),
			'history': history,
			'early_stopping_counter': early_stopping_counter,
			'best_validation_metric': best_valid_metric,
			'train_completed': train_completed,
		}, load_state_dict_from
	)
	print(
		'Saved model state at epoch {} with train and valid loss '
		'{:.4f} and {:.4f}.'.format(
			epoch, train_loss, valid_loss
		)
	)


def _check_if_train_completed(model, load_state_dict_from):
	if load_state_dict_from is not None and os.path.exists(load_state_dict_from):
		checkpoint = torch.load(load_state_dict_from)
		if checkpoint['train_completed']:
			print('Training already completed.')
			# If the program stops after early stopping but before assigning
			# the best model state, it is possible that `checkpoint['model_state_dict']`
			# is not the best model state. Therefore, we load the best model
			# state from `checkpoint['best_model_state_dict']`:
			model.load_state_dict(checkpoint['best_model_state_dict'])
			history = checkpoint['history']
			best_n_epochs = checkpoint['best_n_epochs']
			return True, model, history, best_n_epochs
		else:
			return False, model, None, None
	else:
		return False, model, None, None
