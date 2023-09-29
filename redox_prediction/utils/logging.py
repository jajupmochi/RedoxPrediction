"""
logging



@Author: linlin
@Date: 03.08.23
"""
import sys
import functools
import time

import numpy as np


class AverageMeter(object):
	"""
	Computes and stores the average and current value.

	References
	----------
	https://github.com/priba/siamese_ged/blob/master/LogMetric.py#L16
	"""


	def __init__(
			self,
			keep_all: bool = False,
	):
		"""
		Initializes the average meter.

		Parameters
		----------
		keep_all: bool, optional
			Whether to keep all the values in the history. If False, only the
			last value is kept. Default: False.
		"""
		self.reset()
		self.keep_all = keep_all


	def reset(self):
		"""
		Resets the average meter.
		"""
		self.val = 0
		self.sum = 0
		self.count = 0
		self.history = []


	def tic(self):
		self.tic_time = time.time()


	def toc(self):
		self.toc_time = time.time()
		self.update(self.toc_time - self.tic_time)


	def update(
			self,
			*args,
	):
		"""
		Updates the average meter.

		Parameters
		----------
		*args: Union[float, int, AverageMeter]
			The value to add. If an AverageMeter is passed, its value and count
			are added. If a float is passed, it is added with a count. If
			only a float is passed, the count is assumed to be 1.
		"""
		if len(args) == 1 and isinstance(args[0], AverageMeter):
			self._update(
				args[0].avg,
				args[0].count,
				args[0].history,
			)
		elif len(args) == 1:
			self._update(
				args[0],
				1,
			)
		elif len(args) == 2:
			self._update(
				args[0],
				args[1],
			)
		else:
			raise ValueError('Wrong number of arguments.')


	def _update(
			self,
			val: float,
			count: int,
			history: list = None,
	):
		"""
		Updates the average meter.

		Parameters
		----------
		val: float
			The value to add.
		count: int
			The number of values to add.
		history: list, optional
			The history to be added. If None, val is added to the history.
		"""
		self.val = val
		self.sum += val * count
		self.count += count
		if self.keep_all:
			if history is None or len(history) == 0:
				self.history.append(val)
			else:
				self.history += history


	def to_dict(
			self,
			with_history: bool = False,
	):
		"""
		Returns the average meter as a dictionary.

		Parameters
		----------
		with_history: bool, optional
			Whether to include the history. Default: False.

		Returns
		-------
		dict
			The average meter as a dictionary.
		"""
		dict_avg_meter = {
			'avg': self.avg,
			'std': self.std,
			'count': self.count,
		}
		if with_history:
			dict_avg_meter['history'] = self.history
		return dict_avg_meter


	def __len__(self):
		"""
		Returns the number of values added.
		"""
		return self.count


	@property
	def avg(self):
		"""
		Returns the average value.
		"""
		if self.count == 0:
			return 0
		return self.sum / self.count


	@property
	def std(self):
		"""
		Returns the standard deviation.
		"""
		if self.count == 0:
			return 0
		if len(self.history) == 0:
			return np.nan
		return np.std(self.history, axis=0)


class Logger(object):
	"""
	Logger.
	"""


	def __init__(
			self,
			log_dir: str,
			force: bool = False,
	):
		"""
		Initializes the logger.

		Parameters
		----------
		log_dir: str
			The directory to log to.
		force: bool, optional
			Whether to force the logging. If True, the log directory is
			removed if it already exists. Default: False.
		"""
		# clean previous logged data under the same directory name
		self._remove(log_dir, force)

		# create the summary writer object
		from tensorboardX import SummaryWriter
		self._writer = SummaryWriter(log_dir)

		self.global_step = 0


	def __del__(self):
		self._writer.close()


	def _remove(
			self,
			log_dir: str,
			force: bool,
	):
		"""
		Removes the log directory.

		Parameters
		----------
		log_dir: str
			The directory to log to.
		force: bool, optional
			Whether to force the logging. If True, the log directory is
			removed if it already exists. Default: False.
		"""
		if force:
			if os.path.exists(log_dir):
				shutil.rmtree(log_dir)


class PrintLogger:
	def __init__(self, filename):
		self.terminal = sys.stdout
		self.logfile = open(filename, "a")


	def write(self, message):
		self.terminal.write(message)
		self.logfile.write(message)


	def flush(self):
		pass

	# def close(self):
	# 	self.logfile.close()


def resu_to_serializable(
		resu: dict,
):
	"""
	Convert the results to a serializable format.

	Parameters
	----------
	resu: dict
		The results to convert.

	Returns
	-------
	dict
		The converted results.
	"""
	import torch
	from gklearn.kernels import GraphKernel
	from redox_prediction.models.nn.utils import nn_to_dict
	from redox_prediction.models.embed.kernel import estimator_to_dict

	resu_serializable = {}
	for key in resu.keys():
		# Convert the AverageMeter to a dict:
		if isinstance(resu[key], AverageMeter):
			resu_serializable[key] = resu[key].to_dict(with_history=False)

		# Convert the subclasses of torch.nn.Module to a dict:
		elif issubclass(type(resu[key]), torch.nn.Module):
			resu_serializable[key] = nn_to_dict(
				resu[key], with_state_dict=False
			)

		# Convert the `gklearn.models.graph_kernel.GraphKernel` to a dict:
		elif issubclass(type(resu[key]), GraphKernel):
			resu_serializable[key] = kernel_to_dict(resu[key])

		# Convert the numpy.ndarray to a list:
		elif isinstance(resu[key], np.ndarray):
			resu_serializable[key] = resu[key].tolist()

		# Convert a list to a dict only if it is a list of dicts
		# (for results of trials):
		elif isinstance(resu[key], list) and len(resu[key]) > 0 and isinstance(
				resu[key][0], dict
		):
			resu_serializable[key] = {}
			for idx, val in enumerate(resu[key]):
				resu_serializable[key][idx] = resu_to_serializable(val)

		# Convert a tuple to a dict only if it is a list of estimators
		# (when using kernels/geds + predictors):
		elif isinstance(resu[key], tuple) and len(resu[key]) > 0 and issubclass(
				type(resu[key][0]), GraphKernel
		):
			resu_serializable[key] = {}
			for idx, val in enumerate(resu[key]):
				resu_serializable[key][idx] = estimator_to_dict(val)

		# Convert a dict to a dict only if it is a dict of dicts:
		elif isinstance(resu[key], dict):
			resu_serializable[key] = resu_to_serializable(resu[key])

		# Convert partial functions to a dict:
		# This can happen in `params_best` in the results of each trial for
		# models such as `Treelet`.
		elif isinstance(resu[key], functools.partial):
			resu_serializable[key] = str(resu[key])

		# Convert functions to a dict:
		# This can happen in `params_best` in the results of each trial for
		# models such as `ShortestPath`.
		elif hasattr(resu[key], '__call__'):
			resu_serializable[key] = resu[key].__module__ + '.' + resu[
				key].__name__

		# Convert the type `float32` to `float`:
		elif type(resu[key]) == np.float32:
			resu_serializable[key] = float(resu[key])
		elif isinstance(resu[key], list) and len(resu[key]) > 0 and type(
				resu[key][0]
		) == np.float32:
			resu_serializable[key] = [float(val) for val in resu[key]]

		# Convert the type `int64` to `int`:
		elif type(resu[key]) == np.int64:
			resu_serializable[key] = int(resu[key])
		elif isinstance(resu[key], list) and len(resu[key]) > 0 and type(
				resu[key][0]
		) == np.int64:
			resu_serializable[key] = [int(val) for val in resu[key]]

		else:
			resu_serializable[key] = resu[key]

		# todo
		try:
			import json
			json.dumps(resu_serializable[key])
		except TypeError:
			print()
			print(
				'The following key is ignored because it is not json serializable:'
			)
			# print('resu_serializable[key]: ', resu_serializable[key])
			# print('resu[key]: ', resu[key])
			print('key: ', key)
			print('type(resu[key]): ', type(resu[key]))
			print(
				'type(resu_serializable[key]): ', type(resu_serializable[key])
			)
			# raise ValueError('resu_serializable[key] is not json serializable.')
			# Delete the key:
			del resu_serializable[key]

	return resu_serializable
