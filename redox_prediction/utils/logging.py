"""
logging



@Author: linlin
@Date: 03.08.23
"""
import io
import sys
import time

import functools
import types

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
		self.stdout = sys.stdout
		self.logfile = open(filename, "a")


	def write(self, message):
		self.stdout.write(message)
		self.logfile.write(message)


	def flush(self):
		self.stdout.flush()
		self.logfile.flush()


class StringAndStdoutWriter:
	def __init__(self):
		self.string_buffer = io.StringIO()
		self.stdout = sys.stdout


	def write(self, text):
		self.string_buffer.write(text)
		self.stdout.write(text)


	def getvalue(self):
		return self.string_buffer.getvalue()


	def flush(self):
		self.string_buffer.flush()
		self.stdout.flush()


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
	from redox_prediction.models.nn.utils import nn_to_dict
	from sklearn.base import BaseEstimator
	from redox_prediction.models.embed.kernel import estimator_to_dict

	def custom_serializer(obj):
		"""Custom serializer function for non-serializable types."""
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, np.int64):
			return int(obj)
		elif isinstance(obj, np.float32):
			return float(obj)
		elif isinstance(obj, np.float64):
			return float(obj)
		elif isinstance(obj, np.bool_):
			return bool(obj)
		elif isinstance(obj, torch.Tensor):
			return obj.tolist()
		elif isinstance(obj, functools.partial):
			return {
				'__partial__': True, 'func': obj.func, 'args': obj.args,
				'keywords': obj.keywords
			}
		elif isinstance(obj, types.FunctionType):
			return {
				'__function__': True, 'module': obj.__module__,
				'name': obj.__name__
			}

		# Convert the subclasses of torch.nn.Module to a dict:
		elif isinstance(obj, torch.nn.Module):
			return nn_to_dict(obj, with_state_dict=False)
		# Convert the sklearn.estimator (including GraphKernel, GEDModel) to a dict:
		elif isinstance(obj, BaseEstimator):
			return estimator_to_dict(obj)
		# Convert the AverageMeter to a dict:
		elif isinstance(obj, AverageMeter):
			return obj.to_dict(with_history=False)
		else:
			raise TypeError(
				"Object of type {} is not JSON serializable".format(type(obj))
			)


	try:
		import json
		# Use json.loads to convert the string to a dictionary:
		resu_json = json.loads(json.dumps(resu, default=custom_serializer))
	except TypeError:
		raise ValueError(
			'resu_serializable[key] is not json serializable.'
		)

	return resu_json
