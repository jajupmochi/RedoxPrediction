"""
resource



@Author: linlin
@Date: 05.10.23
"""
import os
import threading
from datetime import datetime
import psutil
import json

from redox_prediction.utils.utils import format_bytes


def get_computing_resource_info(return_json=False, return_joblib=False):
	"""
	Get computing resource info.
	"""
	resource = {
		# machine info:
		'machine': {
			'name': os.uname()[1],
			'os': os.uname()[0],
			'os_release': os.uname()[2],
			'os_version': os.uname()[3],
			'architecture': os.uname()[4],
		},
		# cpu info:
		'cpu': {
			'num_physical_cores': psutil.cpu_count(logical=False),
			'num_logical_cores': psutil.cpu_count(logical=True),
			'num_usable_logical_cores': len(psutil.Process().cpu_affinity()),
			'cpu_freq': {
				k: f"{v:.2f} MHz" for k, v in
				psutil.cpu_freq()._asdict().items()
			},
		},
		'memory': {
			'virtual_memory': {
				k: format_bytes(v) for k, v in
				psutil.virtual_memory()._asdict().items()
			},
		},
		# process info:
		'process': {
			'process_id': psutil.Process().pid,
			'process_name': psutil.Process().name(),
			'create_time': datetime.fromtimestamp(
				psutil.Process().create_time()
			).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
			# 'wait_process_ids': os.wait(),
			'parent_process_id': psutil.Process().ppid(),
			'username': psutil.Process().username(),
			'user_ids': psutil.Process().uids(),
			'group_ids': psutil.Process().gids(),
			'cpu_num': psutil.Process().cpu_num(),
			'memory_info': {
				k: format_bytes(v) for k, v in
				psutil.Process().memory_info()._asdict().items()
			},
		},
		# thread info:
		'thread': {
			'id': threading.get_ident(),
		},
	}

	if return_joblib:
		from joblib.parallel import get_active_backend
		resource['joblib'] = {
			'active_backend': get_active_backend()[0].__class__.__name__,
			'n_jobs': get_active_backend()[1],
		}

	# If using SLURM for parallel computing:
	if 'SLURM_JOB_ID' in os.environ:
		parallel = {
			'type': 'slurm',
			'node_id': os.environ['SLURM_NODEID'],
			'node_local_task_id': os.environ['SLURM_LOCALID'],
			'job_id': os.environ['SLURM_JOB_ID'],
			'job_name': os.environ['SLURM_JOB_NAME'],
			'job_nodelist': os.environ['SLURM_JOB_NODELIST'],
			'job_ntasks': os.environ['SLURM_NTASKS'],
			'job_cpus_per_task': os.environ['SLURM_CPUS_PER_TASK']
		}
		resource['parallel'] = parallel

	if return_json:
		resource = json.dumps(resource, indent=4)

	return resource


if __name__ == "__main__":
	print(get_computing_resource_info(return_json=True))
