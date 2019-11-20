"""Shared logic and abstractions of utilities."""

import os
import abc
import argparse
import json
import time
import re
import shutil
import uuid
import six
import sys
import requests
import io
import tempfile

from nmtwizard.beat_service import start_beat_service
from nmtwizard.utils import md5files
from nmtwizard.logger import get_logger

from systran_storages import StorageClient

ENVVAR_RE = re.compile(r'\${(.*?)}')
ENVVAR_ABS_RE = re.compile(r'(\${.*?}.*)/(.*)')

logger = get_logger(__name__)

os.environ.setdefault('CORPUS_DIR', '/root/corpus')
os.environ.setdefault('MODELS_DIR', '/root/models')

def getenv(m, training=True):
    var = m.group(1)
    if 'TRAIN_' in var:
        if not training:
            return "${%s}" % var
        if var == 'TRAIN_DIR':
            var = 'CORPUS_DIR'
        else:
            var = var.replace('TRAIN_', '')
    value = os.getenv(var)
    if value is None:
        raise ValueError('Environment variable %s is not defined' % var)
    return value

def _map_config_fn(a, fn):
    if isinstance(a, dict):
        new_a = {}
        for k, v in six.iteritems(a):
            new_a[k] = _map_config_fn(v, fn)
        return new_a
    elif isinstance(a, list):
        new_a = []
        for v in a:
            new_a.append(_map_config_fn(v, fn))
        return new_a
    else:
        return fn(a)

def resolve_environment_variables(config, training=True):
    """Returns a new configuration with all environment variables replaced."""
    def _map_fn(value):
        if not isinstance(value, six.string_types):
            return value
        return ENVVAR_RE.sub(lambda m: getenv(m, training=training), value)
    return _map_config_fn(config, _map_fn)

def resolve_remote_files(config, local_dir, storage_client):
    """Downloads remote files present in config locally."""
    def _map_fn(value):
        if not isinstance(value, six.string_types) or not storage_client.is_managed_path(value):
            return value
        storage_id, remote_path = storage_client.parse_managed_path(value)
        if remote_path[0] == '/':
            remote_path = remote_path[1:]
        local_path = os.path.join(local_dir, storage_id, remote_path)
        # can be a file or a directory
        storage_client.get(remote_path, local_path, storage_id=storage_id, directory=None)
        return local_path
    return _map_config_fn(config, _map_fn)

def load_config(config_arg):
    """Loads the configuration from a string, a file, or the standard input."""
    if config_arg.startswith('{'):
        return json.loads(config_arg)
    elif config_arg == '-':
        return json.loads(sys.stdin.read())
    else:
        with open(config_arg) as config_file:
            return json.load(config_file)

@six.add_metaclass(abc.ABCMeta)
class Utility(object):
    """Base class for utilities."""

    def __init__(self):
        self._corpus_dir = os.getenv('CORPUS_DIR')
        workspace_dir = os.getenv('WORKSPACE_DIR', '/root/workspace')
        self._output_dir = os.path.join(workspace_dir, 'output')
        self._data_dir = os.path.join(workspace_dir, 'data')
        self._shared_dir = os.path.join(workspace_dir, 'shared')
        self._tmp_dir = tempfile.mkdtemp()
        try:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            if not os.path.exists(self._data_dir):
                os.makedirs(self._data_dir)
            if not os.path.exists(self._shared_dir):
                os.makedirs(self._shared_dir)
        except OSError:
            pass

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def declare_arguments(self, parser):
        raise NotImplementedError()

    @abc.abstractmethod
    def exec_function(self, args):
        """Launch the utility with provided params
        """
        raise NotImplementedError()

    def run(self, args=None):
        """Main entrypoint."""
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--storage_config', default=None,
                            help=('Configuration of available storages as a file or a JSON string. '
                                  'Setting "-" will read from the standard input.'))
        parser.add_argument('-t', '--task_id', default=None,
                            help="Identifier of this run.")
        parser.add_argument('-i', '--image', default="?",
                            help="Full URL (registry/image:tag) of the image used for this run.")
        parser.add_argument('-b', '--beat_url', default=None,
                            help=("Endpoint that listens to beat requests "
                                  "(push notifications of activity)."))
        parser.add_argument('-bi', '--beat_interval', default=30, type=int,
                            help="Interval of beat requests in seconds.")
        parser.add_argument('--statistics_url', default=None,
                            help=('Endpoint that listens to statistics summaries generated '
                                  'at the end of the execution'))

        parser.add_argument('-ms', '--model_storage', default=os.environ["MODELS_DIR"],
                            help='Model storage in the form <storage_id>:[<path>].')
        parser.add_argument('-msr', '--model_storage_read', default=None,
                            help=('Model storage to read from, in the form <storage_id>:[<path>] '
                                  '(defaults to model_storage).'))
        parser.add_argument('-msw', '--model_storage_write', default=None,
                            help=('Model storage to write to, in the form <storage_id>:[<path>] '
                                  '(defaults to model_storage).'))
        parser.add_argument('-c', '--config', default=None,
                            help=('Configuration as a file or a JSON string. '
                                  'Setting "-" will read from the standard input.'))
        parser.add_argument('-m', '--model', default=None,
                            help='Model to load.')
        parser.add_argument('-g', '--gpuid', default="0",
                            help="Comma-separated list of 1-indexed GPU identifiers (0 for CPU).")
        parser.add_argument('--no_push', default=False, action='store_true',
                            help='Do not push model.')

        self.declare_arguments(parser)
        args = parser.parse_args(args=args)

        if args.task_id is None:
            args.task_id = str(uuid.uuid4())

        self._task_id = args.task_id
        self._image = args.image

        start_beat_service(
            os.uname()[1],
            args.beat_url,
            args.task_id,
            interval=args.beat_interval)

        self._storage = StorageClient(
            config=load_config(args.storage_config) if args.storage_config else None)

        if args.model_storage_read is None:
            args.model_storage_read = args.model_storage
        if args.model_storage_write is None:
            args.model_storage_write = args.model_storage

        self._model_storage_read = args.model_storage_read
        self._model_storage_write = args.model_storage_write

        # for backward compatibility - convert singleton in int
        args.gpuid = args.gpuid.split(',')
        args.gpuid = [int(g) for g in args.gpuid]
        if len(args.gpuid) == 1:
            args.gpuid = args.gpuid[0]

        self._gpuid = args.gpuid

        self._config = load_config(args.config) if args.config is not None else None
        self._model = args.model
        self._no_push = args.no_push

        logger.info('Starting executing utility %s=%s', self.name, args.image)
        start_time = time.time()
        stats = self.exec_function(args)
        end_time = time.time()
        logger.info('Finished executing utility in %s seconds', str(end_time-start_time))

        if args.statistics_url is not None:
            requests.post(args.statistics_url, json={
                'task_id': self._task_id,
                'start_time': start_time,
                'end_time': end_time,
                'statistics': stats or {}
            })

    def convert_to_local_file(self, nextval, is_dir = False):
        new_val = []
        for val in nextval:
            inputs = val.split(',')
            local_inputs = []
            for remote_input in inputs:
                local_input = os.path.join(self._data_dir, self._storage.split(remote_input)[-1])
                if is_dir:
                    self._storage.get_directory(remote_input, local_input)
                else:
                    self._storage.get_file(remote_input, local_input)
                local_inputs.append(local_input)
            new_val.append(','.join(local_inputs))
        return new_val


def build_model_dir(model_dir, objects, config, check_integrity_fn):
    """Prepares the model directory based on the model package."""
    if os.path.exists(model_dir):
        raise ValueError("model directory %s already exists" % model_dir)
    else:
        logger.info("Building model package in %s", model_dir)
    os.mkdir(model_dir)
    for target, source in six.iteritems(objects):
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(model_dir, target))
        else:
            shutil.copyfile(source, os.path.join(model_dir, target))
    config_path = save_model_config(model_dir, config)
    objects[os.path.basename(config_path)] = config_path
    if "description" in config:
        readme_path = os.path.join(model_dir, 'README.md')
        with io.open(readme_path, 'w', encoding='utf-8') as readme_file:
            readme_file.write(config['description'])
        objects['README.md'] = readme_path
    md5 = md5files((k, v) for k, v in six.iteritems(objects) if check_integrity_fn(k))
    with open(os.path.join(model_dir, "checksum.md5"), "w") as f:
        f.write(md5)

def _get_config_path(model_dir):
    return os.path.join(model_dir, 'config.json')

def save_model_config(model_dir, config):
    """Saves the model configuration."""
    config_path = _get_config_path(model_dir)
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file)
    return config_path

def load_model_config(model_dir):
    """Loads the model configuration."""
    config_path = _get_config_path(model_dir)
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def check_model_dir(model_dir, check_integrity_fn):
    """Compares model package MD5."""
    logger.info("Checking integrity of model package %s", model_dir)
    md5_file = os.path.join(model_dir, "checksum.md5")
    if not os.path.exists(md5_file):
        return True
    md5ref = None
    with open(md5_file, "r") as f:
        md5ref = f.read().strip()
    files = os.listdir(model_dir)
    md5check = md5files([(f, os.path.join(model_dir, f)) for f in files if check_integrity_fn(f)])
    return md5check == md5ref

def fetch_model(storage, remote_model_path, model_path, check_integrity_fn):
    """Downloads the remote model."""
    storage.get(remote_model_path, model_path, directory=True,
                check_integrity_fn=lambda m: check_model_dir(m, check_integrity_fn))
    os.environ['MODEL_DIR'] = model_path
    return load_model_config(model_path)
