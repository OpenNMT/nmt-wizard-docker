"""Shared logic and abstractions of utilities."""

import os
import abc
import argparse
import json
import time
import re
import uuid
import six

from nmtwizard.utils import merge_dict
from nmtwizard.beat_service import start_beat_service
from nmtwizard.logger import get_logger
from nmtwizard.storage import StorageClient

ENVVAR_RE = re.compile(r'\${(.*?)}')
ENVVAR_ABS_RE = re.compile(r'(\${.*?}.*)/(.*)')

logger = get_logger(__name__)

def getenv(m):
    var = m.group(1)
    if var == 'TRAIN_DIR':
        var = 'CORPUS_DIR'
    elif 'TRAIN_' in var:
        var = var.replace('TRAIN_', '')
    return os.getenv(var, '')

def resolve_environment_variables(config):
    """Returns a new configuration with all environment variables replaced."""
    if isinstance(config, dict):
        new_config = {}
        for k, v in six.iteritems(config):
            new_config[k] = resolve_environment_variables(v)
        return new_config
    elif isinstance(config, list):
        new_config = []
        for i, _ in enumerate(config):
            new_config.append(resolve_environment_variables(config[i]))
        return new_config
    elif isinstance(config, six.string_types):
        return ENVVAR_RE.sub(lambda m: getenv(m), config)
    return config

def load_config(config_arg):
    """Loads the configuration from a string, a file, or the standard input."""
    if config_arg.startswith('{'):
        return json.loads(config_arg)
    elif config_arg == '-':
        return json.loads(sys.stdin.read())
    else:
        with open(config_arg) as config_file:
            return json.load(config_file)

def merge_config(a, b):
    """Merges config b in a."""
    return merge_dict(a, b)

@six.add_metaclass(abc.ABCMeta)
class Utility(object):
    """Base class for utilities."""

    def __init__(self):
        self._corpus_dir = os.getenv('CORPUS_DIR', '/root/corpus')
        workspace_dir = os.getenv('WORKSPACE_DIR', '/root/workspace')
        self._output_dir = os.path.join(workspace_dir, 'output')
        self._data_dir = os.path.join(workspace_dir, 'data')
        self._tmp_dir = os.path.join(workspace_dir, 'tmp')
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)
        if not os.path.exists(self._tmp_dir):
            os.makedirs(self._tmp_dir)

    @property
    @abc.abstractmethod
    def name(self):
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

        parser.add_argument('utility_args', nargs=argparse.REMAINDER)

        args, unknown = parser.parse_known_args(args=args)

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
            tmp_dir=self._tmp_dir,
            config=load_config(args.storage_config) if args.storage_config else None)

        logger.info('Starting executing utility %s=%s with args [%s]',
                    self.name, args.image, " ".join(args.utility_args))
        start_time = time.time()

        self.exec_function(resolve_environment_variables(unknown + args.utility_args))

        end_time = time.time()
        logger.info('Finished executing utility in %s seconds', str(end_time-start_time))
