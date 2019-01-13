"""Client to abstract storage location: local, S3, SSH, etc."""

import os
import shutil

from nmtwizard.logger import get_logger

from nmtwizard.storages.local import LocalStorage
from nmtwizard.storages.ssh import RemoteStorage
from nmtwizard.storages.s3 import S3Storage
from nmtwizard.storages.http import HTTPStorage

LOGGER = get_logger(__name__)


class StorageClient(object):
    """Client to get and push files to a storage."""

    def __init__(self, config=None, tmp_dir=None):
        """Initializes the client.

        Args:
          config: A dictionary mapping storage identifiers to a storage type
            and its configuration.
        """
        self._config = config
        self._tmp_dir = tmp_dir
        self._storages = {}

    def _get_storage(self, path, storage_id=None):
        """Returns the storage implementation based on storage_id or infer it
        from the path. Defaults to the local filesystem.
        """
        if storage_id is None:
            fields = path.split(':')
            if len(fields) > 2:
                raise ValueError('invalid path format: %s' % path)
            elif len(fields) == 2:
                storage_id = fields[0]
                path = fields[1]

        if storage_id is not None:
            if storage_id not in self._storages:
                if self._config is None or storage_id not in self._config:
                    raise ValueError('unknown storage identifier %s' % storage_id)
                config = self._config[storage_id]
                if config['type'] == 's3':
                    credentials = config.get('aws_credentials', {})
                    client = S3Storage(storage_id,
                                       config['bucket'],
                                       access_key_id=credentials.get('access_key_id'),
                                       secret_access_key=credentials.get('secret_access_key'),
                                       region_name=credentials.get('region_name'))
                elif config['type'] == 'ssh':
                    client = RemoteStorage(storage_id,
                                           config['server'],
                                           config['user'],
                                           config.get('password'),
                                           config.get('pkey'),
                                           port=config.get('port', 22),
                                           basedir=config.get('basedir'))
                elif config['type'] == 'http':
                    client = HTTPStorage(storage_id,
                                         config['get_pattern'],
                                         pattern_push=config.get('post_pattern'),
                                         pattern_list=config.get('list_pattern'))
                elif config['type'] == 'local':
                    client = LocalStorage(storage_id,
                                          basedir=config.get("basedir"))
                else:
                    raise ValueError('unsupported storage type %s for %s' % (config['type'], storage_id))
                self._storages[storage_id] = client
            else:
                client = self._storages[storage_id]
        else:
            client = LocalStorage()

        return client, path

    def join(self, path, *paths):
        """Joins the paths according to the storage implementation."""
        client, rel_path = self._get_storage(path)

        if rel_path == path:
            return client.join(path, *paths)

        prefix, _ = path.split(':')
        return '%s:%s' % (prefix, client.join(rel_path, *paths))  # Only join the actual path.

    def split(self, path):
        """Splits the path according to the storage implementation."""
        client, path = self._get_storage(path)
        return client.split(path)

    # Simple wrappers around get().
    def get_file(self, remote_path, local_path, storage_id=None):
        """Retrieves a file from remote_path to local_path."""
        return self.get(remote_path, local_path, directory=False, storage_id=storage_id)

    def get_directory(self, remote_path, local_path, storage_id=None):
        """Retrieves a full directory from remote_path to local_path."""
        return self.get(remote_path, local_path, directory=True, storage_id=storage_id)

    def get(self,
            remote_path,
            local_path,
            directory=False,
            storage_id=None,
            check_integrity_fn=None):
        """Retrieves file or directory from remote_path to local_path."""
        if directory and os.path.isdir(local_path):
            LOGGER.warning('Directory %s already exists', local_path)
        elif not directory and os.path.isfile(local_path):
            LOGGER.warning('File %s already exists', local_path)
        else:
            if not directory and os.path.isdir(local_path):
                local_path = os.path.join(local_path, os.path.basename(remote_path))
            tmp_path = os.path.join(self._tmp_dir, os.path.basename(local_path))
            LOGGER.info('Downloading %s to %s through tmp %s', remote_path, local_path, tmp_path)
            client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
            client.get(remote_path, tmp_path, directory=directory)
            if not os.path.exists(tmp_path):
                raise RuntimeError('download failed: %s not found' % tmp_path)
            if check_integrity_fn is not None and not check_integrity_fn(tmp_path):
                raise RuntimeError('integrity check failed on %s' % tmp_path)
            # in meantime, the file might have been copied
            if os.path.exists(local_path):
                LOGGER.warning('File/Directory created while copying - taking copy')
            else:
                check_integrity_fn = None  # No need to check again.
                shutil.move(tmp_path, local_path)
        if check_integrity_fn is not None and not check_integrity_fn(local_path):
            raise RuntimeError('integrity check failed on %s' % local_path)

    def stream(self, remote_path, buffer_size=1024, storage_id=None):
        """Returns a generator to stream a remote_path file.
        `buffer_size` is the maximal size of each chunk
        """
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        return client.stream(remote_path, buffer_size)

    def push(self, local_path, remote_path, storage_id=None):
        """Pushes a local_path file or directory to storage."""
        if not os.path.exists(local_path):
            raise RuntimeError('%s not found' % local_path)
        if local_path == remote_path:
            return
        LOGGER.info('Uploading %s to %s', local_path, remote_path)
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        client.push(local_path, remote_path)

    def listdir(self, remote_path, recursive=False, storage_id=None):
        """Lists of the files on a storage:
        * if `recursive` returns all of the files present recursively in the directory
        * if not `recursive` returns only first level, directory are indicated with trailing '/'
        """
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        return client.listdir(remote_path, recursive)

    def delete(self, remote_path, recursive=False, storage_id=None):
        """Deletes a file or directory from a storage."""
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        return client.delete(remote_path, recursive)

    def rename(self, old_remote_path, new_remote_path, storage_id=None):
        """Renames a file or directory on storage from old_remote_path to new_remote_path."""
        client_old, old_remote_path = self._get_storage(old_remote_path, storage_id=storage_id)
        client_new, new_remote_path = self._get_storage(new_remote_path, storage_id=storage_id)

        if client_old._storage_id != client_new._storage_id:
            raise ValueError('rename on different storages')

        return client_old.rename(old_remote_path, new_remote_path)

    def exists(self, remote_path, storage_id=None):
        """Checks if file or directory exists on storage."""
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        return client.exists(remote_path)
