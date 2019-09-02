"""Client to abstract storage location: local, S3, SSH, etc."""

import os
import logging

from nmtwizard.storages.local import LocalStorage
from nmtwizard.storages.ssh import RemoteStorage
from nmtwizard.storages.s3 import S3Storage
from nmtwizard.storages.swift import SwiftStorage
from nmtwizard.storages.http import HTTPStorage

LOGGER = logging.getLogger(__name__)

class StorageClient(object):
    """Client to get and push files to a storage."""

    def __init__(self, config=None):
        """Initializes the client.

        Args:
          config: A dictionary mapping storage identifiers to a storage type
            and its configuration.
        """
        self._config = config
        self._storages = {}

    def is_managed_path(self, path):
        """Returns True if the path references a storage managed by this client."""
        if self._config is None:
            return False
        fields = path.split(':')
        return len(fields) == 2 and fields[0] in self._config

    def parse_managed_path(self, path):
        """Returns the storage ID and the full path from a managed path."""
        fields = path.split(':')
        return fields[0], fields[1]

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
                                       region_name=credentials.get('region_name'),
                                       assume_role=credentials.get('assume_role'),
                                       transfer_config=credentials.get('transfer_config'))
                elif config['type'] == 'swift':
                    credentials = config.get('os_credentials', {})
                    client = SwiftStorage(storage_id,
                                          config['container'],
                                          os_username=credentials.get('os_username'),
                                          os_password=credentials.get('os_password'),
                                          os_tenant_name=credentials.get('os_tenant_name'),
                                          os_auth_url=credentials.get('os_auth_url'),
                                          transfer_config=credentials.get('transfer_config')
                                          )
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

        return client, client._internal_path(path)

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
        LOGGER.info('Synchronizing %s to %s', remote_path, local_path)
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        client.get(
            remote_path,
            local_path,
            directory=directory,
            check_integrity_fn=check_integrity_fn)
        if not os.path.exists(local_path):
            raise RuntimeError('Failed to synchronize %s' % local_path)

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
