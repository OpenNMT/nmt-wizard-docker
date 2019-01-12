import six
import abc
import os

@six.add_metaclass(abc.ABCMeta)
class Storage(object):
    """Abstract class for storage implementations."""

    def __init__(self, storage_id):
        self._storage_id = storage_id

    # Non conventional storage might need to override these.
    def join(self, path, *paths):
        """Build a path respecting storage prefix
        """
        return os.path.join(path, *paths)

    def split(self, path):
        """Split a path
        """
        return os.path.split(path)

    @abc.abstractmethod
    def get(self, remote_path, local_path, directory=False):
        """Get a file or a directory from a storage to a local file
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stream(self, remote_path, buffer_size=1024):
        """return a generator on a remote file
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def push(self, local_path, remote_path):
        """Push a local file on a remote storage
        """
        raise NotImplementedError()

    def ls(self, remote_path, recursive=False):
        """Return a dictionary with all files in the remote directory
        """
        raise NotImplementedError()

    def delete(self, remote_path, recursive=False):
        """Delete a file or a directory from a storage
        """
        raise NotImplementedError()

    def rename(self, old_remote_path, new_remote_path):
        """Delete a file or a directory from a storage
        """
        raise NotImplementedError()

    def exists(self, remote_path):
        """Check if path is existing
        """
        raise NotImplementedError()
