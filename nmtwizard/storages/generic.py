import os
import abc
import six
import fcntl
import contextlib

from nmtwizard.logger import get_logger

LOGGER = get_logger(__name__)

@contextlib.contextmanager
def lock(fname):
    with open(fname, "w") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        yield
        fcntl.lockf(f, fcntl.LOCK_UN)

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
    def _check_existing_file(self, remote_path, local_path):
        """Check if remote file already exists locally. Storage specific method."""
        return False

    @abc.abstractmethod
    def _get_file_safe(self, remote_path, local_path):
        """Get a file from a qualified remote_path to a local_path safely
        """
        raise NotImplementedError()

    def _get_checksum_file(self, local_path):
        """return checksum sum used by storage or None
        """
        return None

    def _sync_file(self, remote_path, local_path):
        (local_dir, basename) = os.path.split(local_path)
        if os.path.isdir(local_path) or (local_dir != "" and basename == ""):
            local_path = os.path.join(local_dir, os.path.basename(remote_path))
            basename = remote_path
        if local_dir != "" and not os.path.exists(local_dir):
            os.makedirs(local_dir)
        if self._check_existing_file(remote_path, local_path):
            return
        LOGGER.info('Synchronizing file %s to %s', remote_path, local_path)
        lock_file = '%s.lock' % (local_path if not local_path.endswith('/') else local_path[:-1])
        lock_dirname = os.path.dirname(lock_file)
        try:
            os.makedirs(lock_dirname)
        except OSError:
            pass
        with lock(lock_file):
            self._get_file_safe(remote_path, local_path)

    def get(self, remote_path, local_path, directory=False):
        """Get a file or a directory from a storage to a local file
        """
        if directory is None:
            directory = self.isdir(remote_path)

        if directory:
            lock_file = '%s.lock' % (local_path if not local_path.endswith('/') else local_path[:-1])
            lock_dirname = os.path.dirname(lock_file)
            try:
                os.makedirs(lock_dirname)
            except OSError:
                pass
            with lock(lock_file):
                allfiles = {}
                for root, dirs, files in os.walk(local_path):
                    for f in files:
                        allfiles[os.path.join(root, f)] = 1

                for f in self.listdir(remote_path, recursive=True):
                    internal_path = self._internal_path(f)
                    assert internal_path.startswith(remote_path)
                    subpath = internal_path[len(remote_path)+1:]
                    path = os.path.join(local_path, subpath)
                    if f.endswith('/'):
                        if not os.path.isdir(path):
                            os.makedirs(path)
                    else:
                        dir_path = os.path.dirname(path)
                        if not os.path.isdir(dir_path):
                            os.makedirs(dir_path)
                        if path in allfiles:
                            del allfiles[path]
                            checksum_file = self._get_checksum_file(path)
                            if checksum_file is not None:
                                del allfiles[checksum_file]
                        self._sync_file(f, path)
                for f in allfiles:
                    os.remove(f)
        else:
            self._sync_file(remote_path, local_path)

    @abc.abstractmethod
    def stream(self, remote_path, buffer_size=1024):
        """return a generator on a remote file
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def push_file(self, local_path, remote_path):
        """Push a local file on a remote storage path
        """
        raise NotImplementedError()

    def push(self, local_path, remote_path):
        """Push a file or directory on a remote storage
        """
        if os.path.isfile(local_path):
            if remote_path.endswith('/') or self.isdir(remote_path):
                remote_path = os.path.join(remote_path, os.path.basename(local_path))
            dirname = os.path.dirname(remote_path)
            self.mkdir(dirname)
            self.push_file(local_path, remote_path)
        else:
            def push_rec(local_path, remote_path):
                files = os.listdir(local_path)
                for f in files:
                    local_filepath = os.path.join(local_path, f)
                    remote_filepath = os.path.join(remote_path, f)
                    if os.path.isdir(local_filepath):
                        push_rec(local_filepath, remote_filepath)
                    else:
                        self.push(local_filepath, remote_filepath)
            push_rec(local_path, remote_path)

    @abc.abstractmethod
    def mkdir(self, remote_path):
        """build a directory - might not be effective for some storages like S3
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def listdir(self, remote_path, recursive=False):
        """Return a list with all files and directory in the remote directory
           The files have full path, directory ends with trailing /
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _delete_single(self, remote_path, isdir):
        """Return a single file or directory
        """
        raise NotImplementedError()

    def delete(self, remote_path, recursive=False):
        """Delete a file or a directory from a storage
        """
        if self.isdir(remote_path):
            def rm_rec(path):
                files = self.listdir(remote_path=path)
                for f in files:
                    internal_path = self._internal_path(f)
                    if internal_path.endswith('/'):
                        rm_rec(internal_path)
                    else:
                        self._delete_single(internal_path, False)
                self._delete_single(path, True)
            if not recursive:
                raise ValueError("non recursive delete can not delete directory")
            rm_rec(remote_path)
        else:
            self._delete_single(remote_path, False)

    @abc.abstractmethod
    def rename(self, old_remote_path, new_remote_path):
        """Delete a file or a directory from a storage
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def exists(self, remote_path):
        """Check if path is existing
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def isdir(self, remote_path):
        """Check if path is a directory
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _internal_path(self, path):
        """convert a storage path into a path/key to the actual object in storage logic
        """
        raise NotImplementedError()

    def _external_path(self, path):
        """convert the internal path to the external user path
        """
        return path
