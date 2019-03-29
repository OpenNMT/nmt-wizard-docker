"""Definition of `local` storage class"""

import shutil
import os
import tempfile

from nmtwizard.storages.generic import Storage

class LocalStorage(Storage):
    """Storage using the local filesystem."""

    def __init__(self, storage_id=None, basedir=None):
        super(LocalStorage, self).__init__(storage_id or "local")
        self._basedir = basedir

    def _get_file_safe(self, remote_path, local_path):
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            shutil.copy(remote_path, tmpfile.name)
            os.rename(tmpfile.name, local_path)

    def _check_existing_file(self, remote_path, local_path):
        return False

    def stream(self, remote_path, buffer_size=1024):
        def generate():
            """generator function to stream local file"""
            with open(remote_path, "rb") as f:
                for chunk in iter(lambda: f.read(buffer_size), b''):
                    yield chunk
        return generate()

    def push_file(self, local_path, remote_path):
        shutil.copy(local_path, remote_path)

    def mkdir(self, remote_path):
        if not os.path.exists(remote_path):
            os.makedirs(remote_path)

    def _delete_single(self, remote_path, isdir):
        if not os.path.isdir(remote_path):
            os.remove(remote_path)
        else:
            shutil.rmtree(remote_path, ignore_errors=True)

    def listdir(self, remote_path, recursive=False):
        listfile = []
        if not os.path.isdir(remote_path):
            raise ValueError("%s is not a directory" % remote_path)

        def getfiles_rec(path):
            """recursive listdir"""
            for f in os.listdir(path):
                fullpath = os.path.join(path, f)
                if self._basedir:
                    rel_fullpath = self._external_path(fullpath)
                else:
                    rel_fullpath = fullpath
                if os.path.isdir(fullpath):
                    if recursive:
                        getfiles_rec(fullpath)
                    else:
                        listfile.append(rel_fullpath+'/')
                else:
                    listfile.append(rel_fullpath)

        getfiles_rec(remote_path)

        return listfile

    def rename(self, old_remote_path, new_remote_path):
        os.rename(old_remote_path, new_remote_path)

    def exists(self, remote_path):
        return os.path.exists(remote_path)

    def isdir(self, remote_path):
        return os.path.isdir(remote_path)

    def _internal_path(self, path):
        if self._basedir:
            if path.startswith('/'):
                path = path[1:]
            path = os.path.join(self._basedir, path)
        return path

    def _external_path(self, path):
        if self._basedir:
            return os.path.relpath(path, self._basedir)
        return path
