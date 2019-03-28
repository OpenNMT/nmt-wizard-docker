"""Definition of `local` storage class"""

import shutil
import os

from nmtwizard.storages.generic import Storage

class LocalStorage(Storage):
    """Storage using the local filesystem."""

    def __init__(self, storage_id=None, basedir=None):
        super(LocalStorage, self).__init__(storage_id or "local")
        self._basedir = basedir

    def get(self, remote_path, local_path, directory=False):
        remote_path = self.build_path(remote_path)
        if directory is None:
            directory = os.path.isdir(remote_path)
        if directory:
            shutil.copytree(remote_path, local_path)
        else:
            shutil.copy(remote_path, local_path)

    def stream(self, remote_path, buffer_size=1024):
        remote_path = self.build_path(remote_path)

        def generate():
            """generator function to stream local file"""
            with open(remote_path, "rb") as f:
                for chunk in iter(lambda: f.read(buffer_size), b''):
                    yield chunk
        return generate()

    def push(self, local_path, remote_path):
        remote_path = self.build_path(remote_path)
        if os.path.isdir(local_path):
            shutil.copytree(local_path, remote_path)
        else:
            if remote_path.endswith('/') or os.path.isdir(remote_path):
                remote_path = os.path.join(remote_path, os.path.basename(local_path))
            dirname = os.path.dirname(remote_path)
            # for local file, there is no path
            if dirname == '':
                dirname = '.'
            if os.path.exists(dirname):
                if not os.path.isdir(dirname):
                    raise ValueError("%s is not a directory" % dirname)
            else:
                os.makedirs(dirname)
            shutil.copy(local_path, remote_path)

    def delete(self, remote_path, recursive=False):
        remote_path = self.build_path(remote_path)
        if recursive:
            if not os.path.isdir(remote_path):
                os.remove(remote_path)
            else:
                shutil.rmtree(remote_path, ignore_errors=True)
        else:
            if not os.path.isfile(remote_path):
                raise ValueError("%s is not a file" % remote_path)
            os.remove(remote_path)

    def listdir(self, remote_path, recursive=False):
        remote_path = self.build_path(remote_path)
        listfile = []
        if not os.path.isdir(remote_path):
            raise ValueError("%s is not a directory" % remote_path)

        def getfiles_rec(path):
            """recursive listdir"""
            for f in os.listdir(path):
                fullpath = os.path.join(path, f)
                if self._basedir:
                    rel_fullpath = self.external_path(fullpath)
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
        old_remote_path = self.build_path(old_remote_path)
        new_remote_path = self.build_path(new_remote_path)
        os.rename(old_remote_path, new_remote_path)

    def exists(self, remote_path):
        remote_path = self.build_path(remote_path)
        return os.path.exists(remote_path)

    def build_path(self, path):
        if path.startswith('/'):
            path = path[1:]
        if self._basedir:
            path = os.path.join(self._basedir, path)
        return path

    def external_path(self, path):
        if self._basedir:
            return os.path.relpath(path, self._basedir)
        else:
            return path
