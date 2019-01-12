import shutil
import os

from generic import Storage

class LocalStorage(Storage):
    """Storage using the local filesystem."""

    def __init__(self, storage_id=None, basedir=None):
        super(LocalStorage, self).__init__(storage_id or "local")
        self._basedir = basedir

    def get(self, remote_path, local_path, directory=False):
        if self._basedir:
            remote_path = os.path.join(self._basedir, remote_path)
        if directory:
            shutil.copytree(remote_path, local_path)
        else:
            shutil.copy(remote_path, local_path)

    def stream(self, remote_path, buffer_size=1024):
        if self._basedir:
            remote_path = os.path.join(self._basedir, remote_path)

        def generate():
                with open(remote_path, "rb") as f:
                    for chunk in iter(lambda: f.read(buffer_size), b''):
                        yield chunk
        return generate()

    def push(self, local_path, remote_path):
        if self._basedir:
            remote_path = os.path.join(self._basedir, remote_path)
        if remote_path.endswith('/') or os.path.isdir(remote_path):
            remote_path = os.path.join(remote_path, os.path.basename(local_path))
        dirname = os.path.dirname(remote_path)
        if os.path.exists(dirname):
            if not os.path.isdir(dirname):
                raise ValueError("%s is not a directory" % dirname)
        else:
            os.makedirs(dirname)
        self.get(local_path, remote_path, directory=os.path.isdir(local_path))

    def delete(self, remote_path, recursive=False):
        if self._basedir:
            remote_path = os.path.join(self._basedir, remote_path)
        if recursive:
            if not os.path.isdir(remote_path):
                os.remove(remote_path)
            else:
                shutil.rmtree(remote_path, ignore_errors=True)
        else:
            if not os.path.isfile(remote_path):
                raise ValueError("%s is not a file" % remote_path)
            os.remove(remote_path)

    def ls(self, remote_path, recursive=False):
        if self._basedir:
            remote_path = os.path.join(self._basedir, remote_path)
        listfile = []
        if not os.path.isdir(remote_path):
            raise ValueError("%s is not a directory" % remote_path)

        def getfiles(path):
            for f in os.listdir(path):
                fullpath = os.path.join(path, f)
                rel_fullpath = fullpath
                if self._basedir:
                    rel_fullpath = os.path.relpath(fullpath, self._basedir)
                if os.path.isdir(fullpath):
                    if recursive:
                        getfiles(fullpath)
                    else:
                        listfile.append(rel_fullpath+'/')
                else:
                    listfile.append(rel_fullpath)

        getfiles(remote_path)

        return listfile

    def rename(self, old_remote_path, new_remote_path):
        if self._basedir:
            old_remote_path = os.path.join(self._basedir, old_remote_path)
        if self._basedir:
            new_remote_path = os.path.join(self._basedir, new_remote_path)
        os.rename(old_remote_path, new_remote_path)

    def exists(self, remote_path):
        if self._basedir:
            remote_path = os.path.join(self._basedir, remote_path)
        return os.path.exists(remote_path)
