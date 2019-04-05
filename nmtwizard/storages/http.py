"""Definition of `http` storage class"""

import os
import requests
import tempfile

from nmtwizard.storages.generic import Storage

class HTTPStorage(Storage):
    """Simple http file-only storage."""

    def __init__(self, storage_id, pattern_get, pattern_push=None, pattern_list=None):
        super(HTTPStorage, self).__init__(storage_id)
        self._pattern_get = pattern_get
        self._pattern_push = pattern_push
        self._pattern_list = pattern_list

    def _get_file_safe(self, remote_path, local_path):
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            res = requests.get(self._pattern_get % remote_path)
            if res.status_code != 200:
                raise RuntimeError('cannot not get %s (response code %d)' % (remote_path, res.status_code))
            tmpfile.write(res.content)
            os.rename(tmpfile.name, local_path)

    def _check_existing_file(self, remote_path, local_path):
        # not optimized for http download yet
        return False

    def stream(self, remote_path, buffer_size=1024):
        res = requests.get(self._pattern_get % remote_path, stream=True)
        if res.status_code != 200:
            raise RuntimeError('cannot not get %s (response code %d)' % (remote_path, res.status_code))

        def generate():
            for chunk in res.iter_content(chunk_size=buffer_size, decode_unicode=None):
                yield chunk

        return generate()

    def push_file(self, local_path, remote_path):
        if self._pattern_push is None:
            raise ValueError('http storage %s can not handle post requests' % self._storage_id)
        with open(local_path, "rb") as f:
            data = f.read()
            res = requests.post(url=self._pattern_push % remote_path,
                                data=data,
                                headers={'Content-Type': 'application/octet-stream'})
            if res.status_code != 200:
                raise RuntimeError('cannot not post %s to %s (response code %d)' % (
                    local_path,
                    remote_path,
                    res.status_code))

    def listdir(self, remote_path, recursive=False):
        if self._pattern_list is None:
            raise ValueError('http storage %s can not handle list request' % self._storage_id)

        res = requests.get(self._pattern_list % remote_path)
        if res.status_code != 200:
            raise RuntimeError('Error when listing remote directory %s (status %d)' % (
                remote_path, res.status_code))
        data = res.json()
        return [os.path.join(remote_path, f["path"]) for f in data]

    def _delete_single(self, remote_path, isdir):
        raise NotImplementedError()

    def rename(self, old_remote_path, new_remote_path):
        raise NotImplementedError()

    def mkdir(self, remote_path):
        return

    def isdir(self, remote_path):
        if remote_path.endswith('/'):
            return True
        return False

    def exists(self, remote_path):
        raise NotImplementedError()

    def _internal_path(self, remote_path):
        return remote_path
