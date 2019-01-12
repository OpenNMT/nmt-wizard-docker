import os
import requests

from generic import Storage

class HTTPStorage(Storage):
    """Simple http file-only storage."""

    def __init__(self, storage_id, pattern_get, pattern_push=None, pattern_list=None):
        super(HTTPStorage, self).__init__(storage_id)
        self._pattern_get = pattern_get
        self._pattern_push = pattern_push
        self._pattern_list = pattern_list

    def get(self, remote_path, local_path, directory=False):
        if not directory:
            res = requests.get(self._pattern_get % remote_path)
            if res.status_code != 200:
                raise RuntimeError('cannot not get %s (response code %d)' % (remote_path, res.status_code))
            if os.path.isdir(local_path):
                local_path = os.path.join(local_path, os.path.basename(remote_path))
            elif not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))
            with open(local_path, "wb") as f:
                f.write(res.content)
        elif self._pattern_list is None:
            raise ValueError('http storage %s can not handle directories' % self._storage_id)
        else:
            res = requests.get(self._pattern_list % remote_path)
            if res.status_code != 200:
                raise RuntimeError('Error when listing remote directory %s (status %d)' % (
                    remote_path, res.status_code))
            data = res.json()
            for f in data:
                path = f["path"]
                self.get(os.path.join(remote_path, path), os.path.join(local_path, path))

    def stream(self, remote_path, buffer_size=1024):
        res = requests.get(self._pattern_get % remote_path, stream=True)
        if res.status_code != 200:
            raise RuntimeError('cannot not get %s (response code %d)' % (remote_path, res.status_code))

        def generate():
            for chunk in res.iter_content(chunk_size=buffer_size, decode_unicode=None):
                yield chunk

        return generate()

    def push(self, local_path, remote_path):
        if self._pattern_push is None:
            raise ValueError('http storage %s can not handle post requests' % self._storage_id)
        if os.path.isfile(local_path):
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
        else:
            raise NotImplementedError('http storage can not handle directories')
