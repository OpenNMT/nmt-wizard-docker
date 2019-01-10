"""Client to abstract storage location: local, S3, SSH, etc."""

import abc
import os
import requests
import shutil
import six
import paramiko
import scp
import boto3

from collections import Counter

from nmtwizard.logger import get_logger

logger = get_logger(__name__)


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
                                       config['password'],
                                       port=config.get('port', 22))
            elif config['type'] == 'http':
                client = HTTPStorage(storage_id,
                                     config['get_pattern'],
                                     pattern_push=config.get('post_pattern'),
                                     pattern_list=config.get('list_pattern'))
            else:
                raise ValueError('unsupported storage type %s for %s' % (config['type'], storage_id))
        else:
            client = LocalStorage()

        return client, path

    def join(self, path, *paths):
        """Joins the paths according to the storage implementation."""
        client, rel_path = self._get_storage(path)
        if rel_path == path:
            return client.join(path, *paths)
        else:
            prefix, _ = path.split(':')
            return '%s:%s' % (prefix, client.join(rel_path, *paths))  # Only join the actual path.

    def split(self, path):
        """Splits the path according to the storage implementation."""
        client, path = self._get_storage(path)
        return client.split(path)

    # Simple wrappers around get().
    def get_file(self, remote_path, local_path, storage_id=None):
        return self.get(remote_path, local_path, directory=False, storage_id=storage_id)

    def get_directory(self, remote_path, local_path, storage_id=None):
        return self.get(remote_path, local_path, directory=True, storage_id=storage_id)

    def get(self,
            remote_path,
            local_path,
            directory=False,
            storage_id=None,
            check_integrity_fn=None):
        if directory and os.path.isdir(local_path):
            logger.warning('Directory %s already exists', local_path)
        elif not directory and os.path.isfile(local_path):
            logger.warning('File %s already exists', local_path)
        else:
            tmp_path = os.path.join(self._tmp_dir, os.path.basename(local_path))
            logger.info('Downloading %s to %s through tmp %s', remote_path, local_path, tmp_path)
            client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
            client.get(remote_path, tmp_path, directory=directory)
            if not os.path.exists(tmp_path):
                raise RuntimeError('download failed: %s not found' % tmp_path)
            if check_integrity_fn is not None and not check_integrity_fn(tmp_path):
                raise RuntimeError('integrity check failed on %s' % tmp_path)
            # in meantime, the file might have been copied
            if os.path.exists(local_path):
                logger.warning('File/Directory created while copying - taking copy')
            else:
                check_integrity_fn = None  # No need to check again.
                shutil.move(tmp_path, local_path)
        if check_integrity_fn is not None and not check_integrity_fn(local_path):
            raise RuntimeError('integrity check failed on %s' % local_path)

    def push(self, local_path, remote_path, storage_id=None):
        if not os.path.exists(local_path):
            raise RuntimeError('%s not found' % local_path)
        if local_path == remote_path:
            return
        logger.info('Uploading %s to %s', local_path, remote_path)
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        client.push(local_path, remote_path)

    def ls(self, remote_path, storage_id=None):
        client, remote_path = self._get_storage(remote_path, storage_id=storage_id)
        return client.ls(remote_path)


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

    def ls(self, remote_path):
        """Return a dictionary with all files in the remote directory
        """
        raise NotImplementedError()


class LocalStorage(Storage):
    """Storage using the local filesystem."""

    def __init__(self):
        super(LocalStorage, self).__init__("local")

    def get(self, remote_path, local_path, directory=False):
        if directory:
            shutil.copytree(remote_path, local_path)
        else:
            shutil.copy(remote_path, local_path)

    def stream(self, remote_path, buffer_size=1024):
        with open(remote_path, "rb") as f:
            def generate():
                for chunk in iter(lambda: body.read(buffer_size), b''):
                    yield chunk
            return generate()

    def push(self, local_path, remote_path):
        self.get(local_path, remote_path, directory=os.path.isdir(local_path))


class RemoteStorage(Storage):
    """Storage on a remote SSH server."""

    def __init__(self, storage_id, server, user, password, port=22):
        super(RemoteStorage, self).__init__(storage_id)
        self._server = server
        self._user = user
        self._password = password
        self._port = port

    def _connect(self):
        ssh_client = paramiko.SSHClient()
        ssh_client.load_system_host_keys()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(self._server, self._port, self._user, self._password)
        return scp.SCPClient(ssh_client.get_transport())

    def get(self, remote_path, local_path, directory=False):
        client = self._connect()
        client.get(remote_path, local_path, recursive=directory)
        client.close()

    def stream(self, remote_path, buffer_size=1024):
        client = self._connect()
        channel = client._open()
        channel.settimeout(client.socket_timeout)
        channel.exec_command("scp -f " + client.sanitize(scp.asbytes(remote_path)))
        while not channel.closed:
            # wait for command as long as we're open
            channel.sendall('\x00')
            msg = channel.recv(1024)
            if not msg:  # chan closed while recving
                break
            assert msg[-1:] == b'\n'
            msg = msg[:-1]
            code = msg[0:1]
            # recv file
            if code == "C":
                cmd = msg[1:]
                parts = cmd.strip().split(b' ', 2)
                mode = int(parts[0], 8)
                size = int(parts[1])
                remote_file = parts[2]

                channel.send(b'\x00')
                try:
                    def generate():
                        buff_size = buffer_size
                        pos = 0
                        while pos < size:
                            # we have to make sure we don't read the final byte
                            if size - pos <= buff_size:
                                buff_size = size - pos
                            s = channel.recv(buff_size)
                            pos += len(s)
                            yield s
                        msg = channel.recv(512)
                        if msg and msg[0:1] != b'\x00':
                            raise scp.SCPException(scp.asunicode(msg[1:]))
                        client.close()
                    return generate()
                except SocketTimeout:
                    channel.close()
                    raise scp.SCPException('Error receiving, socket.timeout')

    def push(self, local_path, remote_path):
        client = self._connect()
        client.put(local_path, remote_path, recursive=os.path.isdir(local_path))
        client.close()


class S3Storage(Storage):
    """Storage on Amazon S3."""

    def __init__(self, storage_id, bucket_name, access_key_id=None, secret_access_key=None, region_name=None):
        super(S3Storage, self).__init__(storage_id)
        if access_key_id is None and secret_access_key is None and region_name is None:
            session = boto3
        else:
            session = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name)
        self._s3 = session.resource('s3')
        self._bucket_name = bucket_name
        self._bucket = self._s3.Bucket(bucket_name)

    def get(self, remote_path, local_path, directory=False):
        if not directory:
            if os.path.isdir(local_path):
                local_path = os.path.join(local_path, os.path.basename(remote_path))
            self._bucket.download_file(remote_path, local_path)
        else:
            objects = list(self._bucket.objects.filter(Prefix=remote_path))
            if not objects:
                raise RuntimeError('%s not found' % remote_path)
            os.mkdir(local_path)
            for obj in objects:
                directories = obj.key.split('/')[1:-1]
                if directories:
                    directory_path = os.path.join(local_path, os.path.join(*directories))
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                path = os.path.join(local_path, os.path.join(*obj.key.split('/')[1:]))
                self._bucket.download_file(obj.key, path)

    def push(self, local_path, remote_path):
        if os.path.isfile(local_path):
            self._bucket.upload_file(local_path, remote_path)
        else:
            for root, _, files in os.walk(local_path):
                for filename in files:
                    path = os.path.join(root, filename)
                    relative_path = os.path.relpath(path, local_path)
                    s3_path = os.path.join(remote_path, relative_path)
                    self._bucket.upload_file(path, s3_path)

    def stream(self, remote_path, buffer_size=1024):
        body = self._s3.Object(self._bucket_name, remote_path).get()['Body']

        def generate():
            for chunk in iter(lambda: body.read(buffer_size), b''):
                yield chunk

        return generate()

    def ls(self, remote_path):
        objects = list(self._bucket.objects.filter(
                             Prefix=remote_path
                        ))
        firstlevel = Counter()
        for obj in objects:
            path = obj.key
            p = path.find('/', len(remote_path)+1)
            if p != -1:
                path = path[0:p]
                firstlevel[path] += 1
            else:
                firstlevel[path] = 0
        return dict((k, v) for k, v in six.iteritems(firstlevel))


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
