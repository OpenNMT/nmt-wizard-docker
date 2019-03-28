"""Definition of `s3` storage class"""

import os
import boto3
import tempfile

from nmtwizard.storages.generic import Storage

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
        if directory is None:
            directory = self.isdir(remote_path)
        remote_path = _sanitize_path(remote_path)
        if not directory:
            (local_dir, basename) = os.path.split(local_path)
            if os.path.isdir(local_path) or (local_dir != "" and basename == ""):
                local_path = os.path.join(local_dir, os.path.basename(remote_path))
                basename = remote_path
            else:
                local_dir = os.path.dirname(local_path)
            md5_path = os.path.join(local_dir, ".5dm#"+basename+"#md5")
            if local_dir != "" and not os.path.exists(local_dir):
                os.makedirs(local_dir)
            if os.path.exists(local_path) and os.path.exists(md5_path):
                with open(md5_path) as f:
                    md5 = f.read()
                obj = self._bucket.Object(remote_path)
                if obj.e_tag == md5:
                    return
            with tempfile.NamedTemporaryFile() as tmpfile:
                self._bucket.download_file(remote_path, tmpfile.name)
                os.rename(tmpfile.name, local_path)
                obj = self._bucket.Object(remote_path)
                with open(md5_path, "w") as fw:
                    fw.write(obj.e_tag)
        else:
            allfiles = {}
            for root, dirs, files in os.walk(local_path):
                for f in files:
                    allfiles[os.path.join(root, f)] = 1
            objects = list(self._bucket.objects.filter(Prefix=remote_path))
            if not objects:
                raise RuntimeError('%s not found' % remote_path)
            if not os.path.exists(local_path):
                os.mkdir(local_path)
            for obj in objects:
                directories = obj.key.split('/')[1:-1]
                if directories:
                    directory_path = os.path.join(local_path, os.path.join(*directories))
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                path = os.path.join(local_path, os.path.join(*obj.key.split('/')[1:]))
                (local_dir, basename) = os.path.split(path)
                md5_path = os.path.join(local_dir, ".5dm#"+basename+"#md5")
                if path in allfiles:
                    del allfiles[path]
                    if md5_path in allfiles:
                        del allfiles[md5_path]
                self.get(obj.key, path)
            for f in allfiles:
                os.remove(f)

    def push(self, local_path, remote_path):
        remote_path = _sanitize_path(remote_path)
        if os.path.isfile(local_path):
            if remote_path.endswith('/') or self.exists(remote_path+'/'):
                remote_path = os.path.join(remote_path, os.path.basename(local_path))
            self._bucket.upload_file(local_path, remote_path)
        else:
            for root, _, files in os.walk(local_path):
                for filename in files:
                    path = os.path.join(root, filename)
                    relative_path = os.path.relpath(path, local_path)
                    s3_path = os.path.join(remote_path, relative_path)
                    self._bucket.upload_file(path, s3_path)

    def stream(self, remote_path, buffer_size=1024):
        remote_path = _sanitize_path(remote_path)
        body = self._s3.Object(self._bucket_name, remote_path).get()['Body']

        def generate():
            for chunk in iter(lambda: body.read(buffer_size), b''):
                yield chunk

        return generate()

    def listdir(self, remote_path, recursive=False):
        remote_path = _sanitize_path(remote_path)
        objects = list(self._bucket.objects.filter(Prefix=remote_path))
        lsdir = {}
        for obj in objects:
            path = obj.key
            if remote_path == '' or \
               path == remote_path or remote_path.endswith('/') or path.startswith(remote_path + '/'):
                p = path.find('/', len(remote_path)+1)
                if not recursive and p != -1:
                    path = path[0:p+1]
                    lsdir[path] = 1
                else:
                    lsdir[path] = 0
            else:
                print("skipping %s (in %s)" % (path, remote_path))
        return lsdir.keys()

    def delete(self, remote_path, recursive=False):
        remote_path = _sanitize_path(remote_path)
        lsdir = self.listdir(remote_path, recursive)
        if recursive:
            if remote_path in lsdir or not lsdir:
                raise ValueError("%s is not a directory" % remote_path)
        else:
            if remote_path not in lsdir:
                raise ValueError("%s is not a file" % remote_path)

        for key in lsdir:
            self._s3.meta.client.delete_object(Bucket=self._bucket_name, Key=key)

    def rename(self, old_remote_path, new_remote_path):
        old_remote_path = _sanitize_path(old_remote_path)
        new_remote_path = _sanitize_path(new_remote_path)
        for obj in self._bucket.objects.filter(Prefix=old_remote_path):
            src_key = obj.key
            if not src_key.endswith('/'):
                copy_source = self._bucket_name + '/' + src_key
                if src_key == old_remote_path:
                    # it is a file that we are copying
                    dest_file_key = new_remote_path
                else:
                    filename = src_key.split('/')[-1]
                    dest_file_key = new_remote_path + '/' + filename
                self._s3.Object(self._bucket_name, dest_file_key).copy_from(CopySource=copy_source)
            self._s3.Object(self._bucket_name, src_key).delete()

    def exists(self, remote_path):
        remote_path = _sanitize_path(remote_path)
        result = self._bucket.objects.filter(Prefix=remote_path)
        for obj in result:
            if (obj.key == remote_path or
                    remote_path == '' or
                    remote_path.endswith('/') or
                    obj.key.startswith(remote_path + '/')):
                return True
        return False

def _sanitize_path(path):
    # S3 does not work with paths but keys. This function possibly adapts a
    # path-like representation to a S3 key.
    if path.startswith('/'):
        return path[1:]
    return path
