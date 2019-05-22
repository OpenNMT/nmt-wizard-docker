"""Definition of `s3` storage class"""

import os
import boto3
import tempfile
import shutil
import datetime
import calendar

from nmtwizard.storages.generic import Storage

from nmtwizard.logger import get_logger

LOGGER = get_logger(__name__)


class S3Storage(Storage):
    """Storage on Amazon S3."""

    def __init__(self, storage_id, bucket_name, access_key_id=None, secret_access_key=None,
                 region_name=None, assume_role=None, transfer_config=None):
        super(S3Storage, self).__init__(storage_id)
        if assume_role is not None:
            session_main = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name)
            client_sts = boto3.client('sts')
            response_assumeRole = client.assume_role(RoleArn=assume_role)
            session = Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                              aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                              aws_session_token=response['Credentials']['SessionToken'])
        else:
            session = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key)
        self._s3 = session.resource('s3')
        self._bucket_name = bucket_name
        self._bucket = self._s3.Bucket(bucket_name)
        if transfer_config is not None:
            self._transfer_config = boto3.s3.transfer.TransferConfig(**transfer_config)
        else:
            self._transfer_config = None

    def _get_file_safe(self, remote_path, local_path):
        (local_dir, basename) = os.path.split(local_path)
        md5_path = os.path.join(local_dir, ".5dm#"+basename+"#md5")
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            self._bucket.download_file(remote_path, tmpfile.name, Config=self._transfer_config)
            shutil.move(tmpfile.name, local_path)
            obj = self._bucket.Object(remote_path)
            with open(md5_path, "w") as fw:
                fw.write(obj.e_tag)

    def _check_existing_file(self, remote_path, local_path):
        (local_dir, basename) = os.path.split(local_path)
        md5_path = os.path.join(local_dir, ".5dm#"+basename+"#md5")
        if os.path.exists(local_path) and os.path.exists(md5_path):
            with open(md5_path) as f:
                md5 = f.read()
            obj = self._bucket.Object(remote_path)
            if obj.e_tag == md5:
                return True
            LOGGER.debug('checksum has changed for file %s (%s/%s)', local_path, md5, obj.e_tag)
        else:
            LOGGER.debug('Cannot find %s or %s', local_path, md5_path)
        return False

    def _get_checksum_file(self, local_path):
        """return checksum sum used by storage or None
        """
        (local_dir, basename) = os.path.split(local_path)
        return os.path.join(local_dir, ".5dm#"+basename+"#md5")

    def push_file(self, local_path, remote_path):
        (local_dir, basename) = os.path.split(local_path)
        md5_path = os.path.join(local_dir, ".5dm#"+basename+"#md5")
        self._bucket.upload_file(local_path, remote_path, Config=self._transfer_config)
        obj = self._bucket.Object(remote_path)
        with open(md5_path, "w") as fw:
            fw.write(obj.e_tag)

    def stream(self, remote_path, buffer_size=1024):
        body = self._s3.Object(self._bucket_name, remote_path).get()['Body']

        def generate():
            for chunk in iter(lambda: body.read(buffer_size), b''):
                yield chunk

        return generate()

    def listdir(self, remote_path, recursive=False):
        lsdir = {}
        if not recursive:
            client = self._s3.meta.client
            paginator = client.get_paginator('list_objects')
            result = paginator.paginate(Bucket=self._bucket_name, Prefix=remote_path, Delimiter='/')
            for prefix in result.search('CommonPrefixes'):
                if prefix:
                    lsdir[prefix.get('Prefix')] = prefix.get('Prefix')
            for content in result.search('Contents'):
                if content:
                    lsdir[content['Key']] = {'Key': content['Key'],
                                             'LastModified': calendar.timegm(content.get('LastModified').timetuple()),
                                             'Size': content.get('Size')}
        else:
            objects = list(self._bucket.objects.filter(Prefix=remote_path))
            for obj in objects:
                path = obj.key
                if remote_path == '' or \
                   path == remote_path or remote_path.endswith('/') or path.startswith(remote_path + '/'):
                    lsdir[path] = {'Key': obj.key,
                                   'LastModified': calendar.timegm(obj.last_modified.timetuple()),
                                   'Size': obj.size}
        return lsdir.values()

    def mkdir(self, remote_path):
        pass

    def _delete_single(self, remote_path, isdir):
        if not isdir:
            self._s3.meta.client.delete_object(Bucket=self._bucket_name, Key=remote_path)

    def rename(self, old_remote_path, new_remote_path):
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
        result = self._bucket.objects.filter(Prefix=remote_path)
        for obj in result:
            if (obj.key == remote_path or
                    remote_path == '' or
                    remote_path.endswith('/') or
                    obj.key.startswith(remote_path + '/')):
                return True
        return False

    def isdir(self, remote_path):
        if not remote_path.endswith('/'):
            return self.exists(remote_path+'/')
        return self.exists(remote_path)

    def _internal_path(self, path):
        # S3 does not work with paths but keys. This function possibly adapts a
        # path-like representation to a S3 key.
        if path.startswith('/'):
            return path[1:]
        return path
