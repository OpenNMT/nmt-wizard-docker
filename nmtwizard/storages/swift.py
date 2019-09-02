"""Definition of `s3` storage class"""

import os
import tempfile
import shutil
import logging

from nmtwizard.storages.generic import Storage
from swiftclient.service import SwiftService, SwiftError, SwiftUploadObject, SwiftCopyObject

logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("swiftclient").setLevel(logging.CRITICAL)

LOGGER = logging.getLogger(__name__)


class SwiftStorage(Storage):
    """Storage on OpenStack swift service."""

    def __init__(self, storage_id, container_name, os_username=None, os_password=None,
                 os_tenant_name=None, os_auth_url=None, transfer_config=None):
        super(SwiftStorage, self).__init__(storage_id)
        opts = transfer_config or {}
        opts["auth_version"] = "2.0"
        if os_username:
            opts["os_username"] = os_username
        if os_password:
            opts["os_password"] = os_password
        if os_tenant_name:
            opts["os_tenant_name"] = os_tenant_name
        if os_auth_url:
            opts["os_auth_url"] = os_auth_url
        self._client = SwiftService(opts)
        self._container = container_name

    def _get_file_safe(self, remote_path, local_path):
        tmpdir = tempfile.mkdtemp()
        results = self._client.download(container=self._container,
                                        objects=[remote_path],
                                        options={"out_directory": tmpdir})
        has_results = False
        for r in results:
            has_results = True
            if not r["success"]:
                raise RuntimeError("Cannot download [%s]: %s" % (remote_path, r["error"]))
        if not has_results:
            raise RuntimeError("Cannot copy download [%s]" % (remote_path, "NO RESULT"))
        shutil.move(os.path.join(tmpdir, remote_path), local_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

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

    def push_file(self, local_path, remote_path):
        (local_dir, basename) = os.path.split(local_path)
        obj = SwiftUploadObject(local_path, object_name=remote_path)
        results = self._client.upload(self._container, [obj])
        has_results = False
        for r in results:
            has_results = True
            if not r["success"]:
                raise RuntimeError("Cannot push file [%s]>[%s]: %s" % (local_path, remote_path, r["error"]))
        if not has_results:
            raise RuntimeError("Cannot push file [%s]>[%s]: %s" % (local_path, remote_path, "NO RESULTS"))

    def stream(self, remote_path, buffer_size=1024):
        def generate():
            tmpdir = tempfile.mkdtemp()
            results = self._client.download(container=self._container,
                                            objects=[remote_path],
                                            options={"out_directory": tmpdir})
            has_results = False
            for r in results:
                has_results = True
                if not r["success"]:
                    raise RuntimeError("Cannot download file [%s]: %s", (remote_path, r["error"]))
            if not has_results:
                raise RuntimeError("Cannot download file [%s]: NO RESULTS", (remote_path))

            with open(os.path.join(tmpdir, remote_path), "rb") as f:
                for chunk in iter(lambda: f.read(buffer_size), b''):
                    yield chunk

            shutil.rmtree(tmpdir, ignore_errors=True)

        return generate()

    def listdir(self, remote_path, recursive=False):
        options = {"prefix": remote_path}
        if not recursive:
            options["delimiter"] = "/"
        list_parts_gen = self._client.list(container=self._container,
                                           options=options)
        lsdir = {}
        for page in list_parts_gen:
            if page["success"]:
                for item in page["listing"]:
                    if "subdir" in item:
                        lsdir[item["subdir"]] = 1
                    else:
                        path = item["name"]
                        lsdir[path] = 0
        return lsdir.keys()

    def mkdir(self, remote_path):
        pass

    def _delete_single(self, remote_path, isdir):
        if not isdir:
            results = self._client.delete(container=self._container, objects=[remote_path])
            has_results = False
            for r in results:
                has_results = True
                if not r["success"]:
                    raise RuntimeError("Cannot delete file [%s]: %s" % (remote_path, r["error"]))
            if not has_results:
                raise RuntimeError("Cannot delete file [%s]: NO RESULT" % (remote_path))

    def rename(self, old_remote_path, new_remote_path):
        listfiles = self.listdir(old_remote_path, True)
        for f in listfiles:
            assert f[:len(old_remote_path)] == old_remote_path, "inconsistent listdir result"
            obj = SwiftCopyObject(f, {"destination": "/%s/%s%s" % (
                                                                   self._container,
                                                                   new_remote_path,
                                                                   f[len(old_remote_path):])})
            results = self._client.copy(self._container, [obj])
            has_results = False
            for r in results:
                has_results = True
                if not r["success"]:
                    raise RuntimeError("Cannot copy file [%s]: %s" % (old_remote_path, r["error"]))
            if not has_results:
                raise RuntimeError("Cannot copy file [%s]: NO RESULT" % (old_remote_path))
            self._delete_single(f, False)

    def exists(self, remote_path):
        result = self._client.list(container=self._container, options={"prefix": remote_path,
                                                                       "delimiter": "/"})
        for page in result:
            if page["success"]:
                for item in page["listing"]:
                    if "subdir" in item:
                        return True
                    if (item["name"] == remote_path or
                            remote_path == '' or
                            remote_path.endswith('/') or
                            item["name"].startswith(remote_path + '/')):
                        return True
        return False

    def isdir(self, remote_path):
        if not remote_path.endswith('/'):
            return self.exists(remote_path+'/')
        return self.exists(remote_path)

    def _internal_path(self, path):
        # OpenStack does not work with paths but keys. This function possibly adapts a
        # path-like representation to a OpenStack key.
        if path.startswith('/'):
            return path[1:]
        return path
