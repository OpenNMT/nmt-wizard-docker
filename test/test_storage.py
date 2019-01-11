import os
import requests_mock
import pytest

from nmtwizard import storage


def test_http_storage_get_dir(tmpdir):
    with requests_mock.Mocker() as m:
        m.register_uri(
            "GET", "http://launcher/model/listfile/model0",
            json=[
                {"path": "checkpoint/model.bin", "size": 42},
                {"path": "config.json", "size": 10}])
        m.register_uri(
            "GET", "http://launcher/model/getfile/model0/checkpoint/model.bin", content=b"model")
        m.register_uri(
            "GET", "http://launcher/model/getfile/model0/config.json", content=b"config")
        http = storage.HTTPStorage(
            "0",
            "http://launcher/model/getfile/%s",
            pattern_list="http://launcher/model/listfile/%s")

        local_dir = tmpdir.join("model0")
        http.get("model0", str(local_dir), directory=True)
        assert local_dir.check()
        assert local_dir.join("checkpoint").join("model.bin").read() == "model"
        assert local_dir.join("config.json").read() == "config"


def test_http_stream(tmpdir):
    http = storage.HTTPStorage("0", "http://www.ovh.net/files/%s")
    size = 0
    nchunk = 0
    for chunk in http.stream("1Mio.dat"):
        size += len(chunk)
        nchunk += 1
    assert size == 1024*1024 and nchunk == 1024


def test_storage_manager(tmpdir):
    config = {
                "s3_models": {
                    "description": "model storage on S3",
                    "type": "s3",
                    "bucket": "my-model-storage",
                    "aws_credentials": {
                        "access_key_id": "AAAAAAAAAAAAAAAAAAAA",
                        "secret_access_key": "abcdefghijklmnopqrstuvwxyz0123456789ABCD",
                        "region_name": "us-east-2"
                    },
                    "default_ms": True
                },
                "s3_test": {
                    "description": "some test files",
                    "type": "s3",
                    "bucket": "my-testfiles-storage",
                    "aws_credentials": {
                        "access_key_id": "AAAAAAAAAAAAAAAAAAAA",
                        "secret_access_key": "abcdefghijklmnopqrstuvwxyz0123456789ABCD",
                        "region_name": "us-east-2"
                    }
                },
                "launcher": {
                    "description": "launcher file storage",
                    "type": "http",
                    "get_pattern": "hereget/%s",
                    "post_pattern": "herepost/%s"
                }
    }
    storages = storage.StorageClient(config=config, tmp_dir=str(tmpdir))
    s3_models_storage, path = storages._get_storage("s3_models:pathdir/mysupermodel")
    assert isinstance(s3_models_storage, storage.S3Storage)
    assert path == "pathdir/mysupermodel"
    assert s3_models_storage._storage_id == "s3_models"

    s3_models_storage, path = storages._get_storage("pathdir/mysupermodel", "s3_models")
    assert isinstance(s3_models_storage, storage.S3Storage)

    local_storage, path = storages._get_storage("/pathdir/mysupermodel")
    assert isinstance(local_storage, storage.LocalStorage)
    assert local_storage._storage_id == "local"

    http_storage, path = storages._get_storage("launcher:/hereget/mysupermodel")
    assert isinstance(http_storage, storage.HTTPStorage)
    with pytest.raises(ValueError):
        storages._get_storage("unknown:/hereget/mysupermodel")


def test_local_storage(tmpdir):
    storages = storage.StorageClient(tmp_dir=str(tmpdir))
    corpus_dir = str(pytest.config.rootdir / "corpus")
    storages.get(os.path.join(corpus_dir, "train", "europarl-v7.de-en.10K.tok.de"), str(tmpdir.join("localcopy")))
    assert os.path.isfile(str(tmpdir.join("localcopy")))

    storages.delete(str(tmpdir.join("localcopy")))
    assert not os.path.exists(str(tmpdir.join("localcopy")))

    # cannot transfer directory if not in remote mode
    with pytest.raises(Exception):
        storages.get(corpus_dir, str(tmpdir.join("localdir")))

    storages.get(corpus_dir, str(tmpdir.join("localdir")), directory=True)
    assert os.path.isfile(str(tmpdir.join("localdir", "train", "europarl-v7.de-en.10K.tok.de")))

    with pytest.raises(ValueError):
        storages.delete(str(tmpdir.join("localdir")))
    storages.delete(str(tmpdir.join("localdir")), directory=True)
    assert not os.path.exists(str(tmpdir.join("localdir")))


