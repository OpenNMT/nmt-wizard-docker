import os
import requests_mock

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
        assert local_dir.join("checkpoint").join("model.bin").read() == b"model"
        assert local_dir.join("config.json").read() == b"config"
