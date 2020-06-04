import pytest
import six
import json
import os


def pytest_generate_tests(metafunc):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "conftest.json")) as f:
        config = json.load(f)

    if 'storages' in config:
        if 'storage_id' in metafunc.fixturenames:
                metafunc.parametrize("storage_id", config["storages"].keys())

        if 'storages' in metafunc.fixturenames:
                metafunc.parametrize("storages", [config["storages"]])
