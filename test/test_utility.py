import os

import pytest

from nmtwizard import utility, storage


def test_resolve_env():
    config = {
        "a": "${A_DIR}/a",
        "b": ["${B_DIR}/b", "${A_TRAIN_DIR}/a"]
    }
    os.environ["A_DIR"] = "foo"
    os.environ["B_DIR"] = "bar"
    config = utility.resolve_environment_variables(config)
    assert config["a"] == "foo/a"
    assert config["b"] == ["bar/b", "foo/a"]
    del os.environ["A_DIR"]
    del os.environ["B_DIR"]

def test_resolve_env_no_training():
    config = {
        "a": "${A_DIR}/a",
        "b": "${A_TRAIN_DIR}/a"
    }
    os.environ["A_DIR"] = "foo"
    config = utility.resolve_environment_variables(config, training=False)
    assert config["a"] == "foo/a"
    assert config["b"] == "${A_TRAIN_DIR}/a"

def test_resolve_remote_files(tmpdir):
    tmpdir.join("remote").join("dir").join("a.txt").write("toto", ensure=True)
    tmpdir.join("local").ensure_dir()
    storage_config = {
        "tmp": {"type": "local", "basedir": str(tmpdir)},
        "tmp2": {"type": "local", "basedir": str(tmpdir.join("remote"))}
    }
    client = storage.StorageClient(config=storage_config)
    config = {
        "a": "/home/ubuntu/a.txt",
        "b": "non_storage:b.txt",
        "c": "tmp:remote/dir/a.txt",
        "d": "tmp2:/dir/a.txt",
        "e": True
    }
    config = utility.resolve_remote_files(config, str(tmpdir.join("local")), client)
    c_path = tmpdir.join("local").join("tmp/remote/dir/a.txt")
    d_path = tmpdir.join("local").join("tmp2/dir/a.txt")
    assert config["a"] == "/home/ubuntu/a.txt"
    assert config["b"] == "non_storage:b.txt"
    assert config["c"] == str(c_path)
    assert config["d"] == str(d_path)
    assert c_path.check(file=1)
    assert d_path.check(file=1)
