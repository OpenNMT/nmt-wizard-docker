import os

import pytest

from nmtwizard import utility


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
