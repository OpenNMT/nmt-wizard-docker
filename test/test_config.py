import pytest

from nmtwizard.framework import merge_config

def test_key_override():
    a = {"a": {"b": 42, "c": "d"}, "e": "f"}
    b = {"a": None}
    c = merge_config(a, b)
    assert c == {"a": None, "e": "f"}

def test_key_replace():
    a = {"a": {"b": 42, "c": "d"}, "e": "f"}
    b = {"e": {"x": "y"}}
    c = merge_config(a, b)
    assert c == {"a": {"b": 42, "c": "d"}, "e": {"x": "y"}}
