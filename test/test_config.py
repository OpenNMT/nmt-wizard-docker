import copy
import pytest
import jsonschema

from nmtwizard.framework import merge_config
from nmtwizard import config

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


_test_inference_options = {
    "json_schema": {
        "title": "Translation options",
        "description": "Translation options",
        "type": "object",
        "required": [
            "bpreprocess"
        ],
        "properties": {
            "bpreprocess": {
                "type": "object",
                "title": "User friendly name",
                "properties": {
                    "politeness": {
                        "type": "string",
                        "title": "Politeness Mode",
                        "default": "neutral",
                        "enum": ["formal", "informal", "neutral"]
                    },
                    "domain": {
                        "type": "string",
                        "title": "Domain",
                        "enum": ["IT", "News"]
                    }
                }
            }
        }
    },
    "options": [
        {
            "option_path": "bpreprocess/politeness",
            "config_path": "bpreprocess/classifiers/0"
        },
        {
            "option_path": "bpreprocess/domain",
            "config_path": "bpreprocess/classifiers/1",
        }
    ]
}

_test_config = {
    "bpreprocess": {
        "classifiers": [
            {
                "name": "politeness"
            },
            {
                "name": "domain"
            }
        ]
    }
}

def test_inference_options_index_schema():
    schema = _test_inference_options["json_schema"]
    politeness_schema = schema["properties"]["bpreprocess"]["properties"]["politeness"]
    assert config.index_schema(schema, "bpreprocess/politeness") == politeness_schema
    with pytest.raises(ValueError, match="Invalid path"):
        config.index_schema(schema, "bpreprocess/domains")

def test_inference_options_index_config():
    cfg = _test_config
    assert (config.index_config(cfg, "bpreprocess/classifiers/1")
            == cfg["bpreprocess"]["classifiers"][1])
    with pytest.raises(ValueError, match="Invalid path"):
        config.index_config(cfg, "bpreprocess/annotate")

def test_inference_options_validation():
    schema = config.validate_inference_options(_test_inference_options, _test_config)
    assert isinstance(schema, dict)

def test_inference_options_invalid_shema():
    opt = copy.deepcopy(_test_inference_options)
    opt["json_schema"]["type"] = "objects"
    with pytest.raises(jsonschema.SchemaError):
        config.validate_inference_options(opt, _test_config)

def test_options_to_config():
    cfg = copy.deepcopy(_test_config)
    cfg['inference_options'] = copy.deepcopy(_test_inference_options)

    with pytest.raises(ValueError):
        options = {"bpreprocess": {"domain": "Technology"}}
        config.update_config_with_options(cfg, options)

    options = {"bpreprocess": {"domain": "IT"}}
    config.update_config_with_options(cfg, options)
    assert cfg["bpreprocess"]["classifiers"][1]["value"] == "IT"
