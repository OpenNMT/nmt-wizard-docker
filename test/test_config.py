import copy
import pytest
import jsonschema

from nmtwizard import config

def test_key_override():
    a = {"a": {"b": 42, "c": "d"}, "e": "f"}
    b = {"a": None}
    c = config.merge_config(a, b)
    assert c == {"a": None, "e": "f"}

def test_key_replace():
    a = {"a": {"b": 42, "c": "d"}, "e": "f"}
    b = {"e": {"x": "y"}}
    c = config.merge_config(a, b)
    assert c == {"a": {"b": 42, "c": "d"}, "e": {"x": "y"}}


_base_schema = {
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
}

_test_inference_options = _base_schema.copy()
_test_inference_options.update({
    "options": [
        {
            "option_path": "bpreprocess/politeness",
            "config_path": "bpreprocess/classifiers/0/value"
        },
        {
            "option_path": "bpreprocess/domain",
            "config_path": "bpreprocess/classifiers/1/value",
        }
    ]
})

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

_test_inference_options_v2 = _base_schema.copy()
_test_inference_options_v2.update({
    "options": [
        {
            "option_path": "bpreprocess/politeness",
            "config_path": "preprocess/my-politeness-op/value"
        },
        {
            "option_path": "bpreprocess/domain",
            "config_path": "preprocess/my-domain-op/value",
        }
    ]
})

_test_config_v2 = {
    "preprocess": [
        {
            "op": "domain-classifier",
            "name": "my-domain-op",
        },
        {
            "op": "politeness-classifier",
            "name": "my-politeness-op",
        },
    ],
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
    assert (config.index_config(cfg, "bpreprocess/classifiers/1/value", index_structure=False)
            == (cfg["bpreprocess"]["classifiers"][1], "value"))
    with pytest.raises(ValueError, match="Invalid path"):
        config.index_config(cfg, "bpreprocess/classifiers/1/value")

def test_inference_options_index_config_v2():
    cfg = _test_config_v2
    assert config.index_config(cfg, "preprocess/my-domain-op") == cfg["preprocess"][0]

def test_inference_options_validation():
    schema = config.validate_inference_options(_test_inference_options, _test_config)
    assert isinstance(schema, dict)

def test_inference_options_invalid_shema():
    opt = copy.deepcopy(_test_inference_options)
    opt["json_schema"]["type"] = "objects"
    with pytest.raises(jsonschema.SchemaError):
        config.validate_inference_options(opt, _test_config)

def test_read_options():
    cfg = copy.deepcopy(_test_config)
    cfg['inference_options'] = copy.deepcopy(_test_inference_options)

    with pytest.raises(ValueError):
        options = {"bpreprocess": {"domain": "Technology"}}
        config.read_options(cfg, options)

    options = {"bpreprocess": {"domain": "IT"}}
    assert config.read_options(cfg, options) == {
        "bpreprocess": {
            "classifiers": [
                {
                    "name": "politeness"
                },
                {
                    "name": "domain",
                    "value": "IT",
                }
            ]
        }
    }

def test_read_options_v2():
    cfg = copy.deepcopy(_test_config_v2)
    cfg['inference_options'] = copy.deepcopy(_test_inference_options_v2)
    options = {"bpreprocess": {"domain": "IT"}}
    assert config.read_options(cfg, options) == {"my-domain-op": {"value": "IT"}}

def test_build_override():
    c = {"a": {"b": {"c": 42}, "d": [{"e": 43}, {"f": 44}]}}
    assert config.build_override(c, "a/z", 45) == {"a": {"z": 45}}
    assert config.build_override(c, "a/d/1/g", 45) == {"a": {"d": [{"e": 43}, {"f": 44, "g": 45}]}}
