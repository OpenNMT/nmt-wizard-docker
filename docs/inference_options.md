## Inference options

Custom frameworks can declare options they accept during inference. Suppose a translation model was trained to support different domains based on an input tag: the inference options mechanism can be used to pass this information at serving time.

1\. The mechanism expects 2 components to be configured in the global model configuration:

* a [JSON Schema](https://json-schema.org/) describing the accepted options and the value constraints
* a mapping between inference options and configuration fields using a path-like representation

```json
{
    "inference_options": {
        "json_schema": {
            "type": "object",
            "title": "Domain",
            "description": "Domain to use for the translation",
            "properties": {
                "domain": {
                    "type": "string",
                    "title": "Domain",
                    "enum": ["IT", "News", "Medical"]
                }
            }
        },
        "options": [
            {
                "option_path": "domain",
                "config_path": "preprocess/domain/value"
            }
        ]
    },
    "preprocess": {
        "domain": {
            "some fields used during training": {}
        }
    },
    "other_section": {...}
}
```

2\. When releasing the model, the `inference_options` configuration will be validated for correctness. Also to simplify the integration with external tools, the JSON Schema will be exported to the file `options.json` in the model directory.

3\. During inference, the options can be passed in the request:

```json
{
    "src": [{
        "text": "Source sentence 1",
        "options": {"domain": "IT"}
    }]
}
```

They will be validated against the schema and values will be injected in the global configuration:

```json
{
    "inference_options": {...},
    "preprocess": {
        "domain": {
            "value": "IT",
            "some fields used during training": {}
        }
    },
    "other_section": {...}
}
```

It is up to the custom preprocessing module to correctly use the injected `value` field during inference. If the field has not been marked as required in the JSON Schema, the module should also work without this runtime value (e.g. it should apply a default value).
