# REST translation API

### `GET /status`

**Output:**

Status 200 if the model is ready to run translations.

### `POST /translate`

**Input (minimum required):**

```json
{
    "src": [
        {"text": "Source sentence 1"},
        {"text": "Source sentence 2"}
    ]
}
```

**Input (with optional fields):**

```json
{
    "options": {
        "max_batch_size": 32,
        "config": {}
    },
    "src": [
        {"text": "Source sentence 1", "config": {}, "options": {}},
        {"text": "Source sentence 2", "config": {}, "options": {}}
    ]
}
```

* The `config` fields define request-specific and sentence-specific overrides to the global JSON configuration file.
* The `options` fields (in `src`) define [inference options](docs/inference_options.md) to be mapped to the global configuration file.

**Output:**

```json
{
    "tgt": [
        [{
            "text": "Phrase cible 1",
            "score": -2.16,
            "align": [
                {"tgt": [ {"range": [0, 5], "id": 0} ],
                 "src": [ {"range": [9, 14], "id": 1} ]},
                {"tgt": [ {"range": [7, 11], "id": 1} ],
                 "src": [ {"range": [0, 5], "id": 0} ]},
                {"tgt": [ {"range": [13, 13], "id": 2} ],
                 "src": [ {"range": [16, 16], "id": 2} ]}
             ]
        }],
        [{
            "text": "Phrase cible 2",
            "score": -2.17,
            "align": [
                {"tgt": [ {"range": [0, 5], "id": 0} ],
                 "src": [ {"range": [9, 14], "id": 1} ]},
                {"tgt": [ {"range": [7, 11], "id": 1} ],
                 "src": [ {"range": [0, 5], "id": 0} ]},
                {"tgt": [ {"range": [13, 13], "id": 2} ],
                 "src": [ {"range": [16, 16], "id": 2} ]}
             ]
        }]
    ]
}
```

The `tgt` field is a list the size of the batch where each entry is a list listing all hypotheses (the N best list) ordered from best to worst (higher score means better prediction).

Note that the `score` and `align` fields might not be set by all frameworks and model types.

**Errors:**

* **HTTP 400**
  * The input data is missing.
  * The input data is not a JSON object.
  * The input data does not contain the `src` field.
  * The `src` field is not a list.
  * The inference option is unexpected or invalid
* **HTTP 500**
  * Internal server exception.
* **HTTP 503**
  * The backend service is unavailable.
* **HTTP 504**
  * The translation request timed out.

### `POST /unload_model`

Unload the model from the reserved resource. In its simplest form, this route will terminate the backend translation service.

### `POST /reload_model`

Reload the model on the reserved resource. In its simplest form, this route will terminate the backend translation service if it is still running and start a new instance.
