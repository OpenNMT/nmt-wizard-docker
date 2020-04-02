# OpenNMT-tf framework

This framework is based on [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf/).

The `options` block should define the [model](https://opennmt.net/OpenNMT-tf/model.html) and the [run configuration](https://opennmt.net/OpenNMT-tf/configuration.html) (or `auto_config` if it applied).

Example with `auto_config`:

```json
{
    "options": {
        "model_type": "Transformer",
        "auto_config": true
    }
}
```

Example using a custom model and configuration:

```json
    "options": {
        "model": "/path/to/model/definition.py",
        "config": {
            "params": {
                "optimizer": "SGD",
                "learning_rate": 0.1,
                "beam_width": 5
            },
            "train": {
                "batch_size": 64,
                "length_bucket_width": 1,
                "maximum_features_length": 50,
                "maximum_labels_length": 50
            },
            "infer": {
                "batch_size": 32
            }
        }
    }
}
```
