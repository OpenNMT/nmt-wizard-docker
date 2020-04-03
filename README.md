# nmt-wizard-docker

The goal of this project is to encapsulate MT frameworks in Docker containers and expose a single interface for preprocessing, training, translating, and serving models. 

The [available Docker images](https://hub.docker.com/u/nmtwizard) extend the original frameworks with the following features:

* Data weighting and sampling from raw training files.
* Data and models synchronization from remote storages such as Amazon S3, Swift, or any server via SSH.
* Metadata on model history such as parent model, training data that was used, training time, etc.
* Regular HTTP request to an URL to declare running status.

It supports [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/) and [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf/) training frameworks, and provides translate-only frameworks using online translation API from DeepL, Google, Baidu, and others.

## Usage

We recommend using the Docker images that are available on [Docker Hub](https://hub.docker.com/u/nmtwizard).

The Docker image entrypoint is a Python script that exposes the same command line interface for all frameworks. For example, run the commands below to download the latest OpenNMT-tf image and list the available options:

```bash
docker pull nmtwizard/opennmt-tf
docker run nmtwizard/opennmt-tf -h
```

[JSON configuration files](#configuration) are used to define training data, tokenization, and framework specific options (model, hyperparameters, etc).

As an example, let's train an English-German Transformer model with OpenNMT-tf.

**1\. Prepare the data.**

For simplicity, we assume that the training data is already tokenized and the current directory contains a `data/` subdirectory with the following structure:

```
$ tree data/
.
├── corpus
│   ├── train.de
│   └── train.en
└── vocab
    └── shared-vocab.txt
```

where:

* `train.de` and `train.en` are tokenized training files
* `shared-vocab.txt` contains one token per line and no special tokens

**2\. Define the configuration.**

The JSON configuration file is used to describe where to read the data (`data`) and how to transform it (`tokenization`). The `options` block is **specific to each framework** and defines the model and training hyperparameters. See [Configuration](#configuration) for more details.

```json
{
    "source": "en",
    "target": "de",
    "data": {
        "sample_dist": [
            {
                "path": "/data/corpus",
                "distribution": [
                    ["train", "*"]
                ]
            }
        ]
    },
    "tokenization": {
        "source": {
            "vocabulary": "/data/vocab/shared-vocab.txt",
            "mode": "space"
        },
        "target": {
            "vocabulary": "/data/vocab/shared-vocab.txt",
            "mode": "space"
        }
    },
    "options": {
        "model_type": "Transformer",
        "auto_config": true
    }
}
```

**3\. Train the model.**

The `train` command is used to start the training:

```bash
cat config.json | docker run -i --gpus all \
    -v $PWD/data:/data -v $PWD/models:/models nmtwizard/opennmt-tf \
    --model_storage /models --task_id my_model_1 --config - train
```

This command runs the training for 1 epoch and produces the model `models/my-model-1`. The model contains the latest checkpoint, the JSON configuration, and all required resources (such as the vocabulary files).

You can run the next epoch by passing the model name as argument:

```bash
docker run --gpus all -v $PWD/data:/data -v $PWD/models:/models nmtwizard/opennmt-tf \
    --model_storage /models --task_id my_model_2 --model my_model_1 train
```

Alternatively, you can run the full training in one command by setting the training step option available in the training framework.

**4\. Translate test files.**

Once a model is saved, you can start translating files with the `trans` command:

```bash
docker run --gpus all -v $PWD/data:/data -v $PWD/models:/models nmtwizard/opennmt-tf \
    --model_storage /models --model my_model_2 trans -i /data/test.en -o /data/output.de
```

This command translates `data/test.en` and saves the result to `data/output.de`.

**5\. Serve the model.**

At some point, you may want to turn your trained model into a translation service with the `serve` command:

```bash
docker run -p 4000:4000 --gpus all -v $PWD/models:/models nmtwizard/opennmt-tf \
    --model_storage /models --model my_model_2 serve --host 0.0.0.0 --port 4000
```

This command starts a translation server that accepts HTTP requests:

```bash
curl -X POST http://localhost:4000/translate -d '{"src":[{"text": "Hello world !"}]}'
```

See the [REST translation API](docs/rest_api.md) for more details.

To optimize the model size and loading latency, you can also `release` the model before serving. It will remove training-only information and possibly run additional optimizations:

```bash
docker run -v $PWD/models:/models nmtwizard/opennmt-tf \
    --model_storage /models --model my_model_2 release
```

This command produces the serving-only model `models/my_model_2_release`.

## Remote storages

Files and directories can be automatically downloaded from remote storages such as Amazon S3, Swift, or any server via SSH. This includes training data, models, and resources used in the configuration.

Remote storages should be configured in a JSON file and passed to the  `--storage_config` command line option:

```json
{
    "storage_id_1": {
        "type": "s3",
        "bucket": "model-catalog",
        "aws_credentials": {
            "access_key_id": "...",
            "secret_access_key": "...",
            "region_name": "..."
        }
    },
    "storage_id_2": {
        "type": "ssh",
        "server": "my-server.com",
        "basedir": "myrepo",
        "user": "root",
        "password": "root"
    }
}
```

*See the supported services and their parameters in [SYSTRAN/storages](https://github.com/SYSTRAN/storages).*

Paths on the command line and in the configuration can reference remote paths with the syntax `<storage_id>:<path>`, where:

* `<storage_id>` is the storage identifier in the JSON file above (e.g. `storage_id_1`)
* `<path>` is a file path on the remote storage

Files will be downloaded in the `/root/workspace/shared` directory within the Docker image. To minimize download cost, it is possible to mount this directory when running the Docker image. Future runs will reuse the local file if the remote file has not changed.

## Configuration

The JSON configuration file contains all parameters necessary to train and run models. It has the following general structure:

```json
{
    "source": "string",
    "target": "string",
    "data": {},
    "tokenization": {
        "source": {
            "vocabulary": "string",
            "mode": "string",
        },
        "target": {
            "vocabulary": "string",
            "mode": "string",
        }
    },
    "options": {},
    "serving": {}
}
```

### Description

#### `source` and `target`

They define the source and target languages (e.g. "en", "de", etc.).

#### `data`

The `data` section of the JSON configuration can be used to select data based on file patterns. The distribution is a list where each element contains:

* `path`: path to a directory where the distribution applies
* `distribution`: a list of filename patterns and weights

For example, the configuration below will randomly select 10,000 training examples in the directory `data/en_nl/train` from files that have `News`, `IT`, or `Dialog` in their name:


```json
{
    "data": {
        "sample": 10000,
        "sample_dist": [
            {
                "path": "/data/en_de/train",
                "distribution": [
                    ["News", 0.7],
                    ["IT", 0.3],
                    ["Dialog", 0.5]
                ]
            }
        ]
    }
}
```

Weights define the relative proportion that each pattern should take in the final sampling. They do not need to sum to 1. The special weight `"*"` can be used to force using all the examples associated with the pattern.

**Source and target files should have the same name and be suffixed by the language code.**

#### `tokenization`

This block accepts any tokenization options from [OpenNMT/Tokenizer](https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md).

The vocabulary file must have the following format:

* one token par line
* no special tokens (such as `<s>`, `</s>`, `<blank>`, `<unk>`, etc.)

We plan to add a `buildvocab` command to automatically generate it from the data.

#### `options`

This block contains the parameters that are **specific to the selected framework**: the model architecture, the training parameters, etc.

See the file `frameworks/./README.md` of the selected framework for more information.

#### `serving`

Serving reads additional values from the JSON configuration:

```json
{
    "serving": {
        "max_batch_size": 64
    }
}
```

where:

* `max_batch_size` is the maximum batch size to execute at once

These values can be overriden for [each request](docs/rest_api.md).

### Overriding the model configuration

When a model is set on the command line with `--model`, its configuration will be used. You can pass a partial configuration to `--config` in order to override some fields from the model configuration.

### Environment variables

Values in the configuration can use environment variables with the syntax `${VARIABLE_NAME}`.

This is especially useful to avoid hardcoding a remote storage identifier in the configuration. For example, one can define a data path as `${DATA_DIR}/en_de/corpus` in the configuration and then configure the storage identifer when running the Docker image with `-e DATA_DIR=storage_id_1:`.

## Add or extend frameworks

This repository consists of a Python module `nmtwizard` that implements the shared interface and extended features mentionned above. Each framework should then:

* extend the `nmtwizard.Framework` class and implement the logic that is specific to the framework
* define a Python entrypoint script that invokes `Framework.run()`
* define a `Dockerfile` that includes the framework, the `nmtwizard` module, and all dependencies.

Advanced users could extend existing frameworks to implement customized behavior.
