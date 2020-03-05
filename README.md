# nmt-wizard-docker

The aim of this project is to encapsulate training frameworks in Docker containers and expose a standardized interface for:

* preprocessing
* training
* translating
* serving

The training data are mounted at the container start and follow a specific directory structure (described later). Models and translation files can be fetched and pushed from various remote storage platform, including Amazon S3.

## Overview

Each framework exposes the same command line interface for providing training, translation, and deployment services. See for example:

```bash
docker run nmtwizard/opennmt-tf -h
```

## Configuration

### Environment variables

Some environment variables can be set (e.g. with the `-e` flags on `docker run`):

* `CORPUS_DIR` (default: `/root/corpus`): Path to the training corpus.
* `MODELS_DIR` (default: `/root/models`): Path to the models directory.
* `WORKSPACE_DIR` (default: `/root/workspace`): Path to the framework workspace.
* `LOG_LEVEL` (default: `INFO`): the Python log level.

Some frameworks may require additional environment variables, see their specific resources in `frameworks/`.

### Run configuration

The JSON configuration file contains the parameters necessary to run the command. It has the following format:

```text
{
    "source": "string",  // (mandatory) 2-letter iso code for source language
    "target": "string",  // (mandatory) 2-letter iso code for target language
    "model": "string",  // (mandatory for trans, serve) Full model name as uuid64
    "data": {
        //  (optional) Data distribution rules.
    },
    "preprocess": [
        // a list of preprocessing operators, such as tokenization :
        {
            "op":"tokenization",
            "source": {
                // source specific tokenization options (from OpenNMT/Tokenizer)
            },
            "target": {
                // target specific tokenization options (from OpenNMT/Tokenizer)
            }
        }
    ],
    "vocabulary": {
        // Vocabularies for translation model.
        "source": {
            "path": "string"
        },
        "target": {
            "path": "string"
        }
    },
    "options": {
        // (optional) Run options specific to each framework.
    }
}
```

**Note:** When loading an existing model and a configuration file is provided, it is merged with the saved model configuration file.

### Storage configuration

Multiple storage destinations can be defined with the `--storage_config` option that references a JSON file:

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

These storages can then be used to define model and file locations, e.g.:

```bash
docker run nmtwizard/opennmt-tf --storage_config storages.json \
    --model_storage storage_id_2: --model MODEL_ID \
    trans -i storage_id_1:test.fr -o storage_id_2:test.en
```

If the configuration is not provided or a storage identifier is not set, the host filesystem is used.

Available storage types are:

* `local`: local storage. Available options:
  * `basedir` (optional): defines base directory for relative paths
* `ssh`: transfer files via SSH. Available options:
  * `server`: server hostname
  * `port` (default: 22): port to use for connecting
  * `user`: username for login
  * `password` or `pkey`: login credentials
  * `basedir` (optional): defines base directory for relative paths
* `s3`: transfer files to and from Amazon S3. Available options:
  * `bucket`: name of the bucket
  * `aws_credentials`: Amazon credentials with,
    * `access_key_id`
    * `secret_access_key`
    * `region_name`
* `http`: transfer files via GET and POST requests. Requires to configure patterns that are URLs containing the `%s` string placeholders that will be expanded with python `%` operator (e.g. `http://opennmt.net/%s/`):
  * `get_pattern`
  * `post_pattern`
  * `list_pattern`

### Training data sampling

The `data` section of the run configuration can be used to define advanced data selection based on file patterns. The distribution is a JSON list where each element is a dictionary with 2 elements:

* `path` : Path to a directory on which theses rules apply
* `distribution`: a dictionary of patterns/weights as defined [here](http://opennmt.net/OpenNMT/training/sampling/#sampling-distribution-rules).

For example:

```json
"data": {
    "sample": 10000,
    "sample_dist": [{
        "path": "${CORPUS_DIR}/en_nl/train",
        "distribution": [
            ["News", 0.7],
            ["IT", 0.3],
            ["Dialog", "*"]
        ]
    }]
}
```

will select 10,000 training examples from `${CORPUS_DIR}/en_nl/train` and only from files containing `News` or `IT` in their name. The majority of the examples will come from `News` files (weight `0.7`).

**Note:** If this section is not provided, all files from `${CORPUS_DIR}/train` will be used for the training.

### Preprocessing

Data sampling and tokenization are preparing corpus for the training process. It is possible to get access sampled and tokenized corpus using `preprocess` command and by mounting `/root/workspace` volume.

The following command is sampling and tokenizing the corpus from `${PWD}/test/corpus` into `${PWD}/workspace`:

```bash
cat config.json | docker run -i --rm -v ${PWD}/test/corpus:/root/corpus -v ${PWD}/workspace:/root/workspace image -c - preprocess
```

## Corpus structure

The corpus directory should contain:

* `train`: Containing the training files, suffixed by the 2-letter iso language code.
* `vocab`: Containing the vocabularies and BPE models.

When running the Docker container, the corpus directory should be mounted, e.g. with `-v /home/corpus/en_fr:/root/corpus`.

**Note:** `${CORPUS_DIR}` can be used in the run configuration to locate data files in particular vocabulary files. `${TRAIN_DIR}` can be used the same way - but the resources accessed through this variable will not be bundled in the translation model.

## Models

The models are saved in a directory named by their ID. This package contains all the resources necessary for translation or deployment (BPE models, vocabularies, etc.). For instance, a typical OpenNMT-tf model will contain:

```text
$ ls -1 952f4f9b-b446-4aa4-bfc0-28a510c6df73/
checkpoint
checksum.md5
config.json
de-vocab.txt
en-vocab.txt
model.ckpt-149.data-00000-of-00002
model.ckpt-149.data-00001-of-00002
model.ckpt-149.index
model.ckpt-149.meta
model_description.py
```

In the `config.json` file, the path to the model dependencies is prefixed by `${MODEL_DIR}` which is automatically set when a model is loaded.

* `checksum.md5` file is generated from the content of the model and is used to check integrity of the model
* optionally, a `README.md` file can describe the model. It is generated from `description` field of the config file and is not taken into account for checksum.

### Released version

Before serving a trained model, it is required to run a `release` step, for example:

```bash
docker run nmtwizard/opennmt-tf \
    --storage_config storages.json \
    --model_storage s3_model: \
    --model 952f4f9b-b446-4aa4-bfc0-28a510c6df73 \
    --gpuid 1 \
    release --destination s3_model:
```

will fetch the model `952f4f9b-b446-4aa4-bfc0-28a510c6df73` from the storage `s3_model` and push the released version `952f4f9b-b446-4aa4-bfc0-28a510c6df73_release` to the same storage.

Released models are smaller and more efficient but can only be used for serving.

## Serving

Compatible frameworks provide an uniform API for serving released model via the `serve` command, e.g.:

```bash
nvidia-docker run nmtwizard/opennmt-tf \
    --storage_config storages.json \
    --model_storage s3_model: \
    --model 952f4f9b-b446-4aa4-bfc0-28a510c6df73_release \
    --gpuid 1 \
    serve --host 0.0.0.0 --port 5000
```

will fetch the released model `952f4f9b-b446-4aa4-bfc0-28a510c6df73_release` from the storage `s3_model` (see the previous section), start a backend translation service on the first GPU, and serve translation on port 5000.

Serving accepts additional run configurations:

```json
{
    "serving": {
        "timeout": 10.0,
        "max_batch_size": 64
    }
}
```

where:

* `timeout` is the maximum duration in seconds to wait for the translation to complete
* `max_batch_size` is the maximum batch size to execute at once

The `timeout` and `max_batch_size` values can be overriden for each request.

### Interface

#### `POST /translate`

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
        "timeout": 10.0,
        "max_batch_size": 32,
        "config": {}
    },
    "src": [
        {"text": "Source sentence 1", "config": {}, "options": {}},
        {"text": "Source sentence 2", "config": {}, "options": {}}
    ]
}
```

* The `config` fields define request-specific and sentence-specific overrides to the global configuration file.
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
* **HTTP 503**
  * The backend service is unavailable.
* **HTTP 504**
  * The translation request timed out.

#### `POST /unload_model`

Unload the model from the reserved resource. In its simplest form, this route will terminate the backend translation service.

#### `POST /reload_model`

Reload the model on the reserved resource. In its simplest form, this route will terminate the backend translation service if it is still running and start a new instance.

## Usage

### Local

During development, `entrypoint.py` can be invoked directly if the environment is properly set to run the required services (installed framework, set environment variables, etc.). See `README.md` files of each framework in `frameworks/` for specific instructions.

If you don't have the required environment, consider building the Docker image instead.

### Docker

*To be able to run the image on GPUs, you need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation) installed on the host machine.*

#### Directory mounts

When running an image, the following mounting points can be used:

* `/root/corpus`: Training corpus and resources.
* `/root/models` (optional): Host repository of models.
* `/root/workspace` (optional): internal workspace data used for corpus preparation. Has to be provided for `sample`.

#### Example

Running the Docker image is equivalent to running `entrypoint.py`. You should pass the same command line options and mount required files or directories, e.g.:

```bash
cat config.json | nvidia-docker run -a STDIN -i --rm \
    -v /home/models:/root/models -v /home/corpus/en_fr:/root/corpus \
    my-image -c - -ms /root/models -g 2 train
```
