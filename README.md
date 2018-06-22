# nmt-wizard-docker

The aim of this project is to encapsulate training frameworks in Docker containers and expose a standardized interface for:

* training
* translating
* data sampling and preprocessing - useful mainly for data preparation and vocabulary building
* serving (for selected frameworks)

The training data are mounted at the container start and follow a specific directory structure (described later). Models and translation files can be fetched and pushed from various remote storage platform. Currently only SSH and Amazon S3 are supported.

## Overview

Each framework exposes the same command line interface for providing training, translation, and deployment services. See for example:

```bash
python frameworks/opennmt_lua/entrypoint.py -h
```

which is defined as the entrypoint in `frameworks/opennmt_lua/Dockerfile`. The environment such as the framework location, the corpus location, the credentials, etc. are defined via environment variables. The `Dockerfile`s also encapsulate the environment specific to each framework.

## Configuration

### Environment variables

Some environment variables can be set (e.g. with the `-e` flags on `docker run`):

* `CORPUS_DIR` (default: `/root/corpus`): Path to the training corpus.
* `MODELS_DIR` (default: `/root/models`): Path to the models directory.
* `WORKSPACE_DIR` (default: `/root/workspace`): Path to the framework workspace (generated and temporary files).
* `LOG_LEVEL` (default: `INFO`): the Python log level.

Some frameworks may require additional environment variables, see their specific resources in `frameworks/`.

### Run configuration

The JSON configuration file contains the parameters necessary to run the command. It has the following format:

```text
{
    "source": "string",  // (mandatory) 2-letter iso code for source language
    "target": "string",  // (mandatory) 2-letter iso code for target language
    "model": "string",  // (mandatory for trans, serve) Full model name as uuid64
    "imageTag": String,  // (mandatory) Full URL of the image: url/image:tag.
    "build": {
        // (optional) Generated at the end of a training.
        "containerId": "string",  // (optional) ID of the builder container
        "distribution": { },  // Files and patterns sampled for this epoch
        "endDate": Float,  // Timestamp of this epoch end
        "startDate": Float  // Timestamp of this epoch start
    },
    "data": {
        //  (optional) Data distribution rules.
    },
    "tokenization": {
        // Vocabularies and tokenization options (from OpenNMT/Tokenizer).
        "source": {
            "vocabulary": "string"
            // other source specific tokenization options
        },
        "target": {
            "vocabulary": "string"
            // other target specific tokenization options
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
        "user": "root",
        "password": "root"
    }
}
```

These storages can then be used to define model and file locations, e.g.:

```bash
python entrypoint.py --storage_config storages.json --model_storage storage_id_2: \
    --model MODEL_ID trans -i storage_id_1:test.fr -o storage_id_2:test.en
```

If the configuration is not provided or a storage identifier is not set, the host filesystem is used.

Available storage types are:
* `ssh`: transfer files or directories using ssh, requires `server` name, `user` and `password`
* `s3`: transfer files or directories using ssh, requires `bucket` and `aws_credentials`
* `http`: transfer files only using simple GET and POST requests. Requires `get_pattern` and `push_pattern` that are urls using `%s` string placeholders, expanded with python `%` operator: for instance `http://opennmt.net/%s/`

### Training data sampling

The `data` section of the run configuration can be used to define advanced data selection based on file patterns. The distribution is a JSON list where each element is a dictionary with 2 elements:

* `path` : Path to sub-directory on which theses rules apply
* `distribution`: a dictionary of patterns/weights as defined [here](http://opennmt.net/OpenNMT/training/sampling/#sampling-distribution-rules).

For example:

```json
"data": {
    "sample_dist": [{
        "path": "train",
        "distribution": [
            ["News", 0.7],
            ["IT", 0.3],
            ["Dialog", "*"]
        ]
    }],
    "version": "1",
    "sample": 10000,
    "train_dir": "en_nl"
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

**Note:** `${CORPUS_DIR}` can be used in the run configuration to locate data files in particular vocabulary files.

## Models

The models are saved in a directory named by their ID. This package contains all the resources necessary for translation or deployment (BPE models, vocabularies, etc.). For instance, a typical OpenNMT-lua model will contain:

```text
952f4f9b-b446-4aa4-bfc0-28a510c6df73/checksum.md5
952f4f9b-b446-4aa4-bfc0-28a510c6df73/config.json
952f4f9b-b446-4aa4-bfc0-28a510c6df73/model.t7
952f4f9b-b446-4aa4-bfc0-28a510c6df73/model_released.t7
952f4f9b-b446-4aa4-bfc0-28a510c6df73/vocab.src.dict
952f4f9b-b446-4aa4-bfc0-28a510c6df73/vocab.tgt.dict
```

In the `config.json` file, the path to the model dependencies is prefixed by `${MODEL_DIR}` which is automatically set when a model is loaded.

## Serving

Compatible frameworks provide an uniform API for serving translation via the `serve` command, e.g.:

```bash
python frameworks/opennmt_lua/entrypoint.py
    --model_storage s3_model: \
    --model 952f4f9b-b446-4aa4-bfc0-28a510c6df73 \
    --gpuid 1 \
    serve --host 0.0.0.0 --port 5000
```

will fetch the model `952f4f9b-b446-4aa4-bfc0-28a510c6df73` from the storage `s3_model`, start a backend translation service on the first GPU, and serve translation on port 5000.

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

**Input:**

```json
{
    "src": [
        {"text": "Source sentence 1"},
        {"text": "Source sentence 2"}
    ],
    "options": {
        "timeout": 10.0
    }
}
```

(The `options` field is optional.)

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
* **HTTP 503**
  * The backend service is unavailable.
* **HTTP 504**
  * The translation request timed out.

#### `POST /unload_model`

Unload the model from the reserved resource. In its simplest form, this route will terminate the backend translation service.

#### `POST /reload_model`

Reload the model on the reserved resource. In its simplest form, this route will terminate the backend translation service if it is still running and start a new instance.

### Supported frameworks

Serving is currently supported by the following frameworks:

* `google_transate`
* `opennmt_lua`
* `opennmt_tf`

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
