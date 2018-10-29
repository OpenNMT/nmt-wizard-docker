"""Shared logic and abstractions of frameworks."""

import os
import abc
import copy
import json
import argparse
import time
import filecmp
import shutil
import re
import uuid
import sys
import six

from nmtwizard.beat_service import start_beat_service
from nmtwizard.logger import get_logger
from nmtwizard.utils import md5files
from nmtwizard.sampler import sample
from nmtwizard.storage import StorageClient
from nmtwizard import serving
from nmtwizard import data
from nmtwizard import tokenizer

ENVVAR_RE = re.compile(r'\${(.*?)}')
ENVVAR_ABS_RE = re.compile(r'(\${.*?}.*)/(.*)')

logger = get_logger(__name__)


@six.add_metaclass(abc.ABCMeta)
class Framework(object):
    """Base class for frameworks."""

    def __init__(self, stateless=False, support_multi_training_files=False):
        """Initializes the framework.

        Args:
          stateless: If True, no models are generated or fetched. This is the case
            for local frameworks that are bridges to remote services (e.g. Google Translate).
          support_multi_training_files: If True, the framework should implement the
            training API receiving a data directory as argument and additional per file
            metadata.
        """
        self._stateless = stateless
        self._support_multi_training_files = support_multi_training_files
        self._corpus_dir = os.getenv('CORPUS_DIR', '/root/corpus')
        self._models_dir = os.getenv('MODELS_DIR', '/root/models')
        if not os.path.exists(self._models_dir):
            os.makedirs(self._models_dir)
        workspace_dir = os.getenv('WORKSPACE_DIR', '/root/workspace')
        self._output_dir = os.path.join(workspace_dir, 'output')
        self._data_dir = os.path.join(workspace_dir, 'data')
        self._tmp_dir = os.path.join(workspace_dir, 'tmp')
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)
        if not os.path.exists(self._tmp_dir):
            os.makedirs(self._tmp_dir)

    @abc.abstractmethod
    def train(self,
              config,
              src_file,
              tgt_file,
              src_vocab_info,
              tgt_vocab_info,
              model_path=None,
              gpuid=0):
        """Trains for one epoch.

        Args:
          config: The run configuration.
          src_file: The local path to the preprocessed (if any) source file.
          tgt_file: The local path to the preprocessed (if any) target file.
          src_vocab_info: Source vocabulary metadata (see _get_vocab_info).
          tgt_vocab_info: Target vocabulary metadata (see _get_vocab_info).
          model_path: The path to a model to load from.
          gpuid: The GPU identifier.

        Returns:
          A dictionary of filenames to paths of objects to save in the model package.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def trans(self, config, model_path, input, output, gpuid=0):
        """Translates a file.

        Args:
          config: The run configuration.
          model_path: The path to the model to use.
          input: The local path to the preprocessed (if any) source file.
          output: The local path to the file that should contain the translation.
          gpuid: The GPU identifier.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def release(self, config, model_path, gpuid=0):
        """Releases a model for serving.

        Args:
          config: The run configuration.
          model_path: The path to the model to release.
          gpuid: The GPU identifier.

        Returns:
          A dictionary of filenames to paths of objects to save in the released model package.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def serve(self, config, model_path, gpuid=0):
        """Start a framework dependent serving service in the background.

        Args:
          config: The run configuration.
          model_path: The path to the model to serve.
          gpuid: The GPU identifier.

        Returns:
          A tuple with the created process and a dictionary containing
          information to reach the backend service (e.g. port number).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_request(self, batch_inputs, info, timeout=None):
        """Forward a frontend translation request to the framework serving service.

        Args:
          batch_inputs: A list of inputs (usually tokens).
          info: The backend service information returned by serve().
          timeout: Timeout in seconds for the translation request.

        Returns:
          A list of list (batch x num. hypotheses) of serving.TranslationOutput.
        """
        raise NotImplementedError()

    def train_multi_files(self,
                          config,
                          data_dir,
                          src_vocab_info,
                          tgt_vocab_info,
                          model_path=None,
                          num_samples=None,
                          samples_metadata=None,
                          gpuid=0):
        """Trains for one epoch on a directory of data.

        If the framework set support_multi_training_files to False (the default),
        the standard train API will be called on a single parallel file.

        Args:
          config: The run configuration.
          data_dir: The directory containing the training files.
          src_vocab_info: Source vocabulary metadata (see _get_vocab_info).
          tgt_vocab_info: Target vocabulary metadata (see _get_vocab_info).
          model_path: The path to a model to load from.
          num_samples: The total number of sentences of the training data.
          samples_metadata: A dictionary mapping filenames to extra metadata set
            in the distribution configuration.
          gpuid: The GPU identifier.

        Returns:
          See train().
        """
        if self._support_multi_training_files:
            raise NotImplementedError()
        else:
            return self.train(
                config,
                os.path.join(data_dir, 'train.%s' % config['source']),
                os.path.join(data_dir, 'train.%s' % config['target']),
                src_vocab_info,
                tgt_vocab_info,
                model_path=model_path,
                gpuid=gpuid)

    def run(self):
        """Main entrypoint."""
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', default=None,
                            help=('Configuration as a file or a JSON string. '
                                  'Setting "-" will read from the standard input.'))
        parser.add_argument('-s', '--storage_config', default=None,
                            help=('Configuration of available storages as a file or a JSON string. '
                                  'Setting "-" will read from the standard input.'))
        parser.add_argument('-ms', '--model_storage',
                            help='Model storage in the form <storage_id>:[<path>].')
        parser.add_argument('-m', '--model', default=None,
                            help='Model to load.')
        parser.add_argument('-g', '--gpuid', default="0",
                            help="Comma-separated list of 1-indexed GPU identifiers (0 for CPU).")
        parser.add_argument('-t', '--task_id', default=None,
                            help="Identifier of this run.")
        parser.add_argument('-i', '--image', default="?",
                            help="Full URL (registry/image:tag) of the image used for this run.")
        parser.add_argument('-b', '--beat_url', default=None,
                            help=("Endpoint that listens to beat requests "
                                  "(push notifications of activity)."))
        parser.add_argument('-bi', '--beat_interval', default=30, type=int,
                            help="Interval of beat requests in seconds.")
        parser.add_argument('--no_push', default=False, action='store_true',
                            help='Do not push model.')

        subparsers = parser.add_subparsers(help='Run type', dest='cmd')
        parser_train = subparsers.add_parser('train', help='Run a training.')

        parser_trans = subparsers.add_parser('trans', help='Run a translation.')
        parser_trans.add_argument('-i', '--input', required=True, nargs='+',
                                  help='Input file.')
        parser_trans.add_argument('-o', '--output', required=True, nargs='+',
                                  help='Output file.')

        parser_release = subparsers.add_parser('release', help='Release a model for serving.')
        parser_release.add_argument('-d', '--destination', default=None,
                                    help='Released model storage (defaults to the model storage).')

        parser_serve = subparsers.add_parser('serve', help='Serve a model.')
        parser_serve.add_argument('-hs', '--host', default="0.0.0.0",
                                  help='Serving hostname.')
        parser_serve.add_argument('-p', '--port', type=int, default=4000,
                                  help='Serving port.')

        parser_preprocess = subparsers.add_parser('preprocess', help='Sample and preprocess corpus.')
        parser_preprocess.add_argument('--build_model', default=False, action='store_true',
                                       help='Preprocess data into a model.')
        parser.build_vocab = subparsers.add_parser('buildvocab', help='Build vocabularies.')

        args = parser.parse_args()
        if args.config is None and args.model is None:
            parser.error('at least one of --config or --model options must be set')
        if (not self._stateless
            and (args.cmd != 'preprocess' or args.build_model)
            and not args.model_storage):
            parser.error('argument -ms/--model_storage is required')
        if args.task_id is None:
            args.task_id = str(uuid.uuid4())

        # for backward compatibility - convert singleton in int
        args.gpuid = args.gpuid.split(',')
        args.gpuid = [int(g) for g in args.gpuid]
        if len(args.gpuid) == 1:
            args.gpuid = args.gpuid[0]

        start_beat_service(
            os.uname()[1],
            args.beat_url,
            args.task_id,
            interval=args.beat_interval)

        config = load_config(args.config) if args.config is not None else {}
        parent_model = args.model or config.get('model')

        storage = StorageClient(
            tmp_dir=self._tmp_dir,
            config=load_config(args.storage_config) if args.storage_config else None)

        if parent_model is not None and not self._stateless:
            # Download model locally and merge the configuration.
            remote_model_path = storage.join(args.model_storage, parent_model)
            model_path = os.path.join(self._models_dir, parent_model)
            fetch_model(storage, remote_model_path, model_path)
            with open(os.path.join(model_path, 'config.json'), 'r') as config_file:
                model_config = json.load(config_file)
            if 'modelType' not in model_config:
                if parent_model.endswith('_release'):
                    model_config['modelType'] = 'release'
                else:
                    model_config['modelType'] = 'checkpoint'
            config = merge_config(copy.deepcopy(model_config), config)
        else:
            model_path = None
            model_config = None

        if args.cmd == 'train':
            if (parent_model is not None
                and config['modelType'] not in ('checkpoint', 'base', 'preprocess')):
                raise ValueError('cannot train from a model that is not a training checkpoint, '
                                 'a base model, or a preprocess model')
            self.train_wrapper(
                args.task_id,
                config,
                storage,
                args.model_storage,
                args.image,
                parent_model=parent_model,
                model_path=model_path,
                model_config=model_config,
                gpuid=args.gpuid,
                push_model=not args.no_push)
        elif args.cmd == 'buildvocab':
            self.build_vocab(
                args.task_id,
                config,
                storage,
                args.model_storage,
                args.image,
                push_model=not args.no_push)
        elif args.cmd == 'trans':
            if (not self._stateless
                and (parent_model is None or config['modelType'] != 'checkpoint')):
                raise ValueError('translation requires a training checkpoint')
            self.trans_wrapper(
                config,
                model_path,
                storage,
                args.input,
                args.output,
                gpuid=args.gpuid)
        elif args.cmd == 'release':
            if (not self._stateless
                and (parent_model is None or config['modelType'] != 'checkpoint')):
                raise ValueError('releasing requires a training checkpoint')
            if args.destination is None:
                args.destination = args.model_storage
            self.release_wrapper(
                config,
                model_path,
                storage,
                args.image,
                args.destination,
                gpuid=args.gpuid,
                push_model=not args.no_push)
        elif args.cmd == 'serve':
            if (not self._stateless
                and (parent_model is None or config['modelType'] != 'release')):
                raise ValueError('serving requires a released model')
            self.serve_wrapper(
                config, model_path, args.host, args.port, gpuid=args.gpuid)
        elif args.cmd == 'preprocess':
            if not args.build_model:
                self.preprocess(config, storage)
            else:
                if (parent_model is not None
                    and config['modelType'] not in ('checkpoint', 'base')):
                    raise ValueError('cannot preprocess from a model that is not a training '
                                     'checkpoint or a base model')
                self.preprocess_into_model(
                    args.task_id,
                    config,
                    storage,
                    args.model_storage,
                    args.image,
                    parent_model=parent_model,
                    model_path=model_path,
                    push_model=not args.no_push)

    def train_wrapper(self,
                      model_id,
                      config,
                      storage,
                      model_storage,
                      image,
                      parent_model=None,
                      model_path=None,
                      model_config=None,
                      gpuid=0,
                      push_model=True):
        logger.info('Starting training model %s', model_id)
        start_time = time.time()

        local_config = resolve_environment_variables(config)
        local_model_config = (
            resolve_environment_variables(model_config)
            if model_config is not None else None)

        src_vocab_info = self._get_vocab_info(
            'source', config, local_config, model_config=local_model_config)
        tgt_vocab_info = self._get_vocab_info(
            'target', config, local_config, model_config=local_model_config)

        parent_model_type = config.get('modelType') if model_path is not None else None

        if parent_model_type == 'preprocess':
            train_dir = 'data'
            data_dir = os.path.join(model_path, train_dir)
            num_samples = config['sampling']['numSamples']
            samples_metadata = config['sampling']['samplesMetadata']
            del config['sampling']
            logger.info('Using preprocessed data from %s' % data_dir)
        else:
            data_dir, train_dir, num_samples, distribution_summary, samples_metadata = (
                self._generate_training_data(local_config))
            if num_samples == 0:
                raise RuntimeError('data sampling generated 0 sentences')
            if not self._support_multi_training_files:
                data_dir = self._merge_multi_training_files(
                    data_dir, train_dir, config['source'], config['target'])

        if parent_model_type in ('base',):
            model_path = None
        objects = self.train_multi_files(
            local_config,
            data_dir,
            src_vocab_info,
            tgt_vocab_info,
            model_path=model_path,
            num_samples=num_samples,
            samples_metadata=samples_metadata,
            gpuid=gpuid)

        end_time = time.time()
        logger.info('Finished training model %s in %s seconds', model_id, str(end_time-start_time))

        # Fill training details.
        config['model'] = model_id
        config['modelType'] = 'checkpoint'
        config['imageTag'] = image
        build_info = {
            'containerId': os.uname()[1],
            'endDate': end_time,
            'startDate': start_time
        }

        if parent_model_type == 'preprocess':
            # Inherit distribution summary and the parent from the preprocess run.
            config['build'].update(build_info)
        else:
            if parent_model:
                config['parent_model'] = parent_model
            parent_build_info = config.get('build')
            build_info = self._summarize_data_distribution(
                build_info, distribution_summary, parent_build_info=parent_build_info)
            config['build'] = build_info

        # Build and push the model package.
        bundle_dependencies(objects, config)
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config)
        if push_model:
            storage.push(objects_dir, storage.join(model_storage, model_id))

    def build_vocab(self,
                    model_id,
                    config,
                    storage,
                    model_storage,
                    image,
                    push_model=True):
        start_time = time.time()
        local_config = resolve_environment_variables(config)
        objects, tokenization_config = self._generate_vocabularies(local_config)
        end_time = time.time()

        config['tokenization'] = tokenization_config
        config['model'] = model_id
        config['modelType'] = 'base'
        config['imageTag'] = image
        config['build'] = {
            'containerId': os.uname()[1],
            'endDate': end_time,
            'startDate': start_time
        }

        bundle_dependencies(objects, config)
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config)
        if push_model:
            storage.push(objects_dir, storage.join(model_storage, model_id))

    def trans_wrapper(self, config, model_path, storage,
                      inputs, outputs, gpuid=0):
        if len(inputs) != len(outputs):
            raise ValueError("Mismatch of input/output files number, got %d and %d" % (
                len(inputs), len(outputs)))

        local_config = resolve_environment_variables(config)
        failed_translation = 0

        for input, output in zip(inputs, outputs):
            try:
                path_input = os.path.join(self._data_dir, storage.split(input)[-1])
                path_output = os.path.join(self._output_dir, storage.split(output)[-1])
                storage.get_file(input, path_input)
                logger.info('Starting translation %s to %s', path_input, path_output)
                start_time = time.time()
                path_input = self._preprocess_file(local_config, path_input)
                self.trans(local_config,
                           model_path,
                           path_input,
                           path_output,
                           gpuid=gpuid)
                path_output = self._postprocess_file(local_config, path_input, path_output)
                storage.push(path_output, output)
                end_time = time.time()
                logger.info('Finished translation in %s seconds', str(end_time-start_time))
            except Exception as e:
                # Catch any exception to not impact other translations.
                logger.error("Translation of %s failed with error %s" % (path_input, str(e)))
                logger.warning("Skipping translation of %s" % path_input)
                failed_translation += 1

        if failed_translation == len(inputs):
            raise RuntimeError("All translation failed, see error logs")

    def release_wrapper(self,
                        config,
                        model_path,
                        storage,
                        image,
                        destination,
                        gpuid=0,
                        push_model=True):
        local_config = resolve_environment_variables(config)
        objects = self.release(local_config, model_path, gpuid=gpuid)
        extract_model_resources(objects, config)
        model_id = config['model'] + '_release'
        config['parent_model'] = config['model']
        config['model'] = model_id
        config['modelType'] = 'release'
        config['imageTag'] = image
        config['build'] = {
            'containerId': os.uname()[1]
        }
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config)
        if push_model:
            storage.push(objects_dir, storage.join(destination, model_id))

    def serve_wrapper(self, config, model_path, host, port, gpuid=0):
        local_config = resolve_environment_variables(config)
        serving.start_server(
            host,
            port,
            local_config.get('serving'),
            self._serving_state(local_config),
            lambda: self.serve(local_config, model_path, gpuid=gpuid),
            self._preprocess_input,
            self.forward_request,
            self._postprocess_output)

    def preprocess(self, config, storage):
        logger.info('Starting preprocessing data ')
        start_time = time.time()

        local_config = resolve_environment_variables(config)
        data_dir, train_dir, num_samples, distribution_summary, samples_metadata = (
            self._generate_training_data(local_config))

        end_time = time.time()
        logger.info('Finished preprocessing data in %s seconds into %s', 
                    str(end_time-start_time), data_dir)

    def preprocess_into_model(self,
                              model_id,
                              config,
                              storage,
                              model_storage,
                              image,
                              parent_model=None,
                              model_path=None,
                              push_model=True):
        logger.info('Starting preprocessing %s', model_id)
        start_time = time.time()

        local_config = resolve_environment_variables(config)
        data_dir, train_dir, num_samples, distribution_summary, samples_metadata = (
            self._generate_training_data(local_config))
        if num_samples == 0:
            raise RuntimeError('data sampling generated 0 sentences')
        if not self._support_multi_training_files:
            data_dir = self._merge_multi_training_files(
                data_dir, train_dir, config['source'], config['target'])

        end_time = time.time()
        logger.info('Finished preprocessing %s in %s seconds', model_id, str(end_time-start_time))

        # Fill training details.
        if parent_model:
            config['parent_model'] = parent_model
        config['model'] = model_id
        config['modelType'] = 'preprocess'
        config['imageTag'] = image
        config['sampling'] = {
            'numSamples': num_samples,
            'samplesMetadata': samples_metadata}
        parent_build_info = config.get('build')
        build_info = {
            'containerId': os.uname()[1],
            'endDate': end_time,
            'startDate': start_time
        }

        build_info = self._summarize_data_distribution(
            build_info, distribution_summary, parent_build_info=parent_build_info)
        config['build'] = build_info

        # Build and push the model package.
        objects = {'data': data_dir}
        bundle_dependencies(objects, config)
        # Forward other files from the parent model.
        if model_path is not None:
            for f in os.listdir(model_path):
                if f not in objects:
                    objects[f] = os.path.join(model_path, f)
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config)
        if push_model:
            storage.push(objects_dir, storage.join(model_storage, model_id))

    def _get_vocab_info(self, side, config, local_config, model_config=None):
        if config.get('tokenization', {}).get(side, {}).get('vocabulary') is None:
            return None
        current_vocab = self._convert_vocab(
            local_config['tokenization'][side]['vocabulary'],
            basename='%s-vocab.txt' % side)
        model_vocab = None
        vocab_changed = False
        if (model_config is not None
            and (local_config['tokenization'][side]['vocabulary']
                 != model_config['tokenization'][side]['vocabulary'])):
            model_vocab = self._convert_vocab(
                model_config['tokenization'][side]['vocabulary'],
                basename='model-%s-vocab.txt' % side)
            vocab_changed = not filecmp.cmp(model_vocab, current_vocab)
            if vocab_changed and not config['tokenization'][side].get('replace_vocab', False):
                raise ValueError('%s vocabulary has changed but replace_vocab is not set.'
                                 % side.capitalize())
        if 'replace_vocab' in config['tokenization'][side]:
            del config['tokenization'][side]['replace_vocab']
            del local_config['tokenization'][side]['replace_vocab']
        return {
            'current': current_vocab,
            'model': model_vocab,
            'changed': vocab_changed
        }

    def _serving_state(self, config):
        state = {}
        if 'tokenization' in config:
            tok_config = config['tokenization']
            state['src_tokenizer'] = tokenizer.build_tokenizer(tok_config['source'])
            state['tgt_tokenizer'] = tokenizer.build_tokenizer(tok_config['target'])
        return state

    def _preprocess_input(self, state, input, extra_config):
        if isinstance(input, list):
            tokens = input
        elif 'src_tokenizer' in state:
            input = input.encode('utf-8')
            tokens, _ = state['src_tokenizer'].tokenize(input)
            tokens = [token.decode('utf-8') for token in tokens]
        else:
            tokens = input.split()
        return tokens

    def _postprocess_output(self, state, source, target, extra_config):
        if not isinstance(target, list):
            text = target
        elif 'tgt_tokenizer' in state:
            output = [out.encode('utf-8') for out in target]
            text = state['tgt_tokenizer'].detokenize(output)
            text.decode('utf-8')
        else:
            text = ' '.join(target)
        return text

    def _preprocess_file(self, config, input):
        if 'tokenization' in config:
            tok_config = config['tokenization']
            src_tokenizer = tokenizer.build_tokenizer(tok_config['source'])
            output = "%s.tok" % input
            tokenizer.tokenize_file(src_tokenizer, input, output)
            return output
        return input

    def _postprocess_file(self, config, source, target):
        if 'tokenization' in config:
            tok_config = config['tokenization']
            tgt_tokenizer = tokenizer.build_tokenizer(tok_config['target'])
            output = "%s.detok" % target
            tokenizer.detokenize_file(tgt_tokenizer, target, output)
            return output
        return target

    def _merge_multi_training_files(self, data_path, train_dir, source, target):
        merged_dir = os.path.join(self._data_dir, 'merged')
        if not os.path.exists(merged_dir):
            os.mkdir(merged_dir)
        merged_path = os.path.join(merged_dir, train_dir)
        logger.info('Merging training data to %s/train.{%s,%s}',
                    merged_path, source, target)
        data.merge_files_in_directory(data_path, merged_path, source, target)
        return merged_path

    def _convert_vocab(self, vocab_file, basename=None):
        if basename is None:
            basename = os.path.basename(vocab_file)
        converted_vocab_file = os.path.join(self._data_dir, basename)
        with open(vocab_file, 'rb') as vocab, open(converted_vocab_file, 'wb') as converted_vocab:
            header = True
            index = 0
            for line in vocab:
                if header and line[0] == b'#':
                    continue
                header = False
                token = line.strip().split()[0]
                self._map_vocab_entry(index, token, converted_vocab)
                index += 1
        return converted_vocab_file

    def _generate_training_data(self, config):
        if 'data' in config and 'train_dir' in config['data']:
            train_dir = config['data']['train_dir']
        else:
            train_dir = 'train'
        data_path = os.path.join(self._corpus_dir, train_dir)
        num_samples = None
        summary = None
        metadata = None
        logger.info('Generating training data from %s', data_path)
        if 'data' in config and 'sample_dist' in config['data']:
            sample_dir = os.path.join(self._data_dir, 'sample')
            if not os.path.exists(sample_dir):
                os.mkdir(sample_dir)
            sample_path = os.path.join(sample_dir, train_dir)
            logger.info('Sampling training data to %s', sample_path)
            summary, metadata = sample(
                config['data']['sample'],
                config['data']['sample_dist'],
                data_path,
                sample_path,
                config['source'],
                config['target'])
            num_samples = sum(six.itervalues(summary['file']))
            data_path = sample_path
        if 'tokenization' in config:
            tok_config = config['tokenization']
            src_tokenizer = tokenizer.build_tokenizer(tok_config['source'])
            tgt_tokenizer = tokenizer.build_tokenizer(tok_config['target'])
            tokenized_dir = os.path.join(self._data_dir, 'tokenized')
            if not os.path.exists(tokenized_dir):
                os.mkdir(tokenized_dir)
            tokenized_path = os.path.join(tokenized_dir, train_dir)
            logger.info('Tokenizing training data to %s', tokenized_path)
            tokenizer.tokenize_directory(
                data_path,
                tokenized_path,
                src_tokenizer,
                tgt_tokenizer,
                config['source'],
                config['target'])
            data_path = tokenized_path

        return data_path, train_dir, num_samples, summary, metadata

    def _generate_vocabularies(self, config):
        raise NotImplementedError('vocabularies generation is not supported yet')

    def _summarize_data_distribution(self, build_info, distribution, parent_build_info=None):
        build_info['distribution'] = distribution
        return build_info


def load_config(config_arg):
    """Loads the configuration from a string, a file, or the standard input."""
    if config_arg.startswith('{'):
        return json.loads(config_arg)
    elif config_arg == '-':
        return json.loads(sys.stdin.read())
    else:
        with open(config_arg) as config_file:
            return json.load(config_file)

def merge_config(a, b):
    """Merges config b in a."""
    for k, v in six.iteritems(b):
        if k in a and isinstance(v, dict):
            merge_config(a[k], v)
        else:
            a[k] = v
    return a

def getenv(m):
    var = m.group(1)
    if var == 'TRAIN_DIR':
        var = 'CORPUS_DIR'
    return os.getenv(var, '')

def resolve_environment_variables(config):
    """Returns a new configuration with all environment variables replaced."""
    if isinstance(config, dict):    
        new_config = {}
        for k, v in six.iteritems(config):
            new_config[k] = resolve_environment_variables(v)
        return new_config
    elif isinstance(config, list):
        new_config = []
        for i in range (len(config)):
            new_config.append(resolve_environment_variables(config[i]))
        return new_config
    elif isinstance(config, six.string_types):
        return ENVVAR_RE.sub(lambda m: getenv(m), config)
    return config

def map_config_fn(config, fn):
    if isinstance(config, list):
        for i in xrange(len(config)):
            config[i] = map_config_fn(config[i], fn)
        return config
    elif isinstance(config, dict):
        for k, v in six.iteritems(config):
            config[k] = map_config_fn(v, fn)
        return config
    else:
        return fn(config)

def bundle_dependencies(objects, options):
    """Bundles additional resources in the model package."""
    def _map_fn(options):
        if isinstance(options, six.string_types):
            m = ENVVAR_ABS_RE.match(options)
            if m and "TRAIN_DIR" not in m.group(1):
                path = ENVVAR_RE.sub(
                    lambda m: os.getenv(m.group(1), ''), str(options))
                objects[m.group(2)] = path
                return '${MODEL_DIR}/%s' % m.group(2)
        return options
    return map_config_fn(options, _map_fn)

def extract_model_resources(objects, config):
    """Returns resources included in the model directory."""
    def _map_fn(config):
        if isinstance(config, six.string_types):
            m = ENVVAR_ABS_RE.match(config)
            if m and "MODEL_DIR" in m.group(1):
                objects[m.group(2)] = ENVVAR_RE.sub(
                    lambda m: os.getenv(m.group(1), ''), str(config))
        return config
    return map_config_fn(config, _map_fn)

def should_check_integrity(f):
    """Returns True if f should be checked for integrity."""
    return f not in ('README.md', 'checksum.md5', 'data')

def build_model_dir(model_dir, objects, config):
    """Prepares the model directory based on the model package."""
    if os.path.exists(model_dir):
        raise ValueError("model directory %s already exists" % model_dir)
    else:
        logger.info("Building model package in %s", model_dir)
    os.mkdir(model_dir)
    for target, source in six.iteritems(objects):
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(model_dir, target))
        else:
            shutil.copyfile(source, os.path.join(model_dir, target))
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file)
    objects['config.json'] = config_path
    if "description" in config:
        readme_path = os.path.join(model_dir, 'README.md')
        with open(readme_path, 'w') as readme_file:
            readme_file.write(config['description'])
        objects['README.md'] = readme_path
    md5 = md5files((k, v) for k, v in six.iteritems(objects) if should_check_integrity(k))
    with open(os.path.join(model_dir, "checksum.md5"), "w") as f:
        f.write(md5)

def check_model_dir(model_dir):
    """Compares model package MD5."""
    logger.info("Checking integrity of model package %s", model_dir)
    md5_file = os.path.join(model_dir, "checksum.md5")
    if not os.path.exists(md5_file):
        return True
    md5ref = None
    with open(md5_file, "r") as f:
        md5ref = f.read().strip()
    files = os.listdir(model_dir)
    md5check = md5files([(f, os.path.join(model_dir, f)) for f in files if should_check_integrity(f)])
    return md5check == md5ref

def fetch_model(storage, remote_model_path, model_path):
    """Downloads the remote model."""
    storage.get(remote_model_path, model_path, directory=True, check_integrity_fn=check_model_dir)
    os.environ['MODEL_DIR'] = model_path
