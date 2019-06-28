"""Shared logic and abstractions of frameworks."""

import os
import abc
import copy
import json
import argparse
import time
import filecmp
import re
import six
import gzip
import shutil

from nmtwizard.logger import get_logger
from nmtwizard.sampler import sample
from nmtwizard.utility import Utility, merge_config
from nmtwizard.utility import resolve_environment_variables, resolve_remote_files
from nmtwizard.utility import build_model_dir, fetch_model
from nmtwizard.utility import ENVVAR_RE, ENVVAR_ABS_RE
from nmtwizard import config as config_util
from nmtwizard import serving
from nmtwizard import tokenizer
from nmtwizard import preprocess


logger = get_logger(__name__)


@six.add_metaclass(abc.ABCMeta)
class Framework(Utility):
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

        super(Framework, self).__init__()

        self._stateless = stateless
        self._support_multi_training_files = support_multi_training_files
        self._models_dir = os.getenv('MODELS_DIR', '/root/models')
        if not stateless and not os.path.exists(self._models_dir):
            os.makedirs(self._models_dir)

    @property
    def name(self):
        return "NMT framework"

    @abc.abstractmethod
    def train(self,
              config,
              src_file,
              tgt_file,
              src_vocab_info,
              tgt_vocab_info,
              align_file=None,
              model_path=None,
              gpuid=0):
        """Trains for one epoch.

        Args:
          config: The run configuration.
          src_file: The local path to the preprocessed (if any) source file.
          tgt_file: The local path to the preprocessed (if any) target file.
          align_file: The local path to the alignment file (between source and target).
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

    def translate_as_release(self,
                             config,
                             model_path,
                             input,
                             output,
                             optimization_level=None,
                             gpuid=0):
        """Translates a file in release condition.

        This is useful when the released model contains optimizations that
        change the translation result (e.g. quantization).

        By default, assumes that the released model does not change the results
        and calls the standard translation method. Otherwise, the framework
        should release the given checkpoint and adapt the translation logic.

        Args:
          config: The run configuration.
          model_path: The path to the checkpoint to release.
          input: The local path to the preprocessed (if any) source file.
          output: The local path to the file that should contain the translation.
          optimization_level: An integer defining the level of optimization to
            apply to the released model. 0 = no optimization, 1 = quantization.
          gpuid: The GPU identifier.
        """
        return self.trans(config, model_path, input, output, gpuid=gpuid)

    @abc.abstractmethod
    def release(self, config, model_path, optimization_level=None, gpuid=0):
        """Releases a model for serving.

        Args:
          config: The run configuration.
          model_path: The path to the model to release.
          optimization_level: An integer defining the level of optimization to
            apply to the released model. 0 = no optimization, 1 = quantization.
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
                align_file=os.path.join(data_dir, 'train.align'),
                model_path=model_path,
                gpuid=gpuid)

    def declare_arguments(self, parser):
        subparsers = parser.add_subparsers(help='Run type', dest='cmd')
        parser_train = subparsers.add_parser('train', help='Run a training.')

        parser_trans = subparsers.add_parser('trans', help='Run a translation.')
        parser_trans.add_argument('-i', '--input', required=True, nargs='+',
                                  help='Input files')
        parser_trans.add_argument('-o', '--output', required=True, nargs='+',
                                  help='Output files')
        parser_trans.add_argument('--copy_source', default=False, action='store_true',
                                  help=('Copy source files on same storage and same name '
                                        'as the outputs (useful for chaining back-translation '
                                        'trainings). By default, the original source file is '
                                        'copied but when --no_postprocess is used, the '
                                        'preprocessed source file is copied instead.'))
        parser_trans.add_argument('--as_release', default=False, action='store_true',
                                  help='Translate from a released model.')
        parser_trans.add_argument('--release_optimization_level', type=int, default=1,
                                  help=('Control the level of optimization applied to '
                                        'released models (for compatible frameworks). '
                                        '0 = no optimization, 1 = quantization.'))
        parser_trans.add_argument('--no_postprocess', default=False, action='store_true',
                                  help='Do not apply postprocessing on the target files.')

        parser_release = subparsers.add_parser('release', help='Release a model for serving.')
        parser_release.add_argument('-d', '--destination', default=None,
                                    help='Released model storage (defaults to the model storage).')
        parser_release.add_argument('-o', '--optimization_level', type=int, default=1,
                                    help=('Control the level of optimization applied to '
                                          'released models (for compatible frameworks). '
                                          '0 = no optimization, 1 = quantization.'))

        parser_serve = subparsers.add_parser('serve', help='Serve a model.')
        parser_serve.add_argument('-hs', '--host', default="0.0.0.0",
                                  help='Serving hostname.')
        parser_serve.add_argument('-p', '--port', type=int, default=4000,
                                  help='Serving port.')

        parser_preprocess = subparsers.add_parser('preprocess', help='Sample and preprocess corpus.')
        parser_preprocess.add_argument('--build_model', default=False, action='store_true',
                                       help='Preprocess data into a model.')
        parser.build_vocab = subparsers.add_parser('buildvocab', help='Build vocabularies.')
        self.parser = parser

    def exec_function(self, args):
        """Main entrypoint."""
        if self._config is None and self._model is None:
            self.parser.error('at least one of --config or --model options must be set')

        config = {} if self._config is None else self._config

        if not self._stateless and \
           (args.cmd != 'preprocess' or args.build_model) and \
           not args.no_push and \
           (self._model_storage_write is None or self._model_storage_write is None):
            self.parser.error('Missing model storage argument')

        parent_model = self._model or config.get('model')

        if parent_model is not None and not self._stateless:
            # Download model locally and merge the configuration.
            remote_model_path = self._storage.join(self._model_storage_read, parent_model)
            model_path = os.path.join(self._models_dir, parent_model)
            fetch_model(self._storage, remote_model_path, model_path, should_check_integrity)
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
            if parent_model is not None and config['modelType'] not in ('checkpoint', 'base', 'preprocess'):
                raise ValueError('cannot train from a model that is not a training checkpoint, '
                                 'a base model, or a preprocess model')
            return self.train_wrapper(
                self._task_id,
                config,
                self._storage,
                self._model_storage_write,
                self._image,
                parent_model=parent_model,
                model_path=model_path,
                model_config=model_config,
                gpuid=self._gpuid,
                push_model=not self._no_push)
        elif args.cmd == 'buildvocab':
            self.build_vocab(
                self._task_id,
                config,
                self._storage,
                self._model_storage_write,
                self._image,
                push_model=not self._no_push)
        elif args.cmd == 'trans':
            if not self._stateless and (parent_model is None or config['modelType'] != 'checkpoint'):
                raise ValueError('translation requires a training checkpoint')
            return self.trans_wrapper(
                config,
                model_path,
                self._storage,
                args.input,
                args.output,
                as_release=args.as_release,
                release_optimization_level=args.release_optimization_level,
                gpuid=self._gpuid,
                copy_source=args.copy_source,
                no_postprocess=args.no_postprocess)
        elif args.cmd == 'release':
            if not self._stateless and (parent_model is None or config['modelType'] != 'checkpoint'):
                raise ValueError('releasing requires a training checkpoint')
            if args.destination is None:
                args.destination = self._model_storage_write
            self.release_wrapper(
                config,
                model_path,
                self._storage,
                self._image,
                args.destination,
                optimization_level=args.optimization_level,
                gpuid=self._gpuid,
                push_model=not self._no_push)
        elif args.cmd == 'serve':
            if not self._stateless and (parent_model is None or config['modelType'] != 'release'):
                raise ValueError('serving requires a released model')
            self.serve_wrapper(
                config, model_path, args.host, args.port, gpuid=self._gpuid)
        elif args.cmd == 'preprocess':
            if not args.build_model:
                self.preprocess(config, self._storage)
            else:
                if parent_model is not None and config['modelType'] not in ('checkpoint', 'base'):
                    raise ValueError('cannot preprocess from a model that is not a training '
                                     'checkpoint or a base model')
                return self.preprocess_into_model(
                    self._task_id,
                    config,
                    self._storage,
                    self._model_storage_write,
                    self._image,
                    parent_model=parent_model,
                    model_path=model_path,
                    push_model=not self._no_push)

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

        parent_model_type = config.get('modelType') if model_path is not None else None

        local_config = self._finalize_config(config)
        local_model_config = (
            self._finalize_config(model_config)
            if model_config is not None else None)

        src_vocab_info = self._get_vocab_info(
            'source', config, local_config, model_config=local_model_config)
        tgt_vocab_info = self._get_vocab_info(
            'target', config, local_config, model_config=local_model_config)

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
        bundle_dependencies(objects, config, local_config)
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config, should_check_integrity)
        if push_model:
            storage.push(objects_dir, storage.join(model_storage, model_id))
        return {
            'num_sentences': config['build'].get('sentenceCount')
        }

    def build_vocab(self,
                    model_id,
                    config,
                    storage,
                    model_storage,
                    image,
                    push_model=True):
        start_time = time.time()
        local_config = self._finalize_config(config)
        objects, tokenization_config = self._generate_vocabularies(local_config)
        end_time = time.time()

        local_config['tokenization'] = resolve_environment_variables(tokenization_config)
        config['tokenization'] = tokenization_config
        config['model'] = model_id
        config['modelType'] = 'base'
        config['imageTag'] = image
        config['build'] = {
            'containerId': os.uname()[1],
            'endDate': end_time,
            'startDate': start_time
        }

        bundle_dependencies(objects, config, local_config)
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config, should_check_integrity)
        if push_model:
            storage.push(objects_dir, storage.join(model_storage, model_id))

    def trans_wrapper(self,
                      config,
                      model_path,
                      storage,
                      inputs,
                      outputs,
                      as_release=False,
                      release_optimization_level=None,
                      gpuid=0,
                      copy_source=False,
                      no_postprocess=False):
        if len(inputs) != len(outputs):
            raise ValueError("Mismatch of input/output files number, got %d and %d" % (
                len(inputs), len(outputs)))

        def translate_fn(*args, **kwargs):
            if as_release:
                return self.translate_as_release(
                    *args, optimization_level=release_optimization_level, **kwargs)
            else:
                return self.trans(*args, **kwargs)

        local_config = self._finalize_config(config, training=False)
        failed_translation = 0
        translated_lines = 0
        generated_tokens = 0

        for input, output in zip(inputs, outputs):
            try:
                path_input = os.path.join(self._data_dir, storage.split(input)[-1])
                path_output = os.path.join(self._output_dir, storage.split(output)[-1])
                storage.get_file(input, path_input)

                path_input_unzipped = decompress_file(path_input)
                path_input_unzipped_parts = path_input_unzipped.split('.')
                if copy_source and len(path_input_unzipped_parts) < 2:
                    raise ValueError("In copy_source mode, input files should have language suffix")
                path_output_is_zipped = False
                if path_output.endswith(".gz"):
                    path_output_is_zipped = True
                    path_output = path_output[:-3]
                path_output_parts = path_output.split('.')
                if copy_source and len(path_output_parts) < 2:
                    raise ValueError("In copy_source mode, output files should have language suffix")

                logger.info('Starting translation %s to %s', path_input, path_output)
                start_time = time.time()
                path_input_preprocessed = self._preprocess_file(local_config, path_input_unzipped)
                metadata = None
                if isinstance(path_input_preprocessed, tuple):
                    path_input_preprocessed, metadata = path_input_preprocessed
                translate_fn(local_config,
                             model_path,
                             path_input_preprocessed,
                             path_output,
                             gpuid=gpuid)
                if metadata is not None:
                    path_input_preprocessed = (path_input_preprocessed, metadata)
                num_lines, num_tokens = file_stats(path_output)
                translated_lines += num_lines
                generated_tokens += num_tokens
                if not no_postprocess:
                    path_output = self._postprocess_file(
                        local_config, path_input_preprocessed, path_output)

                if copy_source:
                    copied_input = output
                    copied_input_parts = copied_input.split('.')
                    source_to_copy = (
                        path_input_unzipped if not no_postprocess else path_input_preprocessed)
                    if path_output_is_zipped:
                        copied_input_parts[-2] = path_input_unzipped_parts[-1]
                        if path_input_unzipped == path_input:
                            path_input = compress_file(source_to_copy)
                    else:
                        copied_input_parts[-1] = path_input_unzipped_parts[-1]
                        path_input = source_to_copy

                    storage.push(path_input, ".".join(copied_input_parts))

                if path_output_is_zipped:
                    path_output = compress_file(path_output)

                storage.push(path_output, output)

                end_time = time.time()
                logger.info('Finished translation in %s seconds', str(end_time-start_time))
            except Exception as e:
                # Catch any exception to not impact other translations.
                filename = path_input if not isinstance(path_input, tuple) else path_input[0]
                logger.error("Translation of %s failed with error \"%s: %s\"" % (
                    filename, e.__class__.__name__, str(e)))
                logger.warning("Skipping translation of %s" % filename)
                failed_translation += 1

        if failed_translation == len(inputs):
            raise RuntimeError("All translation failed, see error logs")
        return {
            'num_sentences': translated_lines,
            'num_tokens': generated_tokens
        }

    def release_wrapper(self,
                        config,
                        model_path,
                        storage,
                        image,
                        destination,
                        optimization_level=None,
                        gpuid=0,
                        push_model=True):
        local_config = self._finalize_config(config, training=False)
        objects = self.release(
            local_config,
            model_path,
            optimization_level=optimization_level,
            gpuid=gpuid)
        bundle_dependencies(objects, config, local_config)
        model_id = config['model'] + '_release'
        config['model'] = model_id
        config['modelType'] = 'release'
        config['imageTag'] = image
        for name in ("parent_model", "build", "data"):
            if name in config:
                del config[name]
        inference_options = config.get('inference_options')
        if inference_options is not None:
            schema = config_util.validate_inference_options(inference_options, config)
            options_path = os.path.join(self._output_dir, 'options.json')
            with open(options_path, 'w') as options_file:
                json.dump(schema, options_file)
            objects[os.path.basename(options_path)] = options_path
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config, should_check_integrity)
        if push_model:
            storage.push(objects_dir, storage.join(destination, model_id))

    def serve_wrapper(self, config, model_path, host, port, gpuid=0):
        local_config = self._finalize_config(config, training=False)
        serving.start_server(
            host,
            port,
            local_config,
            self._serving_state(local_config),
            lambda: self.serve(local_config, model_path, gpuid=gpuid),
            self._preprocess_input,
            self.forward_request,
            self._postprocess_output)

    def preprocess(self, config, storage):
        logger.info('Starting preprocessing data ')
        start_time = time.time()

        local_config = self._finalize_config(config)
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

        local_config = self._finalize_config(config)
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
        bundle_dependencies(objects, config, local_config)
        # Forward other files from the parent model.
        if model_path is not None:
            for f in os.listdir(model_path):
                if f not in objects:
                    objects[f] = os.path.join(model_path, f)
        objects_dir = os.path.join(self._models_dir, model_id)
        build_model_dir(objects_dir, objects, config, should_check_integrity)
        if push_model:
            storage.push(objects_dir, storage.join(model_storage, model_id))
        return {
            'num_sentences': build_info.get('sentenceCount')
        }

    def _get_vocab_info(self, side, config, local_config, model_config=None):
        if config.get('tokenization', {}).get(side, {}).get('vocabulary') is None:
            return None
        current_vocab = self._convert_vocab(
            local_config['tokenization'][side]['vocabulary'],
            basename='%s-vocab.txt' % side)
        model_vocab = None
        vocab_changed = False
        if model_config is not None:
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

    def _preprocess_input(self, state, input, config):
        if isinstance(input, list):
            tokens = input
        elif 'src_tokenizer' in state:
            input = input.encode('utf-8')
            tokens, _ = state['src_tokenizer'].tokenize(input)
        else:
            tokens = input.split()
        return tokens

    def _postprocess_output(self, state, source, target, config):
        if not isinstance(target, list):
            text = target
        elif 'tgt_tokenizer' in state:
            output = [out.encode('utf-8') for out in target]
            text = state['tgt_tokenizer'].detokenize(output)
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
        return preprocess.generate_preprocessed_data(config, self._corpus_dir, self._data_dir)

    def _generate_vocabularies(self, config):
        raise NotImplementedError('vocabularies generation is not supported yet')

    def _summarize_data_distribution(self, build_info, distribution, parent_build_info=None):
        build_info['distribution'] = distribution
        if distribution is not None:
            cum_sent_count = 0
            if parent_build_info is not None:
                cum_sent_count = parent_build_info.get('cumSentenceCount')
            sent_count = sum(v.get('linefiltered', 0) for v in six.itervalues(distribution))
            build_info['sentenceCount'] = sent_count
            build_info['cumSentenceCount'] = (
                cum_sent_count + sent_count if cum_sent_count is not None else None)
        return build_info

    def _finalize_config(self, config, training=True):
        config = resolve_environment_variables(config, training=training)
        config = self._upgrade_data_config(config, training=training)
        config = resolve_remote_files(config, self._shared_dir, self._storage)
        return config

    def _upgrade_data_config(self, config, training=True):
        if not training or 'data' not in config or 'sample_dist' not in config['data']:
            return config
        data = config['data']
        if 'train_dir' in data:
            train_dir = data['train_dir']
            del data['train_dir']
        else:
            train_dir = 'train'
        basedir = os.path.join(self._corpus_dir, train_dir)
        for dist in data['sample_dist']:
            if not self._storage.is_managed_path(dist['path']) and not os.path.isabs(dist['path']):
                dist['path'] = os.path.join(basedir, dist['path'])
        return config


def map_config_fn(config, fn):
    if isinstance(config, list):
        for i, _ in enumerate(config):
            config[i] = map_config_fn(config[i], fn)
        return config
    elif isinstance(config, dict):
        for k, v in six.iteritems(config):
            config[k] = map_config_fn(v, fn)
        return config
    else:
        return fn(config)

def bundle_dependencies(objects, config, local_config):
    """Bundles additional resources in the model package."""
    if isinstance(config, list):
        for i, _ in enumerate(config):
            config[i] = bundle_dependencies(objects, config[i], local_config[i])
        return config
    elif isinstance(config, dict):
        for k, v in six.iteritems(config):
            if k in ('sample_dist',):
                continue
            local_v = local_config.get(k) if local_config is not None else None
            config[k] = bundle_dependencies(objects, v, local_v)
        return config
    else:
        if isinstance(config, six.string_types):
            m = ENVVAR_ABS_RE.match(config)
            if m and "TRAIN" not in m.group(1):
                objects[m.group(2)] = local_config
                return '${MODEL_DIR}/%s' % m.group(2)
        return config

def should_check_integrity(f):
    """Returns True if f should be checked for integrity."""
    return f not in ('README.md', 'TRAINING_LOG', 'checksum.md5', 'data') and not f.startswith('.')

def file_stats(path):
    num_lines = 0
    num_tokens = 0
    with open(path, "rb") as f:
        for line in f:
            num_lines += 1
            num_tokens += len(line.strip().split())
    return num_lines, num_tokens

def compress_file(path_input):
    path_input_new = path_input
    if not path_input.endswith(".gz"):
        logger.info('Starting gzip %s', path_input)
        path_input_new += ".gz"
        with open(path_input, 'rb') as f_in, gzip.open(path_input_new, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return path_input_new

def decompress_file(path_input):
    path_input_new = path_input
    if path_input.endswith(".gz"):
        logger.info('Starting unzip %s', path_input)
        path_input_new = path_input[:-3]
        with gzip.open(path_input, 'rb') as f_in, open(path_input_new, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return path_input_new
