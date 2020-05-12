# -*- coding: utf-8 -*-
"""Shared logic and abstractions of frameworks."""

import os
import abc
import copy
import json
import time
import filecmp
import re
import six
import gzip
import shutil
import collections
import traceback

from nmtwizard.logger import get_logger
from nmtwizard import config as config_util
from nmtwizard import data as data_util
from nmtwizard import serving
from nmtwizard.preprocess import preprocess
from nmtwizard import utility


logger = get_logger(__name__)


@six.add_metaclass(abc.ABCMeta)
class Framework(utility.Utility):
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
        self._models_dir = os.getenv('MODELS_DIR')
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
        """Loads the model for serving.

        Frameworks could start a backend server or simply load the model from Python.

        Args:
          config: The run configuration.
          model_path: The path to the model to serve.
          gpuid: The GPU identifier.

        Returns:
          A tuple with the created process (if any) and a dictionary containing
          information to use the model (e.g. port number for a backend server).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_request(self, model_info, inputs, outputs=None, options=None):
        """Forwards a translation request to the model.

        Args:
          model_info: The information to reach the model, as returned by serve().
          inputs: A list of inputs.
          outputs: A list of (possibly partial) outputs.
          options: Additional translation options.

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
        parser_trans.add_argument('--add_bt_tag', default=False, action='store_true',
                                  help=('Add back-translation tag to the front of the generated output. ',
                                       'Refer to https://arxiv.org/pdf/1906.06442.pdf'))
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
        parser_serve.add_argument('--release_optimization_level', type=int, default=1,
                                  help=('Control the level of optimization applied to '
                                        'released models (for compatible frameworks). '
                                        '0 = no optimization, 1 = quantization.'))

        parser_preprocess = subparsers.add_parser('preprocess', help='Sample and preprocess corpus.')
        parser_preprocess.add_argument('--build_model', default=False, action='store_true',
                                       help='Preprocess data into a model.')
        parser.build_vocab = subparsers.add_parser('buildvocab', help='Build vocabularies.')
        self.parser = parser

    def exec_function(self, args):
        """Main entrypoint."""
        if self._config is None and self._model is None:
            self.parser.error('at least one of --config or --model options must be set')

        config = self._config or {}
        parent_model = self._model or config.get('model')
        if parent_model is not None and not self._stateless:
            # Download model locally and merge the configuration.
            remote_model_path = self._storage.join(self._model_storage_read, parent_model)
            model_path = os.path.join(self._models_dir, parent_model)
            model_config = utility.fetch_model(
                self._storage,
                remote_model_path,
                model_path,
                should_check_integrity)
            if 'modelType' not in model_config:
                if parent_model.endswith('_release'):
                    model_config['modelType'] = 'release'
                else:
                    model_config['modelType'] = 'checkpoint'
            config = config_util.update_config(
                copy.deepcopy(model_config), config, mode=args.config_update_mode)
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
                add_bt_tag=args.add_bt_tag,
                no_postprocess=args.no_postprocess)
        elif args.cmd == 'release':
            if not self._stateless and (parent_model is None or config['modelType'] != 'checkpoint'):
                raise ValueError('releasing requires a training checkpoint')
            if args.destination is None:
                args.destination = self._model_storage_write
            self.release_wrapper(
                config,
                model_path,
                self._image,
                storage=self._storage,
                destination=args.destination,
                optimization_level=args.optimization_level,
                gpuid=self._gpuid,
                push_model=not self._no_push)
        elif args.cmd == 'serve':
            if (not self._stateless
                and (parent_model is None
                     or config['modelType'] not in ('checkpoint', 'release'))):
                raise ValueError('serving requires a training checkpoint or a released model')
            if config['modelType'] == 'checkpoint':
                model_path = self.release_wrapper(
                    config,
                    model_path,
                    self._image,
                    local_destination=self._output_dir,
                    optimization_level=args.release_optimization_level,
                    gpuid=self._gpuid,
                    push_model=False)
                config = utility.load_model_config(model_path)
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
                    model_config=model_config,
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
        if parent_model_type == 'preprocess':
            data_dir = os.path.join(model_path, 'data')
            num_samples = config['sampling']['numSamples']
            samples_metadata = config['sampling']['samplesMetadata']
            tokens_to_add = {}
            del config['sampling']
            logger.info('Using preprocessed data from %s' % data_dir)
        else:
            data_dir, num_samples, distribution_summary, samples_metadata, tokens_to_add = (
                self._build_data(local_config))

        src_vocab_info, tgt_vocab_info, _ = self._get_vocabs_info(
            config,
            local_config,
            model_config=model_config,
            tokens_to_add=tokens_to_add)

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
        utility.build_model_dir(objects_dir, objects, config, should_check_integrity)
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
        objects, preprocess_config, vocab_config = self._generate_vocabularies(local_config)
        end_time = time.time()

        # Old tokenization configuration
        if isinstance(preprocess_config, dict):
            local_config["tokenization"] = utility.resolve_environment_variables(preprocess_config)
            config["tokenization"] = preprocess_config
        elif isinstance(preprocess_config, list):
            local_config["preprocess"] = utility.resolve_environment_variables(preprocess_config)
            config["preprocess"] = preprocess_config
        else:
            raise RuntimeError("Unknown preprocess configuration after buildvocab: \"{}\"".format(preprocess_config))

        local_config["vocabulary"] = utility.resolve_environment_variables(vocab_config)
        config["vocabulary"] = vocab_config
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
        utility.build_model_dir(objects_dir, objects, config, should_check_integrity)
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
                      add_bt_tag=False,
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

        self._set_preprocessor('inference', local_config)

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
                path_input_preprocessed = self._preprocess_file(path_input_unzipped)
                metadata = None
                if isinstance(path_input_preprocessed, tuple):
                    path_input_preprocessed, metadata = path_input_preprocessed
                translate_fn(local_config,
                             model_path,
                             path_input_preprocessed,
                             path_output,
                             gpuid=gpuid)
                path_postprocess_input = path_input_preprocessed
                if metadata is not None:
                    path_postprocess_input = (path_input_preprocessed, metadata)
                num_lines, num_tokens = file_stats(path_output)
                translated_lines += num_lines
                generated_tokens += num_tokens
                if not no_postprocess:
                    path_output = self._postprocess_file(path_postprocess_input, path_output)

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

                if add_bt_tag:
                    post_add_bt_tag(path_output)
                if path_output_is_zipped:
                    path_output = compress_file(path_output)

                storage.push(path_output, output)

                end_time = time.time()
                logger.info('Finished translation in %s seconds', str(end_time-start_time))
            except Exception as e:
                # Catch any exception to not impact other translations.
                filename = path_input if not isinstance(path_input, tuple) else path_input[0]
                logger.error("Translation of file %s failed with error:\n%s" % (
                    filename, traceback.format_exc()))
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
                        image,
                        storage=None,
                        local_destination=None,
                        destination=None,
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
        model_options = {}
        supported_features = config.get('supported_features')
        if supported_features is not None:
            model_options['supported_features'] = supported_features
        inference_options = config.get('inference_options')
        if inference_options is not None:
            schema = config_util.validate_inference_options(inference_options, config)
            model_options['json_schema'] = schema
        if model_options:
            options_path = os.path.join(self._output_dir, 'options.json')
            with open(options_path, 'w') as options_file:
                json.dump(model_options, options_file)
            objects[os.path.basename(options_path)] = options_path
        if local_destination is None:
            local_destination = self._models_dir
        objects_dir = os.path.join(local_destination, model_id)
        utility.build_model_dir(objects_dir, objects, config, should_check_integrity)
        if push_model:
            storage.push(objects_dir, storage.join(destination, model_id))
        return objects_dir

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
        outputs = self._generate_training_data(local_config)
        data_dir = outputs[0]

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
                              model_config=None,
                              push_model=True):
        logger.info('Starting preprocessing %s', model_id)
        start_time = time.time()

        local_config = self._finalize_config(config)
        data_dir, num_samples, distribution_summary, samples_metadata, tokens_to_add = (
            self._build_data(local_config))

        end_time = time.time()
        logger.info('Finished preprocessing %s in %s seconds', model_id, str(end_time-start_time))

        _, _, parent_dependencies = self._get_vocabs_info(
            config,
            local_config,
            model_config=model_config,
            tokens_to_add=tokens_to_add,
            keep_previous=True)

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
        # Forward other files from the parent model that are not tracked by the config.
        if model_path is not None:
            for f in os.listdir(model_path):
                if f not in objects and f not in parent_dependencies:
                    objects[f] = os.path.join(model_path, f)
        objects_dir = os.path.join(self._models_dir, model_id)
        utility.build_model_dir(objects_dir, objects, config, should_check_integrity)
        if push_model:
            storage.push(objects_dir, storage.join(model_storage, model_id))
        return {
            'num_sentences': build_info.get('sentenceCount')
        }

    def _get_vocabs_info(self,
                         config,
                         local_config,
                         model_config=None,
                         tokens_to_add=None,
                         keep_previous=False):
        if tokens_to_add is None:
            tokens_to_add = {}
        vocab_config = config.get('vocabulary', {})
        vocab_local_config = local_config.get('vocabulary', {})
        # For compatibility with old configurations
        tok_config = config.get('tokenization', {})
        tok_local_config = local_config.get('tokenization', {})
        joint_vocab = is_joint_vocab(vocab_config)
        parent_dependencies = {}
        if model_config:
            model_vocab_config = model_config.get('vocabulary', {})
            model_vocab_local_config = self._finalize_config(model_vocab_config)
            model_joint_vocab = is_joint_vocab(model_vocab_config)
            if joint_vocab != model_joint_vocab:
                raise ValueError("Changing joint vocabularies to split vocabularies "
                                 "(or vice-versa) is currently not supported.")
            if keep_previous:
                bundle_dependencies(
                    parent_dependencies,
                    copy.deepcopy(model_vocab_config),
                    copy.deepcopy(model_vocab_local_config))
        else:
            model_vocab_config = None
            model_vocab_local_config = None
        source_tokens_to_add = tokens_to_add.get('source') or []
        target_tokens_to_add = tokens_to_add.get('target') or []
        if joint_vocab:
            source_tokens_to_add = set(list(source_tokens_to_add) + list(target_tokens_to_add))
            target_tokens_to_add = source_tokens_to_add
        src_info = self._get_vocab_info(
            'source',
            vocab_config,
            vocab_local_config,
            tok_config,
            tok_local_config,
            model_config=model_vocab_config,
            local_model_config=model_vocab_local_config,
            tokens_to_add=source_tokens_to_add,
            keep_previous=keep_previous,
            joint_vocab=joint_vocab)
        tgt_info = self._get_vocab_info(
            'target',
            vocab_config,
            vocab_local_config,
            tok_config,
            tok_local_config,
            model_config=model_vocab_config,
            local_model_config=model_vocab_local_config,
            tokens_to_add=target_tokens_to_add,
            keep_previous=keep_previous,
            joint_vocab=joint_vocab)

        return src_info, tgt_info, parent_dependencies

    def _get_vocab_info(self,
                        side,
                        config,
                        local_config,
                        tok_config,
                        tok_local_config,
                        model_config=None,
                        local_model_config=None,
                        tokens_to_add=None,
                        keep_previous=False,
                        joint_vocab=False):
        if not config:
            return None
        opt = config.get(side)
        if opt is None or 'path' not in opt:
            return None
        local_opt = local_config[side]
        vocab_name = side if not joint_vocab else 'joint'

        current_basename = '%s-vocab.txt' % vocab_name
        current_vocab = self._convert_vocab(
            local_opt['path'], basename=current_basename)

        # First read previous_vocabulary if given.
        previous_basename = 'previous-%s-vocab.txt' % vocab_name
        previous_vocab = local_opt.get('previous_vocabulary')
        if previous_vocab is not None:
            previous_vocab = self._convert_vocab(
                previous_vocab, basename=previous_basename)
            del opt['previous_vocabulary']
            del local_opt['previous_vocabulary']
        # Otherwise check if the vocabulary is different than the parent model.
        elif model_config is not None:
            model_opt = model_config[side]
            local_model_opt = local_model_config[side]
            previous_vocab = self._convert_vocab(
                local_model_opt['path'], basename=previous_basename)
            vocab_changed = not filecmp.cmp(previous_vocab, current_vocab)
            if vocab_changed:
                if not opt.get('replace_vocab', False):
                    raise ValueError('%s vocabulary has changed but replace_vocab is not set.'
                                     % vocab_name.capitalize())
                if keep_previous:
                    opt['previous_vocabulary'] = os.path.join("${MODEL_DIR}", previous_basename)
                    local_opt['previous_vocabulary'] = local_model_opt['path']
            else:
                os.remove(previous_vocab)
                previous_vocab = None

        if 'replace_vocab' in opt:
            del opt['replace_vocab']
            del local_opt['replace_vocab']

        if tokens_to_add:
            new_filename = next_filename_version(os.path.basename(local_opt["path"]))
            new_vocab = os.path.join(self._data_dir, new_filename)
            shutil.copy(local_opt["path"], new_vocab)
            with open(new_vocab, "a") as vocab:
                for token in tokens_to_add:
                    vocab.write("%s\n" % token)
            if previous_vocab is None:
                previous_vocab = current_vocab
                if keep_previous:
                    opt['previous_vocabulary'] = opt['path']
                    local_opt['previous_vocabulary'] = local_opt['path']
            current_vocab = self._convert_vocab(
                new_vocab, basename="updated-%s-vocab.txt" % vocab_name)
            opt["path"] = new_vocab
            local_opt["path"] = new_vocab

            # For compatibility with old configurations
            if tok_config:
                tok_config[side]['vocabulary'] = new_vocab
            if tok_local_config:
                tok_local_config[side]['vocabulary'] = new_vocab

        VocabInfo = collections.namedtuple('VocabInfo', ['current', 'previous'])
        return VocabInfo(current=current_vocab, previous=previous_vocab)

    def _serving_state(self, config):
        state = {}
        self._set_preprocessor('inference', config)
        state['preprocessor'] = self._preprocessor
        state['postprocessor'] = self._postprocessor
        return state

    def _preprocess_input(self, state, source, target, config):
        metadata = None
        if not isinstance(source, list) and not isinstance(target, list):
            preprocessor = state.get('preprocessor')

            if target is not None:
                preprocess_input = (source, target)
            else :
                preprocess_input = source

            preprocess_output = preprocessor.process_input(preprocess_input)
            if preprocess_output == preprocess_input: # no preprocess is done
                (source, metadata), target = output
            else:
                source = source.split()
                if target is not None:
                    target = target.split()

        return source, target, metadata

    def _postprocess_output(self, state, source, target, config):
        if not isinstance(target, list):
            return target
        postprocessor = state.get('postprocessor')
        postprocess_input = (source,target)
        postprocess_output = postprocessor.process_input(postprocess_input)
        if postprocess_output == postprocess_input: # no postprocess is done
            return ' '.join(target)
        return postprocess_output

    def _preprocess_file(self, preprocess_input):
        return self._preprocessor.process_file(preprocess_input)

    def _postprocess_file(self, source, target):
        return self._postprocessor.process_file((source, target))

    def _convert_vocab(self, vocab_file, basename=None):
        if basename is None:
            basename = os.path.basename(vocab_file)
        converted_vocab_file = os.path.join(self._data_dir, basename)
        with open(vocab_file, 'rb') as vocab, open(converted_vocab_file, 'wb') as converted_vocab:
            header = True
            index = 0
            for line in vocab:
                if header and line.startswith(b'#'):
                    continue
                header = False
                token = line.strip().split()[0]
                self._map_vocab_entry(index, token, converted_vocab)
                index += 1
        return converted_vocab_file

    def _build_data(self, config):
        data_dir, train_dir, num_samples, distribution_summary, samples_metadata = (
            self._generate_training_data(config))
        if num_samples == 0:
            raise RuntimeError('data sampling generated 0 sentences')
        if distribution_summary is not None:
            tokens_to_add = distribution_summary.get("tokens_to_add")
        else:
            tokens_to_add = None
        if not self._support_multi_training_files:
            data_dir = self._merge_multi_training_files(
                data_dir, train_dir, config['source'], config['target'])
        return data_dir, num_samples, distribution_summary, samples_metadata, tokens_to_add

    def _merge_multi_training_files(self, data_path, train_dir, source, target):
        merged_dir = os.path.join(self._data_dir, 'merged')
        if not os.path.exists(merged_dir):
            os.mkdir(merged_dir)
        merged_path = os.path.join(merged_dir, train_dir)
        logger.info('Merging training data to %s/train.{%s,%s}',
                    merged_path, source, target)
        data_util.merge_files_in_directory(data_path, merged_path, source, target)
        return merged_path

    def _set_preprocessor(self, cmd, config):
        if cmd == 'train':
            self._preprocessor = preprocess.TrainingProcessor(config, self._corpus_dir, self._data_dir)
        elif cmd == 'inference':
            self._preprocessor = preprocess.InferenceProcessor(config)
            self._postprocessor = preprocess.InferenceProcessor(config, postprocess=True)
        else:
            raise RuntimeError('Invalid preprocess type: %s.' % cmd)

    def _generate_training_data(self, config):
        self._set_preprocessor('train', config)
        return self._preprocessor.generate_preprocessed_data()

    def _generate_vocabularies(self, config):
        self._set_preprocessor('train', config)
        return self._preprocessor.generate_vocabularies()

    def _summarize_data_distribution(self, build_info, distribution, parent_build_info=None):
        build_info['distribution'] = distribution
        if distribution is not None:
            cum_sent_count = 0
            if parent_build_info is not None:
                cum_sent_count = parent_build_info.get('cumSentenceCount')
            sent_count = sum(v.get('lines_filtered', 0) for v in six.itervalues(distribution))
            build_info['sentenceCount'] = sent_count
            build_info['cumSentenceCount'] = (
                cum_sent_count + sent_count if cum_sent_count is not None else None)
        return build_info


    def _finalize_config(self, config, training=True):
        config_util.old_to_new_config(config)
        config = utility.resolve_environment_variables(config, training=training)
        config = self._upgrade_data_config(config, training=training)
        config = utility.resolve_remote_files(config, self._shared_dir, self._storage)
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


def bundle_dependencies(objects, config, local_config):
    """Bundles additional resources in the model package."""
    if local_config is None:
        return config
    if isinstance(config, list):
        for i, _ in enumerate(config):
            config[i] = bundle_dependencies(objects, config[i], local_config[i])
        return config
    elif isinstance(config, dict):
        for k, v in six.iteritems(config):
            if k in ('sample_dist', 'build'):
                continue
            config[k] = bundle_dependencies(objects, v, local_config.get(k))
        return config
    else:
        if isinstance(config, six.string_types):
            if os.path.isabs(config) and os.path.exists(config):
                filename = os.path.basename(config)
            else:
                match = utility.ENVVAR_ABS_RE.match(config)
                if match and "TRAIN" not in match.group(1):
                    filename = match.group(2)
                else:
                    filename = None
            if filename is not None:
                objects[filename] = local_config
                return '${MODEL_DIR}/%s' % filename
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

def post_add_bt_tag(path_input):
    const_bt_tag = "｟mrk_bt｠"

    path_input_new = '%s.raw' % path_input
    os.rename(path_input, path_input_new)

    with open(path_input_new, 'r') as f_in, open(path_input, 'w') as f_out:
        for line in f_in:
            f_out.write('%s %s' % (const_bt_tag, line))

def next_filename_version(filename):
    regexp = re.compile(r'^(.+)\.v([0-9]+)$')
    match = regexp.match(filename)
    if match:
        filename = match.group(1)
        version = int(match.group(2)) + 1
    else:
        version = 2
    return '%s.v%d' % (filename, version)

def is_joint_vocab(vocabulary_config):
    source = vocabulary_config.get('source', {})
    target = vocabulary_config.get('target', {})
    return source.get('path') == target.get('path')
