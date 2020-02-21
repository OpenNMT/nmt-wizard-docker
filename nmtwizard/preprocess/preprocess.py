"""Functions for corpus preprocessing."""

import os
import sys

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import consumer
from nmtwizard.preprocess import loader
from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess import sampler
from nmtwizard.preprocess import tokenizer

logger = get_logger(__name__)

def _get_tok_configs(config):
    tok_configs = []
    preprocess_config = config.get("preprocess")
    if preprocess_config is not None:
        for operator_config in preprocess_config:
            if prepoperator.get_operator_type(operator_config) == "tokenization":
                tok_configs.append(prepoperator.get_operator_params(operator_config))
    return tok_configs


class Processor(object):

    def process(self, loader, consumer):
        lines_num = 0

        # TODO V2 : parallelization
        for tu_batch in loader():
            tu_batch = self._pipeline(tu_batch)
            consumer(tu_batch)
            lines_num += len(tu_batch[0])

        return lines_num


class TrainingProcessor(Processor):

    def __init__(self, config, corpus_dir, data_dir):
        self._config = config
        self._corpus_dir = corpus_dir
        self._data_dir = data_dir

    def _set_pipeline(self, preprocess_exit_step=None):
        self._pipeline = prepoperator.TrainingPipeline(self._config, preprocess_exit_step)


    def generate_preprocessed_data(self, result='preprocess', preprocess_exit_step=None):

        # TODO V2 : annotations
        # TODO V2 : file-specific rules/extra

        # For backward compatibility with old relative path configurations.
        train_dir = 'train'
        if 'data' in self._config :
            if 'train_dir' in self._config['data']:
                train_dir = self._config['data']['train_dir']
        else :
            logger.warning('No \'data\' field in configuration, \
                            default corpus directory and all corpora are used.)')

        # Default data path.
        data_path = os.path.join(self._corpus_dir, train_dir)

        num_samples = None
        summary = None
        metadata = None

        # If some sampling OR preprocessing is applied, change result directory.
        if 'data' in self._config or 'preprocess' in self._config:

            result_dir = os.path.join(self._data_dir, result)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            if not os.path.isdir(result_dir):
                raise RuntimeError('%s is not a directory' % result_dir)
            logger.info('Generating data to %s', result_dir)

            # Sample files and write information to a special file structure.
            all_files, summary, metadata = sampler.sample(self._config, data_path)

            num_samples = 0

            # Default batch size is the whole sample size.
            batch_size = sys.maxsize
            if 'preprocess' in self._config and 'batch_size' in self._config['preprocess'] :
                batch_size = config['preprocess']['batch_size']

            sampler_consumer = consumer.make_consumer(self._config, result_dir, result, preprocess_exit_step)
            self._set_pipeline(preprocess_exit_step)

            for f in all_files:
                lines_filtered = 0
                if f.lines_kept :
                    sampler_loader=loader.SamplerFileLoader(f, batch_size)
                    if hasattr(sampler_consumer, "open_files"):
                        sampler_consumer.open_files(f)
                    lines_filtered = self.process(sampler_loader, sampler_consumer)
                    sampler_loader.close_files()
                    if hasattr(sampler_consumer, "close_files"):
                        sampler_consumer.close_files()

                    if lines_filtered != f.lines_kept:
                        num_samples += lines_filtered
                        summary[f.base_name]["lines_filtered"] = lines_filtered
                    else:
                        num_samples += f.lines_kept
                        summary[f.base_name]["lines_filtered"] = f.lines_kept

                    if hasattr(sampler_consumer, "finalize"):
                        sampler_consumer.finalize(self._config)

            data_path = result_dir

        return data_path, train_dir, num_samples, summary, metadata


    def _generate_models(self, tokenization_step, option):

        build_option = "build_" + option

        tok_config = self._config['preprocess'][tokenization_step]

        opt_multi = tok_config.get('multi', {}).get(build_option)
        opt_source = tok_config.get('source', {}).get(build_option)
        opt_target = tok_config.get('target', {}).get(build_option)

        if not opt_multi and not (opt_source and opt_target):
            logger.warning('No \'%s\' option specified, exit without preprocessing.' % build_option)
            return

        if (opt_multi and opt_source) or \
           (opt_multi and opt_target) :
            raise RuntimeError('Cannot specify \'%s\' for both \'multi\' and either \'source\' or \'target\'.' % build_option)

        # Generate preprocessed sentences and feed them to subword learners or to vocabularies.
        self.generate_preprocessed_data(option, preprocess_exit_step=tokenization_step)


    def generate_vocabularies(self):

        # Generate vocabularies and subword models for each tokenization block.
        tok_configs = _get_tok_configs(self._config)

        if not tok_configs:
            raise RuntimeError('No \'tokenization\' operator in preprocess configuration, cannot build vocabularies.)')

        for i, tok_config in enumerate(tok_configs):
            if ('source' not in tok_config or 'target' not in tok_config) and 'multi' not in tok_config :
                raise RuntimeError('Each \'tokenization\' operator should contain \
                                    either both \'source\' and \'target\' fields \
                                or \'multi\' field.')

            for side in tok_config:
                build_vocab = tok_config[side].get('build_vocabulary')
                if build_vocab:
                    if tok_config[side].get('vocabulary_path', {}):
                        raise RuntimeError('Cannot build vocabulary if \'%s\' vocabulary path is already specified.' % side)
                    if i == len(tok_configs)-1 and self._config.get('vocabulary',{}).get(side,{}).get('path'):
                        raise RuntimeError('Cannot build vocabulary for final tokenization if \'%s\' vocabulary path for model is already specified.' % side)
                    if not build_vocab.get('size'):
                        raise RuntimeError('\'size\' option is mandatory to build vocabulary for \'%s\'.' % side)

            self._generate_models(i, 'subword')

            self._generate_models(i, 'vocabulary')

            # Use vocabulary from final tokenization as vocabulary for translation framework.
            if i == len(tok_configs)-1:
                for side in tok_config:
                    if side == 'source' or side == 'target':
                        if 'vocabulary' not in self._config:
                            self._config['vocabulary'] = {}
                        if side not in self._config['vocabulary']:
                            self._config['vocabulary'][side] = {}
                        self._config['vocabulary'][side]['path'] = tok_config[side]['vocabulary_path']
                        # Only keep 'vocabulary_path' option for final tokenization if explicitly specified.
                        if not tok_config[side].get('use_vocab_in_tok', False):
                            del tok_config[side]['vocabulary_path']

        preprocess_config = None
        if "preprocess" in self._config:
            preprocess_config = self._config["preprocess"]

        vocab_config = None
        if "vocabulary" in self._config:
            vocab_config = self._config["vocabulary"]

        # TODO V2 : why we use a copy here ?
        return {}, preprocess_config, vocab_config


class InferenceProcessor(Processor):

    def __init__(self, config):
        self._pipeline = prepoperator.InferencePipeline(config)


    def process_input(self, input):
        """Processes one translation example at inference.

              In preprocess:
                 input is one single-part source as raw string
                 output is one (source, metadata), where source is tokenized and possibly multipart.

             In postprocess:
                 input is ((source, metadata), target), where source and target are tokenized and possibly multipart
                 outputs is one single-part postprocessed target."""

        basic_loader = loader.BasicLoader(input, self._pipeline.start_state)
        basic_writer = consumer.BasicWriter()
        self.process(basic_loader,
                     basic_writer)

        return basic_writer.output


    def process_file(self, input_files):
        """Process translation file at inference.

              In preprocess:
                 input is a file with raw sources
                 output is (source, metadata), where source is a file with tokenized and possibly multipart sources.
              In postprocess:
                 input is ((source, metadata), target), where source and target are files with tokenized and multi-part data
                 output is a file with postprocessed single-part targets."""

        # TODO :  can this file be compressed ?
        if isinstance(input_files, tuple):
            output_file = "%s.detok" % input_files[-1]
        else:
            output_file = "%s.tok" % input_files

        file_loader = loader.FileLoader(input_files, self._pipeline.start_state)
        file_consumer = consumer.FileWriter(output_file)

        self.process(file_loader, file_consumer)

        file_loader.close_files()
        file_consumer.close_files()

        if file_consumer.metadata:
            output_file = (output_file, file_consumer.metadata)

        return output_file


class Postprocessor(InferenceProcessor):

    def __init__(self, config):
        self._pipeline = prepoperator.PostprocessPipeline(config)
