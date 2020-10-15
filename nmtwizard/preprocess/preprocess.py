"""Functions for corpus preprocessing."""

import collections
import multiprocessing
import os

from nmtwizard import utils
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
                tok_configs.append(operator_config)
    return tok_configs


def _get_num_workers():
    num_cpus = int(os.environ.get('NB_CPU', '1'))
    return num_cpus if num_cpus > 1 else 0  # Run the sequential path if only 1 CPU is available.


class Processor(object):

    def __init__(self, config, pipeline_type):
        self._config = config
        self._pipeline_type = pipeline_type
        self._preprocess_exit_step = None
        self._pipeline = None

    def build_pipeline(self, config=None, override_label=None):
        if config is None:
            config = self._config
        return prepoperator.Pipeline(
            config,
            self._pipeline_type,
            preprocess_exit_step=self._preprocess_exit_step,
            override_label=override_label
        )

    def process_batch(self, tu_batch, options=None):
        # Lazily create the pipeline so that it is created in each worker process.
        _, batch_meta = tu_batch
        override_label = batch_meta.get('label', None) if batch_meta else None
        if self._pipeline is None or self._pipeline.override_label != override_label:
            self._pipeline = self.build_pipeline(override_label=override_label)
        return self._pipeline(tu_batch, options=options)

    def process(self,
                loader,
                consumer,
                num_workers=0,
                preprocess_exit_step=None,
                options=None):
        self._preprocess_exit_step = preprocess_exit_step

        if num_workers == 0:

            for tu_batch in loader():
                tu_batch = self.process_batch(tu_batch, options=options)
                consumer(tu_batch)

        else:

            # Because of the Python GIL (Global Interpreter Lock), we need to use
            # process-based workers to enable true parallelism. The downside is
            # that it duplicates resources for each worker, increasing the
            # memory usage. This is mitigated by the better stream processing of
            # the loader/consumer which avoids loading the full corpus in memory.
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = collections.deque()

                for tu_batch in loader():
                    # Push the batch in the process queue and get a handle on the result.
                    results.append(pool.apply_async(self.process_batch, (tu_batch, options)))

                    # Limit the queue max size to avoid loading too many batches in advance.
                    if len(results) == 2 * num_workers:
                        results[0].wait()

                    # Consume batches that are ready.
                    while len(results) > 0 and results[0].ready():
                        consumer(results.popleft().get())

                # Wait and consume all remaining batches.
                while len(results) > 0:
                    consumer(results.popleft().get())


class TrainingProcessor(Processor):

    def __init__(self, config, corpus_dir, data_dir):
        super().__init__(config, prepoperator.ProcessType.TRAINING)
        self._corpus_dir = corpus_dir
        self._data_dir = data_dir

    def generate_preprocessed_data(self, result='preprocess', preprocess_exit_step=None):

        # TODO V2 : annotations

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
            batch_size = self._config.get('data', {}).get('batch_size', 100000)
            sampler_loader = loader.SamplerFilesLoader(all_files, batch_size)
            sampler_consumer = consumer.MultiConsumer([
                consumer.OpsProfileLogger(),
                consumer.FilterSummaryLogger(),
            ])

            if result == 'subword':
                sampler_consumer.add(consumer.SubwordLearner(
                    self._config, result_dir, preprocess_exit_step))
            elif result == 'vocabulary':
                sampler_consumer.add(consumer.VocabularyBuilder(
                    self._config, result_dir, preprocess_exit_step))
            else:
                sampler_consumer.add(consumer.SamplerFileWriter(
                    self._config, result_dir, preprocess_exit_step, summary))

            self.process(
                sampler_loader,
                sampler_consumer,
                num_workers=_get_num_workers(),
                preprocess_exit_step=preprocess_exit_step)

            sampler_consumer.finalize()
            num_samples = sampler_consumer.num_samples
            data_path = result_dir

        return data_path, train_dir, num_samples, summary, metadata


    def _generate_models(self, tokenization_step, option):

        build_option = "build_" + option

        tok_config = self._config['preprocess'][tokenization_step]

        opt_multi = tok_config.get('multi', {}).get(build_option)
        opt_source = tok_config.get('source', {}).get(build_option)
        opt_target = tok_config.get('target', {}).get(build_option)

        if not opt_multi and not opt_source and not opt_target:
            logger.warning('No \'%s\' option specified, exit without preprocessing.' % build_option)
            return

        if opt_multi and (opt_source or opt_target):
            raise RuntimeError('Cannot specify \'%s\' for both \'multi\' and either \'source\' or \'target\'.' % build_option)

        # Generate preprocessed sentences and feed them to subword learners or to vocabularies.
        self.generate_preprocessed_data(option, preprocess_exit_step=tokenization_step)


    def generate_vocabularies(self):

        # Generate vocabularies and subword models for each tokenization block.
        tok_configs = _get_tok_configs(self._config)

        if not tok_configs:
            raise RuntimeError('No \'tokenization\' operator in preprocess configuration, cannot build vocabularies.)')

        for i, tok_config in enumerate(tok_configs):
            if 'source' not in tok_config or 'target' not in tok_config:
                raise RuntimeError('Each \'tokenization\' operator should contain '
                                   'both \'source\' and \'target\' fields.')

            for side in tok_config:
                if side not in ["source", "target", "multi"]:
                    continue
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

    def __init__(self, config, postprocess=False):
        pipeline_type = (prepoperator.ProcessType.POSTPROCESS
                         if postprocess
                         else prepoperator.ProcessType.INFERENCE)
        super().__init__(config, pipeline_type)
        self._postprocess = postprocess
        self._pipeline = self.build_pipeline()

    def process_input(self, source, target=None, metadata=None, options=None):
        """Processes one translation example at inference.

              In preprocess:
                 input is source or (source, target), single-part and raw string, with incomplete translation in target (if any).
                 output is ((source, metadata), target), tokenized and possibly multipart, where target can be None.

             In postprocess:
                 input is ((source, metadata), target), where source and target are tokenized and possibly multipart.
                 output is single-part postprocessed target."""
        basic_loader = loader.BasicLoader(
            source=source,
            target=target,
            metadata=metadata,
            start_state=self._pipeline.start_state)
        basic_writer = consumer.BasicWriter(self._postprocess)
        self.process(basic_loader, basic_writer, options=options)

        return basic_writer.output


    def process_file(self, source_file, target_file=None, metadata=None):
        """Process translation file at inference.

              In preprocess:
                 input is a file with raw sources
                 output is (source, metadata), where source is a file with tokenized and possibly multipart sources.
              In postprocess:
                 input is ((source, metadata), target), where source and target are files with tokenized and multi-part data
                 output is a file with postprocessed single-part targets."""

        if self._postprocess:
            output_prefix = target_file
            output_suffix = 'detok'
        else:
            output_prefix = source_file
            output_suffix = 'tok'

        output_file = '%s.%s' % (
            output_prefix if not utils.is_gzip_file(output_prefix) else output_prefix[:-3],
            output_suffix)

        file_loader = loader.FileLoader(
            source_file,
            target_file=target_file,
            metadata=metadata,
            start_state=self._pipeline.start_state)
        with consumer.FileWriter(output_file, self._postprocess) as file_consumer:
            self.process(file_loader, file_consumer)
            if self._postprocess:
                return output_file
            else:
                return output_file, file_consumer.metadata
