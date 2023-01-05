"""Functions for corpus preprocessing."""

import copy
import collections
import functools
import multiprocessing
import multiprocessing.managers
import threading
import os
import gc

from nmtwizard import config as config_util
from nmtwizard import utils
from nmtwizard import beat_service
from nmtwizard.logger import get_logger
from nmtwizard.preprocess import consumer
from nmtwizard.preprocess import loader
from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess import sampler
from nmtwizard.preprocess import tokenizer
from nmtwizard.preprocess.tu import TranslationUnit

logger = get_logger(__name__)


def _get_tok_configs(config):
    tok_configs = []
    preprocess_config = config.get("preprocess")
    if preprocess_config is not None:
        for i, operator_config in enumerate(preprocess_config):
            if prepoperator.get_operator_type(operator_config) == "tokenization":
                tok_configs.append((i, operator_config))
    return tok_configs


def _get_num_workers():
    num_cpus = int(os.environ.get("NB_CPU", "1"))
    return (
        num_cpus if num_cpus > 1 else 0
    )  # Run the sequential path if only 1 CPU is available.


def _get_corpus_label(tu_batch):
    _, batch_meta = tu_batch
    label = batch_meta.get("label") if batch_meta else None
    if label:
        if isinstance(label, list):
            label = set(label)
        elif isinstance(label, str):
            label = {label}
    return label


def _get_corpus_name(tu_batch):
    _, batch_meta = tu_batch
    return batch_meta.get("base_name")


def _process_batch(
    pipeline,
    tu_batch,
    inference_config=None,
    inference_options=None,
    # Arguments below are used to rebuild the pipeline, if required.
    config=None,
    process_type=None,
    exit_step=None,
    shared_state=None,
):
    """Rebuilds the pipeline if required and processes a batch of TUs."""
    override_label = _get_corpus_label(tu_batch)
    if pipeline is None or override_label != pipeline.override_label:
        if override_label is None:
            logger.info("Building default processing pipeline")
        else:
            logger.info("Building processing pipeline for label %s", override_label)
        pipeline = prepoperator.Pipeline(
            config,
            process_type,
            inference_config=inference_config,
            inference_options=inference_options,
            preprocess_exit_step=exit_step,
            override_label=override_label,
            shared_state=shared_state,
        )

    tu_list, batch_meta = tu_batch

    if not batch_meta.get("no_preprocess"):
        base_name = _get_corpus_name(tu_batch)
        logger.info(
            "Processing %d samples%s",
            len(tu_list),
            " from %s" % base_name if base_name is not None else "",
        )

        tu_list, batch_meta = pipeline(tu_batch, options=inference_options)

        logger.info(
            "Exporting %d samples%s",
            len(tu_list),
            " from %s" % base_name if base_name is not None else "",
        )

    outputs = [tu.export(pipeline.process_type) for tu in tu_list]
    return (outputs, batch_meta), pipeline


# In multiprocessing, we can't build the pipeline in the master process and pass it to
# the worker process because some resources may not be serializable. Instead, the pipeline
# is defined as a global variable that is local to each worker process.
worker_pipeline = None


def _process_batch_on_worker(
    inputs,
    inference_config=None,
    inference_options=None,
    config=None,
    process_type=None,
    exit_step=None,
):
    """Processes a batch of TUs using the pipeline cached on the worker process."""
    global worker_pipeline
    try:
        tu_batch, shared_state = inputs
        outputs, worker_pipeline = _process_batch(
            worker_pipeline,
            tu_batch,
            inference_config=inference_config,
            inference_options=inference_options,
            config=config,
            process_type=process_type,
            exit_step=exit_step,
            shared_state=shared_state,
        )
    except Exception as e:
        corpus_name = _get_corpus_name(tu_batch)
        worker_name = multiprocessing.current_process().name
        raise RuntimeError(
            "An exception occured %sin worker process %s (see above)"
            % (
                "when processing file '%s' " % corpus_name if corpus_name else "",
                worker_name,
            )
        ) from e
    return outputs


class Processor(object):
    def __init__(
        self, config, pipeline_type, preprocess_exit_step=None, num_workers=None
    ):
        if num_workers is None:
            num_workers = _get_num_workers()
        self._num_workers = num_workers
        self._config = config
        self._pipeline_type = pipeline_type
        self._preprocess_exit_step = preprocess_exit_step

        inference = config.get("inference", {})
        if inference and self._pipeline_type.training:
            raise RuntimeError("'inference' field cannot be specified in training")
        self._inference_config = inference.get("overrides")
        self._inference_options = inference.get("options")
        if self._inference_options:
            self._inference_options = prepoperator.read_options(
                config, self._inference_options
            )

        # The global shared state contains all objects that are shared accross workers.
        # It includes shared objects defined in the main configuration as well as shared
        # objects that are corpus-specific.
        self._global_shared_state = SharedState(
            self._config,
            self._pipeline_type,
            self._inference_config,
            preprocess_exit_step=self._preprocess_exit_step,
            num_workers=self._num_workers,
        )

    def process(self, loader, consumer, preprocess_exit_step=None, pipeline=None):

        if self._num_workers == 0:
            logger.info("Start processing")

            with beat_service.monitor_activity() as monitor:
                for tu_batch in loader():
                    override_label = _get_corpus_label(tu_batch)
                    shared_state = self._global_shared_state.get(override_label)
                    outputs, pipeline = _process_batch(
                        pipeline,
                        tu_batch,
                        inference_config=self._inference_config,
                        inference_options=self._inference_options,
                        config=self._config,
                        process_type=self._pipeline_type,
                        exit_step=preprocess_exit_step,
                        shared_state=shared_state,
                    )
                    monitor.notify()
                    consumer(outputs)
                    del tu_batch
                    del outputs
                    gc.collect()

        else:
            logger.info("Start processing using %d worker(s)", self._num_workers)

            def _get_iterator(semaphore, stop_event):
                for tu_batch in loader():
                    override_label = _get_corpus_label(tu_batch)
                    shared_state = self._global_shared_state.get(override_label)
                    yield tu_batch, shared_state
                    del tu_batch
                    # If the semaphore value reaches 0, the iterator will block so that no more
                    # batches are loaded.
                    semaphore.acquire()
                    if stop_event.is_set():
                        break

            process_func = functools.partial(
                _process_batch_on_worker,
                inference_config=self._inference_config,
                inference_options=self._inference_options,
                config=self._config,
                process_type=self._pipeline_type,
                exit_step=preprocess_exit_step,
            )

            # Because of the Python GIL (Global Interpreter Lock), we need to use
            # process-based workers to enable true parallelism. The downside is
            # that it duplicates resources for each worker, increasing the
            # memory usage. This is mitigated by the better stream processing of
            # the loader/consumer which avoids loading the full corpus in memory.
            with multiprocessing.Pool(processes=self._num_workers) as pool:
                # We use a semaphore to control how many batches can be loaded in advance.
                buffer_size = self._num_workers
                semaphore = multiprocessing.Semaphore(buffer_size)
                stop_event = multiprocessing.Event()
                iterable = _get_iterator(semaphore, stop_event)

                with beat_service.monitor_activity() as monitor:
                    try:
                        for result in pool.imap_unordered(process_func, iterable):
                            # Increment the semaphore value to allow loading another batch.
                            semaphore.release()
                            monitor.notify()
                            consumer(result)
                            del result
                            gc.collect()
                    except Exception:
                        # When an exception occurs in a worker, unblock and exit the iterator
                        # to allow the pool to terminate properly.
                        stop_event.set()
                        semaphore.release()
                        raise


class TrainingProcessor(Processor):
    def __init__(
        self, config, corpus_dir, data_dir, preprocess_exit_step=None, num_workers=None
    ):
        super().__init__(
            config,
            prepoperator.ProcessType(utils.Task.TRAINING),
            preprocess_exit_step=preprocess_exit_step,
            num_workers=num_workers,
        )
        self._corpus_dir = corpus_dir
        self._data_dir = data_dir

    def generate_preprocessed_data(
        self, result="preprocess", preprocess_exit_step=None
    ):

        if preprocess_exit_step is None:
            preprocess_exit_step = self._preprocess_exit_step

        # TODO V2 : annotations

        # For backward compatibility with old relative path configurations.
        train_dir = "train"
        if "data" in self._config:
            if "train_dir" in self._config["data"]:
                train_dir = self._config["data"]["train_dir"]
        else:
            logger.warning(
                "No 'data' field in configuration, "
                "all data from the default corpus directory will be used."
            )

        # Default data path.
        data_path = os.path.join(self._corpus_dir, train_dir)

        num_samples = None
        summary = None

        # If some sampling OR preprocessing is applied, change result directory.
        if "data" in self._config or "preprocess" in self._config:

            result_dir = os.path.join(self._data_dir, result)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            if not os.path.isdir(result_dir):
                raise RuntimeError("%s is not a directory" % result_dir)

            # Sample files and write information to a special file structure.
            oversample_as_weights = self._config.get("data", {}).get(
                "oversample_with_sentence_weighting", False
            )
            all_files, summary = sampler.sample(
                self._config, data_path, oversample_as_weights
            )
            batch_size = self._config.get("data", {}).get("batch_size", 100000)
            context = self._config.get("data", {}).get("context", {})
            sampler_loader = loader.SamplerFilesLoader(
                all_files, batch_size, context=context
            )
            sampler_consumer = consumer.MultiConsumer(
                [
                    consumer.OpsProfileLogger(),
                    consumer.SummaryLogger(summary),
                ]
            )

            new_tokens_consumer = None
            if result == "subword":
                sampler_consumer.add(
                    consumer.SubwordLearner(
                        self._config, result_dir, preprocess_exit_step
                    )
                )
            elif result == "vocabulary":
                sampler_consumer.add(
                    consumer.VocabularyBuilder(
                        self._config, result_dir, preprocess_exit_step
                    )
                )
            else:
                new_tokens_consumer = consumer.RegisterNewTokens()
                sampler_consumer.add(new_tokens_consumer)
                sampler_consumer.add(
                    consumer.SamplerFileWriter(
                        self._config, result_dir, preprocess_exit_step, summary
                    )
                )

            logger.info("Generating data to %s", result_dir)
            self.process(
                sampler_loader,
                sampler_consumer,
                preprocess_exit_step=preprocess_exit_step,
            )

            sampler_consumer.finalize()
            num_samples = sampler_consumer.num_samples
            tokens_to_add = None
            if new_tokens_consumer is not None:
                tokens_to_add = {
                    side: list(tokens)
                    for side, tokens in new_tokens_consumer.new_tokens.items()
                }

            data_path = result_dir

        return data_path, train_dir, num_samples, summary, tokens_to_add

    def _generate_models(self, tokenization_step, option):

        build_option = "build_" + option

        tok_config = self._config["preprocess"][tokenization_step]

        opt_multi = tok_config.get("multi", {}).get(build_option)
        opt_source = tok_config.get("source", {}).get(build_option)
        opt_target = tok_config.get("target", {}).get(build_option)

        if not opt_multi and not opt_source and not opt_target:
            logger.warning(
                "Field '%s' is not specified for tokenization operator at step %d, "
                "skipping processing.",
                build_option,
                tokenization_step,
            )
            return

        if opt_multi and (opt_source or opt_target):
            raise RuntimeError(
                "Cannot specify '%s' for both 'multi' and either 'source' or 'target'."
                % build_option
            )

        # Generate preprocessed sentences and feed them to subword learners or to vocabularies.
        self.generate_preprocessed_data(option, preprocess_exit_step=tokenization_step)

    def generate_vocabularies(self):

        # Generate vocabularies and subword models for each tokenization block.
        tok_configs = _get_tok_configs(self._config)

        if not tok_configs:
            raise RuntimeError(
                "No 'tokenization' operator in preprocess configuration, cannot build vocabularies.)"
            )

        for tok_idx, (prep_idx, tok_config) in enumerate(tok_configs):
            if "source" not in tok_config or "target" not in tok_config:
                raise RuntimeError(
                    "Each 'tokenization' operator should contain "
                    "both 'source' and 'target' fields."
                )

            restrict_subword_vocabulary = {}
            for side in tok_config:
                if side not in ["source", "target", "multi"]:
                    continue
                restrict_subword_vocabulary[side] = tok_config[side].pop(
                    "restrict_subword_vocabulary", None
                )
                build_vocab = tok_config[side].get("build_vocabulary")
                if build_vocab:
                    if tok_config[side].get("vocabulary_path", {}):
                        raise RuntimeError(
                            "Cannot build vocabulary if '%s' vocabulary path is already specified."
                            % side
                        )
                    if tok_idx == len(tok_configs) - 1 and self._config.get(
                        "vocabulary", {}
                    ).get(side, {}).get("path"):
                        raise RuntimeError(
                            "Cannot build vocabulary for final tokenization if '%s' vocabulary path for model is already specified."
                            % side
                        )
                    if not build_vocab.get("size"):
                        raise RuntimeError(
                            "'size' option is mandatory to build vocabulary for '%s'."
                            % side
                        )

            self._generate_models(prep_idx, "subword")

            self._generate_models(prep_idx, "vocabulary")

            # Use vocabulary from final tokenization as vocabulary for translation framework.
            if tok_idx == len(tok_configs) - 1:
                for side in tok_config:
                    get_restrict_subword_vocabulary = restrict_subword_vocabulary.get(
                        side
                    )
                    if get_restrict_subword_vocabulary:
                        tok_config[side][
                            "restrict_subword_vocabulary"
                        ] = get_restrict_subword_vocabulary
                    if side == "source" or side == "target":
                        if "vocabulary" not in self._config:
                            self._config["vocabulary"] = {}
                        if side not in self._config["vocabulary"]:
                            self._config["vocabulary"][side] = {}
                        self._config["vocabulary"][side]["path"] = tok_config[side][
                            "vocabulary_path"
                        ]
                        # Only keep 'vocabulary_path' option for final tokenization if explicitly specified.
                        if not tok_config[side].get("use_vocab_in_tok", False):
                            del tok_config[side]["vocabulary_path"]

        preprocess_config = self._config.get("preprocess")
        vocab_config = self._config.get("vocabulary")

        return preprocess_config, vocab_config


class InferenceProcessor(Processor):
    def __init__(self, config, task=utils.Task.TRANSLATION, postprocess=False):
        process_type = prepoperator.ProcessType(task, postprocess=postprocess)
        super().__init__(config, process_type, num_workers=0)
        self._postprocess = postprocess
        # Build a generic pipeline that will be used in process_input.
        self._pipeline = self.build_pipeline(self._config)

    def build_pipeline(self, config):
        return prepoperator.Pipeline(
            config,
            self._pipeline_type,
            inference_config=self._inference_config,
            inference_options=self._inference_options,
            shared_state=self._global_shared_state.get(),
        )

    def process_input(
        self,
        source,
        target=None,
        target_name=None,
        source_context=None,
        target_context=None,
        metadata=None,
        config=None,
        options=None,
    ):
        """Processes one translation example at inference.

        Args:
          source: In preprocess, a string. In postprocess, a (possibly multipart)
            list of tokens.
          target: In preprocess, a string. In postprocess, a (possibly multipart)
            list of tokens.
          target_name: The name of the target that is passed during inference.
          metadata: Additional metadata of the input.
          config: A configuration override for this example.
          options: A dictionary with operators options.

        Returns:
          - In preprocess, a tuple (source_tokens, target_tokens, metadata).
          - In postprocess, a string (the postprocessed target)
        """
        # This method should be thread-safe as the inference server is starting a new
        # thread for each request.

        # Rebuild pipeline if the example has its own configuration.
        if config:
            if config_util.is_v2_config(self._config):
                raise ValueError(
                    "Configuration override is not supported for V2 configurations"
                )
            config = config_util.merge_config(copy.deepcopy(self._config), config)
            pipeline = self.build_pipeline(config)
        else:
            pipeline = self._pipeline

        tu = TranslationUnit(
            source=source,
            metadata=metadata,
            source_tokenizer=pipeline.start_state.get("src_tokenizer"),
        )

        proc = "Postprocess" if self._postprocess else "Preprocess"
        logger.debug(
            "[%d] %s source input:  %s", threading.current_thread().ident, proc, source
        )

        if source_context is not None:
            for sc_idx, sc in enumerate(reversed(source_context)):
                tu.add_source(
                    sc,
                    name=f"context_{sc_idx}",
                    output_delimiter=utils.context_placeholder,
                    before_main=True,
                )

        if target_context is not None:
            if target is None:
                target = ""
            for tc_idx, tc in enumerate(reversed(target_context)):
                tu.add_target(
                    tc,
                    name=f"context_{tc_idx}",
                    output_delimiter=utils.context_placeholder,
                    before_main=True,
                )

        if target is not None:
            tu.add_target(
                target,
                name=target_name,
                tokenizer=pipeline.start_state.get("tgt_tokenizer"),
            )
            logger.debug(
                "[%d] %s target input:  %s",
                threading.current_thread().ident,
                proc,
                target,
            )

        tu_batch = ([tu], {})
        tu_batch = pipeline(tu_batch, options=options)
        tu = tu_batch[0][0]

        if self._postprocess:
            logger.debug(
                "[%d] %s target output:  %s",
                threading.current_thread().ident,
                proc,
                tu.tgt_detok,
            )
            return tu.tgt_detok
        src_tokens = tu.src_tok.tokens
        tgt_tokens = (
            tu.tgt_tok.tokens if tu.tgt_tok is not None else [None for _ in src_tokens]
        )
        logger.debug(
            "[%d] %s source output:  %s",
            threading.current_thread().ident,
            proc,
            src_tokens,
        )
        if tu.tgt_tok is not None:
            logger.debug(
                "[%d] %s target output:  %s",
                threading.current_thread().ident,
                proc,
                tgt_tokens,
            )
        return src_tokens, tgt_tokens, tu.metadata

    def process_file(
        self,
        source_file,
        target_file=None,
        metadata=None,
        target_score_type=None,
        delete_input_files=False,
    ):
        """Process translation file at inference.

        Args:
          source_file: Path to the source file.
          target_file: Path to the target file.
          metadata: A list of metadata, one per example. (Note that in multipart translation,
            multiple lines can refer to the same example.)
          target_score_type: The type of scores that are included in the target file, if any.
          delete_input_files: Delete the input files once the processing is done.

        Returns:
          - In preprocess: a tuple with the path to the preprocessed source file,
            the preprocessed target file (if any), and the metadata.
          - In postprocess: the path to the postprocessed target file.
        """

        def _build_output_path(path, suffix):
            if utils.is_gzip_file(path):
                path = path[:-3]
            return "%s.%s" % (path, suffix)

        batch_size = self._config.get("data", {}).get("batch_size", 100000)
        if self._postprocess:
            file_loader = loader.PostprocessFileLoader(
                source_file,
                target_file,
                metadata=metadata,
                start_state=self._pipeline.start_state,
                batch_size=batch_size,
                target_score_type=target_score_type,
            )
            file_consumer = consumer.PostprocessFileWriter(
                _build_output_path(target_file, "detok")
            )
        else:
            context = self._config.get("data", {}).get("context", {})
            file_loader = loader.PreprocessFileLoader(
                source_file, target_file, batch_size=batch_size, context=context
            )
            file_consumer = consumer.PreprocessFileWriter(
                _build_output_path(source_file, "tok"),
                _build_output_path(target_file, "tok")
                if target_file is not None
                else None,
            )

        with file_consumer:
            self.process(file_loader, file_consumer, pipeline=self._pipeline)

            if delete_input_files:
                os.remove(source_file)
                if target_file is not None:
                    os.remove(target_file)

            return file_consumer.outputs


class SharedManager(multiprocessing.managers.BaseManager):
    """Custom manager for shared resources with multiprocessing."""


class SharedState:
    """A class collecting shared objects created by operators."""

    def __init__(
        self,
        config,
        process_type,
        inference_config,
        preprocess_exit_step=None,
        num_workers=0,
    ):
        self._all_state = collections.defaultdict(dict)
        self._cached_state = {}
        self._config = config
        self._process_type = process_type
        self._inference_config = inference_config
        self._preprocess_exit_step = preprocess_exit_step
        self._num_workers = num_workers
        self._manager = None
        self.get()  # Cache default shared state.

    def get(self, override_label=None):
        """Returns the shared state for this configuration and corpus label."""
        if isinstance(override_label, dict):
            return None
        override_label_str = repr(override_label)
        cached_state = self._cached_state.get(override_label_str)
        if cached_state is not None:
            return cached_state
        preprocess_config = self._config.get("preprocess")
        if not preprocess_config:
            return {}

        supported_features = self._config.get("supported_features")

        if self._num_workers > 0 and self._manager is None:
            # On initialization, register all classes that can be shared.
            for operator_cls, _, _, _ in prepoperator.operator_info_generator(
                preprocess_config,
                self._process_type,
                self._config["source"],
                self._config["target"],
                override_label,
                self._inference_config,
                self._preprocess_exit_step,
                supported_features=supported_features,
                ignore_disabled=False,
            ):
                shared_classes = operator_cls.get_shared_classes()
                if shared_classes is not None:
                    for cls in operator_cls.get_shared_classes():
                        SharedManager.register(cls.__name__, cls)

            self._manager = SharedManager()
            self._manager.start()

        all_builders = {}
        for operator_cls, operator_params, _, i in prepoperator.operator_info_generator(
            preprocess_config,
            self._process_type,
            self._config["source"],
            self._config["target"],
            override_label,
            self._inference_config,
            self._preprocess_exit_step,
            supported_features=supported_features,
        ):
            # Save how to build shared classes for this operator.
            builders = operator_cls.get_shared_builders(
                operator_params, self._process_type
            )
            if builders:
                all_builders[i] = builders

        # Create all new shared instances.
        shared_state = collections.defaultdict(dict)
        for i, builders in all_builders.items():
            existing_state = self._all_state[i]
            for name, (cls, args) in builders.items():
                key = "%s_%s" % (cls.__name__, str(args))
                if key not in existing_state:
                    logger.info(
                        "Building shared instance %s(%s)",
                        cls.__name__,
                        ", ".join(repr(arg) for arg in args),
                    )
                    if self._manager is not None:
                        shared_instance = getattr(self._manager, cls.__name__)(*args)
                    else:
                        shared_instance = cls(*args)
                    existing_state[key] = shared_instance
                shared_state[i].update({name: existing_state[key]})

        self._cached_state[override_label_str] = shared_state
        return shared_state
