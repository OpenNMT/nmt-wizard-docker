import abc
import copy
import collections
import time
import jsonschema
from itertools import chain

from nmtwizard.logger import get_logger
from nmtwizard.config import merge_config
from nmtwizard.config import read_options

logger = get_logger(__name__)

_OPERATORS_REGISTRY = {}


def register_operator(name):
    """A class decorator to register an operator.

    Example:

        @register_operator("length_filter")
        class LengthFilter(Operator):
            ...
    """
    if name in _OPERATORS_REGISTRY:
        raise ValueError("An operator with name '%s' is already registered" % name)

    def _decorator(cls):
        _OPERATORS_REGISTRY[name] = cls
        return cls

    return _decorator


def operator_info_generator(
    config,
    process_type,
    override_label=None,
    preprocess_exit_step=None,
    supported_features=None,
    ignore_disabled=True,
):
    for i, operator_config in enumerate(config):
        if preprocess_exit_step is not None and i > preprocess_exit_step:
            return
        # Get operator class and config.
        operator_type = get_operator_type(operator_config)
        operator_cls = get_operator_class(operator_type)
        if not operator_cls.is_applied_for(process_type):
            continue
        # Ignore 'disabled' for operators supported in inference
        if supported_features and operator_cls.is_supported(
            supported_features, process_type
        ):
            ignore_disabled = False
        operator_params = get_operator_params(
            operator_config, operator_type, override_label=override_label
        )
        if ignore_disabled and operator_params.get("disabled", False):
            continue
        yield operator_cls, operator_params, operator_type, i


def get_operator_type(config):
    """Returns the operator type from the configuration."""
    op = config.get("op")
    if op is None:
        raise ValueError(
            "Missing 'op' field in operator configuration: %s" % str(config)
        )
    return op


def get_operator_class(op):
    operator_cls = _OPERATORS_REGISTRY.get(op)
    if operator_cls is None:
        raise ValueError("Unknown operator '%s'" % op)
    return operator_cls


def get_operator_params(config, operator_type, override_label=None):
    """Returns the operator parameters from the configuration."""
    config = copy.deepcopy(config)
    config.pop("op", None)
    override_config = config.pop("overrides", None)

    if override_label and override_config:
        override = [label for label in override_label if label in override_config]
        override_num = len(override)
        if override_num > 1:
            raise RuntimeError(
                "One corpus requires different overrides (%s) for the same operator (%s)."
                % (override_config, operator_type)
            )
        if override_num == 1:
            override = override[0]
            override_config = override_config[override]
            config = merge_config(config, override_config)
    return config


def _add_lang_info(operator_params, config, side):
    side_params = operator_params.get(side)
    if side_params is not None:
        side_params["lang"] = config[side]
    else:
        operator_params["%s_lang" % side] = config[side]


def build_operator(
    operator_type,
    operator_cls,
    operator_params,
    global_config,
    process_type,
    build_state,
    index,
    shared_state=None,
    inference_config=None,
):
    """Creates an operator instance from its configuration."""

    # Propagate source and target languages
    _add_lang_info(operator_params, global_config, "source")
    _add_lang_info(operator_params, global_config, "target")

    args = []
    if shared_state:
        args.append(shared_state)
    name = operator_params.get("name", "%s_%d" % (operator_type, index + 1))
    if inference_config:
        op_inference_config = inference_config.get(name)
        if op_inference_config:
            operator_params = merge_config(operator_params, op_inference_config)
    if process_type == ProcessType.TRAINING:
        operator_cls.validate_parameters(operator_params, name)
    logger.debug("Building operator %s", name)
    operator = operator_cls(operator_params, process_type, build_state, *args)
    # We set common private attributes here so that operators do not need to call
    # the base constructor.
    operator._name = name
    operator._verbose = operator_params.get("verbose", False)
    operator._process_type = process_type
    return operator


class ProcessType(object):
    """Type of processing pipeline.

    Possible values are:
      * ``TRAINING``
      * ``INFERENCE``
      * ``POSTPROCESS``
    """

    TRAINING = 0
    INFERENCE = 1
    POSTPROCESS = 2


class Pipeline(object):
    """Pipeline for building and applying pre/postprocess operators in order."""

    def __init__(
        self,
        config,
        process_type,
        preprocess_exit_step=None,
        override_label=None,
        shared_state=None,
    ):
        self._process_type = process_type

        # When building a pipeline, a special label can be activated to override operator configuration in some cases (e.g. for some corpora).
        self.override_label = override_label

        supported_features = config.get("supported_features")

        # Start state is used in loader, to inform it about input tokenization.
        # TODO: can we do it better ?
        self.start_state = {
            "src_tokenizer": None,
            "tgt_tokenizer": None,
            "postprocess_only": False,
            "supported_features": supported_features,
            "src_vocabulary": config.get("vocabulary", {})
            .get("source", {})
            .get("path"),
            "tgt_vocabulary": config.get("vocabulary", {})
            .get("target", {})
            .get("path"),
        }

        # Current state of pipeline.
        # Passed to and modified by operator initializers if necessary.
        self.build_state = dict(self.start_state)

        self._build_pipeline(config, preprocess_exit_step, shared_state)

    @property
    def process_type(self):
        return self._process_type

    def _add_op_list(self, op_list_config, exit_step=None, shared_state=None):
        # Parameters from global configuration that can be useful for individual operators.

        supported_features = self.build_state.get("supported_features")
        for operator_info in operator_info_generator(
            op_list_config,
            self._process_type,
            self.override_label,
            exit_step,
            supported_features,
        ):
            operator_cls, operator_params, operator_type, i = operator_info

            operator = build_operator(
                operator_type,
                operator_cls,
                operator_params,
                self._config,
                self._process_type,
                self.build_state,
                i,
                shared_state=shared_state.get(i) if shared_state else None,
                inference_config=self._inference_config,
            )
            if operator is not None:
                self._ops.append(operator)

    def _build_pipeline(
        self,
        config,
        preprocess_exit_step=None,
        shared_state=None,
    ):
        self._ops = []
        self._config = config
        inference = config.get("inference", {})
        if inference and self._process_type == ProcessType.TRAINING:
            raise RuntimeError("'inference' field can only be specified in translation")
        self._inference_config = inference.get("overrides")
        self._inference_options = inference.get("options")
        if self._inference_options:
            self._inference_options = read_options(config, self._inference_options)

        preprocess_config = config.get("preprocess")
        if preprocess_config:
            self._add_op_list(
                preprocess_config,
                exit_step=preprocess_exit_step,
                shared_state=shared_state,
            )

        if self._process_type == ProcessType.POSTPROCESS:
            # Reverse preprocessing operators.
            self._ops = list(reversed(self._ops))

            # Reverse start and build states.
            self.start_state, self.build_state = self.build_state, self.start_state

            # Flag current pipeline state as 'postprocess_only'.
            # Subsequent operators may need to be aware that they come from 'postprocess' configuration.
            self.build_state["postprocess_only"] = True

            # Add pure postprocessing operators.
            postprocess_config = config.get("postprocess")
            if postprocess_config:
                self._add_op_list(postprocess_config)

    def __call__(self, tu_batch, options=None):
        if self._process_type == ProcessType.TRAINING:
            ops_profile = collections.defaultdict(float)
        else:
            ops_profile = None

        if options is None:
            options = self._inference_options

        for op in self._ops:
            if ops_profile is not None:
                start = time.time()

            kwargs = {}
            op_options = options.get(op.name) if options else None
            if op_options is not None:
                if not op.accept_options():
                    raise RuntimeError(
                        "Operator %s does not accept runtime options" % op.name
                    )
                kwargs["options"] = op_options

            process_type = (
                "postprocess"
                if self._process_type == ProcessType.POSTPROCESS
                else "preprocess"
            )
            logger.debug("Applying operator %s in %s", op.name, process_type)
            tu_batch = op(tu_batch, **kwargs)

            if ops_profile is not None:
                end = time.time()
                ops_profile[op.name] += end - start

        tu_list, batch_meta = tu_batch
        if ops_profile is not None:
            batch_meta["ops_profile"] = ops_profile

        for tu in tu_list:
            tu.finalize(self._process_type)

        return tu_list, batch_meta


class Operator(abc.ABC):
    """Base class for preprocessing operators."""

    @classmethod
    def _config_schema(cls):
        return {
            "type": "object",
            "properties": {
                "source_lang": {"type": "string"},
                "target_lang": {"type": "string"},
                "verbose": {"type": "boolean"},
                "comment": {"type": "string"},
                "overrides": {"type": "object"},
                "disabled": {"type": "boolean"},
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        }

    def __init__(self, params, process_type, build_state):
        pass

    @property
    def name(self):
        return self._name

    @property
    def process_type(self):
        return self._process_type

    @classmethod
    def validate_parameters(cls, params, name):
        try:
            jsonschema.validate(instance=params, schema=cls._config_schema())
        except jsonschema.exceptions.ValidationError as err:
            raise ValueError(
                f"Invalid configuration for '{name}' operator: {err.message}"
            ) from None

    def __call__(self, tu_batch, **kwargs):
        if self.process_type == ProcessType.POSTPROCESS:
            tu_batch = self._postprocess(tu_batch, **kwargs)
        else:
            tu_batch = self._preprocess(tu_batch, **kwargs)
        # TODO : do we need a separate function for inference ?
        return tu_batch

    @abc.abstractmethod
    def _preprocess(self, tu_batch, **kwargs):
        raise NotImplementedError()

    def _postprocess(self, tu_batch, **kwargs):
        # Postprocess is not mandatorily reimplemented by each operator.
        # If the operator has no postprocess, it should implement "is_applied_for" function correctly and never get here.
        raise NotImplementedError()

    @staticmethod
    def is_applied_for(process_type):
        return True

    @staticmethod
    def is_supported(supported_features, process_type):
        return False

    @staticmethod
    def accept_options():
        return False

    # When the preprocessing is running in parallel, the state of each operator is
    # duplicated in each worker process. This can be an issue for resources that
    # use a lot of memory. The methods below allow building objects that will be
    # shared across workers. Operators with a shared state will take an additional
    # "shared_state" argument in the their constructor.
    #
    # See the alignment operator for an example.

    @staticmethod
    def get_shared_classes():
        return None

    @staticmethod
    def get_shared_builders(config, process_type):
        return None


class TUOperator(Operator):
    """Base class for operations iterating on each TU in a batch."""

    def _preprocess(self, tu_batch, **kwargs):
        # TU operator applies an action to each tu.
        # The action yields zero, one or more element for the new list
        tu_list, meta_batch = tu_batch
        tu_list = list(
            chain.from_iterable(
                self._preprocess_tu(tu, meta_batch, **kwargs) for tu in tu_list
            )
        )
        return tu_list, meta_batch

    def _postprocess(self, tu_batch, **kwargs):
        tu_list, meta_batch = tu_batch
        tu_list = [self._postprocess_tu(tu, **kwargs) for tu in tu_list]
        return tu_list, meta_batch

    @abc.abstractmethod
    def _preprocess_tu(self, tu, meta_batch, **kwargs):
        raise NotImplementedError()

    def _postprocess_tu(self, tu, **kwargs):
        raise NotImplementedError()


class MonolingualOperator(TUOperator):
    """Base class for operations applying monolingual processing in each TU in a batch."""

    @classmethod
    def _config_schema(cls):
        schema = super(MonolingualOperator, cls)._config_schema()
        schema["properties"].update(
            {
                "source": {"type": "object", "additionalProperties": False},
                "target": {"type": "object", "additionalProperties": False},
            }
        )
        return schema

    def __init__(self, config, process_type, build_state):
        self._postprocess_only = build_state.get("postprocess_only")
        self._verbose = config.get("verbose")
        self._process_type = process_type
        self._source_process = None
        self._target_process = None

        self._source_lang = config.get("lang") or config.get("source_lang")
        self._target_lang = config.get("lang") or config.get("target_lang")

        if self._postprocess_only:
            # For postprocess only, the config only applies to the target.
            self._target_process = self._build_process(config, "target", build_state)
        else:
            source_config = config.get("source")
            if source_config is not None:
                self._source_process = self._build_process(
                    source_config, "source", build_state
                )
            target_config = config.get("target")
            if target_config is not None:
                self._target_process = self._build_process(
                    target_config, "target", build_state
                )

    @abc.abstractmethod
    def _build_process(self, config):
        raise NotImplementedError()

    @abc.abstractmethod
    def _apply_process(self, process, arg, **kwargs):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _detok(self):
        raise NotImplementedError()

    @property
    def _apply_in_postprocess(self):
        return False

    def _preprocess_tu(self, tu, meta_batch, **kwargs):

        options = kwargs.get(
            "options"
        )  # Inference options are only applied in source in preprocess.
        if self._source_process is not None or options:
            if self._detok:
                for name, detok in tu.src_detok_gen():
                    src_detok = self._apply_process(
                        self._source_process, detok, **kwargs
                    )
                    tu.set_src_detok(src_detok, name)
            else:
                for name, tok in tu.src_tok_gen():
                    src_tok = self._apply_process(self._source_process, tok, **kwargs)
                    tu.set_src_tok(src_tok, name)

        if self._target_process is not None:
            if self._detok:
                for name, detok in tu.tgt_detok_gen():
                    tgt_detok = self._apply_process(
                        self._target_process, detok, **kwargs
                    )
                    tu.set_tgt_detok(tgt_detok, name)
            else:
                for name, tok in tu.tgt_tok_gen():
                    tgt_tok = self._apply_process(self._target_process, tok, **kwargs)
                    tu.set_tgt_tok(tgt_tok, name)

        return [tu]

    def _postprocess_tu(self, tu, **kwargs):
        if self._postprocess_only or self._apply_in_postprocess:
            options = kwargs.get("options")
            if self._target_process is not None or options:
                if self._detok:
                    tu.tgt_detok = self._apply_process(
                        self._target_process, tu.tgt_detok, **kwargs
                    )
                else:
                    tu.tgt_tok = self._apply_process(
                        self._target_process, tu.tgt_tok, **kwargs
                    )
        return tu


class Filter(TUOperator):
    def __init__(self, criteria=None):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        if criteria is None:
            criteria = []
        self._criteria = criteria

    @staticmethod
    def is_applied_for(process_type):
        return process_type == ProcessType.TRAINING

    def __call__(self, tu_batch):
        before = len(tu_batch[0])
        tu_batch = super().__call__(tu_batch)
        after = len(tu_batch[0])
        filter_summary = tu_batch[1].setdefault(
            "filter_summary", collections.defaultdict(int)
        )
        filter_summary[self.name] += before - after
        return tu_batch

    def _preprocess_tu(self, tu, meta_batch):
        for c in self._criteria:
            to_filter = c(tu)
            message = ""
            if isinstance(to_filter, tuple):
                to_filter, message = to_filter
            if to_filter:
                if self._verbose:
                    logger.info(
                        f"{message} : '{self.name}' operator filters the following sentence \tSRC : {tu.src_detok}\tTGT : {tu.tgt_detok}"
                    )
                return []
        return [tu]
