# coding: utf-8
import six
import abc
from itertools import chain

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import tokenizer

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

def get_operator_type(config):
    """Returns the operator type from the configuration."""
    op = config.get("op")
    if op is None:
        raise ValueError("Missing 'op' field in operator configuration: %s" % str(config))
    return op

def get_operator_params(config):
    """Returns the operator parameters from the configuration."""
    config = config.copy()
    config.pop("op", None)
    return config

def build_operator_from_config(config, state):
    """Creates an operator instance from its configuration."""
    operator_type = get_operator_type(config)
    operator_cls = _OPERATORS_REGISTRY.get(operator_type)
    if operator_cls is None:
        raise ValueError("Unknown operator '%s'" % operator_type)
    params = get_operator_params(config)
    if operator_cls.is_stateful():
        return operator_cls(params, state)
    return operator_cls(params)


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

    def _build_pipeline(self, op_list_config, exit_step=None):
        for i, op in enumerate(op_list_config):
            operator = build_operator_from_config(op, self._build_state)
            if operator.is_applied_for(self._process_type):
                self._ops.append(operator)
            if exit_step and i == exit_step:
                break


    def __call__(self, tu_batch):
        for op in self._ops:
            tu_batch = op(tu_batch, self._process_type)
        return tu_batch


class TrainingPipeline(Pipeline):

    def __init__(self, config, preprocess_exit_step):
        self._build_state = None
        self._process_type = ProcessType.TRAINING
        self._ops = []

        preprocess_config = config.get("preprocess")
        if preprocess_config:
            self._build_pipeline(preprocess_config, preprocess_exit_step)


class InferencePipeline(Pipeline):

    def __init__(self, config, process_type=ProcessType.INFERENCE):
        self._process_type = process_type
        self._ops = []

        # TODO : do we really need those now ?
        self.start_state = { "src_tok_config" : None,
                             "tgt_tok_config" : None }

        # Current state of pipeline.
        # Passed to and modified by operator initializers if necessary.
        self._build_state = dict(self.start_state)

        preprocess_config = config.get("preprocess")
        if preprocess_config:
            self._build_pipeline(preprocess_config)


class PostprocessPipeline(InferencePipeline):

    def __init__(self, config):
        super(PostprocessPipeline, self).__init__(config, ProcessType.POSTPROCESS)

        # Reverse preprocessing operators.
        self._ops = reversed(self._ops)

        # TODO
        self.start_state, self._build_state = self._build_state, self.start_state
        # Add pure postprocessing operators.
        postprocess_config = config.get("postprocess")
        if postprocess_config:
            self._build_pipeline(postprocess_config)


@six.add_metaclass(abc.ABCMeta)
class Operator(object):
    """Base class for preprocessing operators."""

    def __call__(self, tu_batch, process_type):
        if process_type == ProcessType.POSTPROCESS:
            tu_batch = self._postprocess(tu_batch)
        else:
            tu_batch = self._preprocess(tu_batch, training=process_type == ProcessType.TRAINING)
        # TODO : do we need a separate function for inference ?
        return tu_batch


    @abc.abstractmethod
    def _preprocess(self, tu_batch, training):
        raise NotImplementedError()


    def _postprocess(self, tu_batch):
        # Postprocess is not mandatorily reimplemented by each operator.
        # If the operator has no postprocess, it should implement "is_applied_for" function correctly and never get here.
        raise NotImplementedError()


    def is_applied_for(self, process_type):
        return True


    @staticmethod
    def is_stateful():
        return False


class TUOperator(Operator):
    """Base class for operations iterating on each TU in a batch."""

    def _preprocess(self, tu_batch, training):
        # TU operator applies an action to each tu.
        # The action yields zero, one or more element for the new list
        tu_batch = list(chain.from_iterable(self._preprocess_tu(tu, training) for tu in tu_batch))

        return tu_batch


    @abc.abstractmethod
    def _preprocess_tu(self, tu, training):
        raise NotImplementedError()


class Filter(TUOperator):

    def __init__(self):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        self._criteria = []


    def is_applied_for(self, process_type):
        return process_type == ProcessType.TRAINING


    def _preprocess_tu(self, tu, training):
        for c in self._criteria:
            if (c(tu)):
                return []
        return [tu]


@register_operator("length_filter")
class LengthFilter(Filter):

    def __init__(self, config):

        super(LengthFilter, self).__init__()

        self._source_max = config.get('source', {}).get('max_length_char')
        self._target_max = config.get('target', {}).get('max_length_char')

        if self._source_max:
            self._criteria.append(lambda x:len(x.src_detok) > self._source_max)

        if self._target_max:
            self._criteria.append(lambda x:len(x.tgt_detok) > self._target_max)


@register_operator("tokenization")
class Tokenizer(Operator):

    def __init__(self, tok_config, state):
        self._src_tok_config = tok_config.get("source") or tok_config.get("multi")
        self._tgt_tok_config = tok_config.get("target") or tok_config.get("multi")

        if state:
            self._src_tok_config_prev = state["src_tok_config"]
            self._tgt_tok_config_prev = state["tgt_tok_config"]

            state["src_tok_config"] = self._src_tok_config
            state["tgt_tok_config"] = self._tgt_tok_config

        self._src_tokenizer = None
        self._tgt_tokenizer = None


    @staticmethod
    def is_stateful():
        return True


    def _preprocess(self, tu_batch, training=True):
        tu_batch = self._set_tokenizers(tu_batch, self._src_tok_config, self._tgt_tok_config)
        return tu_batch


    def _postprocess(self, tu_batch):
        tu_batch = self._set_tokenizers(tu_batch, self._src_tok_config_prev, self._tgt_tok_config_prev)
        return tu_batch


    def _set_tokenizers(self, tu_batch, src_tok_config, tgt_tok_config):

        if not self._src_tokenizer and src_tok_config:
            self._src_tokenizer = tokenizer.build_tokenizer(src_tok_config)

        if not self._tgt_tokenizer and tgt_tok_config:
            self._tgt_tokenizer = tokenizer.build_tokenizer(tgt_tok_config)

        # Set tokenizers for TUs.
        for tu in tu_batch :
            tu.src_tok = (self._src_tokenizer, None)
            tu.tgt_tok = (self._tgt_tokenizer, None)

        return tu_batch
