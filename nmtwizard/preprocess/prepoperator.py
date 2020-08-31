# coding: utf-8
import six
import abc
import os
from itertools import chain

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import tokenizer

import systran_align

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


def build_operator(operator_config, global_config, process_type, state):
    """Creates an operator instance from its configuration."""

    operator_type = get_operator_type(operator_config)
    operator_cls = _OPERATORS_REGISTRY.get(operator_type)
    if operator_cls is None:
        logger.warning('Unknown operator \'%s\' will be ignored.' % operator_type)
        return None
    operator_params = get_operator_params(operator_config)

    # Propagate source and target languages
    if "source" not in operator_params:
        operator_params["source"] = {}
    operator_params["source"]["lang"] = global_config["source"]
    if "target" not in operator_params:
        operator_params["target"] = {}
    operator_params["target"]["lang"] = global_config["target"]

    return operator_cls(operator_params, process_type, state)


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

    def __init__(self, config, process_type, preprocess_exit_step=None):
        self._process_type = process_type

        # Start state is used in loader, to inform it about input tokenization.
        # TODO: can we do it better ?
        self.start_state = { "src_tok_config" : None,
                             "tgt_tok_config" : None,
                             "postprocess_only" : False }

        # Current state of pipeline.
        # Passed to and modified by operator initializers if necessary.
        self.build_state = dict(self.start_state)

        self._build_pipeline(config, preprocess_exit_step)


    def _add_op_list(self, op_list_config, exit_step=None):
        # Parameters from global configuration that can be useful for individual operators.

        for i, op_config in enumerate(op_list_config):
            operator = build_operator(op_config, self._config, self._process_type, self.build_state)
            if operator and operator.is_applied_for(self._process_type):
                self._ops.append(operator)
            if exit_step and i == exit_step:
                break


    def _build_pipeline(self, config, preprocess_exit_step=None):
        self._ops = []
        self._config = config
        preprocess_config = config.get("preprocess")
        if preprocess_config:
            self._add_op_list(preprocess_config, exit_step=preprocess_exit_step)

        if self._process_type == ProcessType.POSTPROCESS:
            # Reverse preprocessing operators.
            self._ops = reversed(self._ops)

            # Reverse start and build states.
            self.start_state, self.build_state = self.build_state, self.start_state

            # Flag current pipeline state as 'postprocess_only'.
            # Subsequent operators may need to be aware that they come from 'postprocess' configuration.
            self.build_state['postprocess_only'] = True

            # Add pure postprocessing operators.
            postprocess_config = config.get("postprocess")
            if postprocess_config:
                self._add_op_list(postprocess_config)


    def __call__(self, tu_batch):
        for op in self._ops:
            tu_batch = op(tu_batch, self._process_type)
        return tu_batch


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


    @staticmethod
    def is_applied_for(process_type):
        return True


class TUOperator(Operator):
    """Base class for operations iterating on each TU in a batch."""

    def _preprocess(self, tu_batch, training):
        # TU operator applies an action to each tu.
        # The action yields zero, one or more element for the new list
        tu_list, meta_batch = tu_batch
        tu_list = list(chain.from_iterable(self._preprocess_tu(tu, training) for tu in tu_list))

        return tu_list, meta_batch


    def _postprocess(self, tu_batch):
        tu_list, meta_batch = tu_batch
        tu_list = [self._postprocess_tu(tu) for tu in tu_list]
        return tu_list, meta_batch


    @abc.abstractmethod
    def _preprocess_tu(self, tu, training):
        raise NotImplementedError()


    def _postprocess_tu(self, tu):
        raise NotImplementedError()


class PlaceholderOperator(TUOperator):

    """Base class for operators inserting some placeholder annotation."""

    def _check_placeholder_consistency(self):
        # TODO : implement placeholder checks in preprocess in training and in postprocess
        pass

    def _random_injection(self):
        # TODO : implement common mechanism for random placeholder injection
        # To be activates before and in place any specific processing if configured
        pass

    # TODO : implement random placeholder insertion


class Filter(TUOperator):

    def __init__(self):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        self._criteria = []


    @staticmethod
    def is_applied_for(process_type):
        return process_type == ProcessType.TRAINING


    def _preprocess_tu(self, tu, training):
        for c in self._criteria:
            if (c(tu)):
                return []
        return [tu]


@register_operator("length_filter")
class LengthFilter(Filter):

    def __init__(self, config, process_type, build_state):

        super(LengthFilter, self).__init__()

        self._source_max = config.get('source', {}).get('max_length_char')
        self._target_max = config.get('target', {}).get('max_length_char')

        if self._source_max:
            self._criteria.append(lambda x:len(x.src_detok) > self._source_max)

        if self._target_max:
            self._criteria.append(lambda x:len(x.tgt_detok) > self._target_max)


@register_operator("tokenization")
class Tokenizer(Operator):

    def __init__(self, tok_config, process_type, build_state):
        self._src_tok_config = tok_config.get("source") or tok_config.get("multi")
        self._tgt_tok_config = tok_config.get("target") or tok_config.get("multi")

        self._src_tok_config.pop("lang", None)
        self._tgt_tok_config.pop("lang", None)

        if build_state:
            self._src_tok_config_prev = build_state["src_tok_config"]
            self._tgt_tok_config_prev = build_state["tgt_tok_config"]

            build_state["src_tok_config"] = self._src_tok_config
            build_state["tgt_tok_config"] = self._tgt_tok_config

        self._postprocess_only = build_state['postprocess_only']

        self._src_tokenizer = None
        self._tgt_tokenizer = None


    def _preprocess(self, tu_batch, training=True):
        tu_batch = self._set_tokenizers(tu_batch, self._src_tok_config, self._tgt_tok_config)
        return tu_batch


    def _postprocess(self, tu_batch):
        # Tokenization from 'postprocess' field applies current tokenization in postprocess.
        if self._postprocess_only:
            src_tok_config = self._src_tok_config
            tgt_tok_config = self._tgt_tok_config
        # Tokenization from 'preprocess' field applies previous tokenization in postprocess.
        else:
            src_tok_config = self._src_tok_config_prev
            tgt_tok_config = self._tgt_tok_config_prev
        tu_batch = self._set_tokenizers(tu_batch, src_tok_config, tgt_tok_config)
        return tu_batch


    def _set_tokenizers(self, tu_batch, src_tok_config, tgt_tok_config):
        tu_list, meta_batch = tu_batch
        if not self._src_tokenizer and src_tok_config:
            self._src_tokenizer = tokenizer.build_tokenizer(src_tok_config)

        if not self._tgt_tokenizer and tgt_tok_config:
            self._tgt_tokenizer = tokenizer.build_tokenizer(tgt_tok_config)

        # Set tokenizers for TUs.
        for tu in tu_list :
            tu.src_tok = (self._src_tokenizer, None)
            tu.tgt_tok = (self._tgt_tokenizer, None)

        return tu_list, meta_batch


@register_operator("alignment")
class Aligner(Operator):

    @staticmethod
    def is_applied_for(process_type):
        return process_type == ProcessType.TRAINING

    def __init__(self, align_config, process_type, build_state):
        self._align_config = align_config
        self._aligner = None
        build_state['write_alignment'] = self._align_config.get('write_alignment', False)

    def _preprocess(self, tu_batch, training=True):
        tu_list, meta_batch = tu_batch
        self._build_aligner()
        tu_list = self._set_aligner(tu_list)
        return tu_list, meta_batch


    def _build_aligner(self):
        if not self._aligner and self._align_config:
            # TODO : maybe add monotonic alignment ?
            forward_probs_path=self._align_config.get('forward', {}).get('probs')
            backward_probs_path=self._align_config.get('backward', {}).get('probs')
            if forward_probs_path and backward_probs_path:
                if not os.path.exists(forward_probs_path) or not os.path.isfile(forward_probs_path):
                    raise ValueError("Forward probs file for alignment doesn't exist: %s" % forward_probs_path)
                if not os.path.exists(backward_probs_path) or not os.path.isfile(backward_probs_path):
                    raise ValueError("Backward probs file for alignment doesn't exist: %s" % backward_probs_path)
                self._aligner = systran_align.Aligner(forward_probs_path, backward_probs_path)
            else:
                self._aligner = None

    def _set_aligner(self, tu_list):
        # Set aligner for TUs.
        for tu in tu_list :
            tu.set_aligner(self._aligner)
        return tu_list
