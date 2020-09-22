# coding: utf-8
import six
import abc
import os
import copy
import collections
import time
from itertools import chain

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import tokenizer
from nmtwizard.config import merge_config

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

def get_operator_params(config, override_label=None):
    """Returns the operator parameters from the configuration."""
    config = copy.deepcopy(config)
    config.pop("op", None)
    override_config = config.pop("overrides", None)
    # TODO: implement multiple override labels per batch/corpus.
    if override_config and override_label and override_label in override_config:
        override_config = override_config[override_label]
        config = merge_config(config, override_config)
    return config

def _add_lang_info(operator_params, config, side):
    side_params = operator_params.get(side)
    if side_params is not None:
        side_params["lang"] = config[side]
    else:
        operator_params["%s_lang" % side] = config[side]

def build_operator(operator_config, global_config, process_type, state, override_label):
    """Creates an operator instance from its configuration."""

    operator_type = get_operator_type(operator_config)
    operator_cls = _OPERATORS_REGISTRY.get(operator_type)
    if operator_cls is None:
        raise ValueError("Unknown operator '%s'" % operator_type)
    if not operator_cls.is_applied_for(process_type):
        return None
    operator_params = get_operator_params(operator_config, override_label)
    disabled = operator_params.get("disabled", False)
    if disabled:
        return None

    # Propagate source and target languages
    _add_lang_info(operator_params, global_config, "source")
    _add_lang_info(operator_params, global_config, "target")

    return operator_type, operator_cls(operator_params, process_type, state)


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

    def __init__(self, config, process_type, preprocess_exit_step=None, override_label=None):
        self._process_type = process_type

        # When building a pipeline, a special label can be activated to override operator configuration in some cases (e.g. for some corpora).
        self.override_label = override_label

        # Start state is used in loader, to inform it about input tokenization.
        # TODO: can we do it better ?
        self.start_state = { "src_tokenizer" : None,
                             "tgt_tokenizer" : None,
                             "postprocess_only" : False,
                             "src_vocabulary" : config.get("vocabulary", {}).get("source", {}).get("path"),
                             "tgt_vocabulary" : config.get("vocabulary", {}).get("target", {}).get("path")}

        # Current state of pipeline.
        # Passed to and modified by operator initializers if necessary.
        self.build_state = dict(self.start_state)

        self._build_pipeline(config, preprocess_exit_step)


    def _add_op_list(self, op_list_config, exit_step=None):
        # Parameters from global configuration that can be useful for individual operators.

        for i, op_config in enumerate(op_list_config):
            operator = build_operator(
                op_config,
                self._config,
                self._process_type,
                self.build_state,
                self.override_label)
            if operator is not None:
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
            self._ops = list(reversed(self._ops))

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
        if self._process_type == ProcessType.TRAINING:
            ops_profile = collections.defaultdict(float)
        else:
            ops_profile = None

        for i, (op_type, op) in enumerate(self._ops):
            if ops_profile is not None:
                start = time.time()

            tu_batch = op(tu_batch, self._process_type)

            if ops_profile is not None:
                end = time.time()
                ops_profile['%s_%d' % (op_type, i)] += end - start

        tu_list, batch_meta = tu_batch
        if ops_profile is not None:
            batch_meta['ops_profile'] = ops_profile

        if self._process_type != ProcessType.POSTPROCESS:
            for tu in tu_list:
                tu.synchronize()

        return tu_list, batch_meta


@six.add_metaclass(abc.ABCMeta)
class Operator(object):
    """Base class for preprocessing operators."""

    def __call__(self, tu_batch, process_type):
        self._process_type = process_type
        if process_type == ProcessType.POSTPROCESS:
            tu_batch = self._postprocess(tu_batch)
        else:
            tu_batch = self._preprocess(tu_batch)
        # TODO : do we need a separate function for inference ?
        return tu_batch


    @abc.abstractmethod
    def _preprocess(self, tu_batch):
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

    def _preprocess(self, tu_batch):
        # TU operator applies an action to each tu.
        # The action yields zero, one or more element for the new list
        tu_list, meta_batch = tu_batch
        tu_list = list(chain.from_iterable(self._preprocess_tu(tu, meta_batch) for tu in tu_list))

        return tu_list, meta_batch


    def _postprocess(self, tu_batch):
        tu_list, meta_batch = tu_batch
        tu_list = [self._postprocess_tu(tu) for tu in tu_list]
        return tu_list, meta_batch


    @abc.abstractmethod
    def _preprocess_tu(self, tu, meta_batch):
        raise NotImplementedError()


    def _postprocess_tu(self, tu):
        raise NotImplementedError()


class Filter(TUOperator):

    def __init__(self):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        self._criteria = []


    @staticmethod
    def is_applied_for(process_type):
        return process_type == ProcessType.TRAINING


    def _preprocess_tu(self, tu, meta_batch):
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
        self._src_tokenizer = tokenizer.build_tokenizer(tok_config["source"])
        self._tgt_tokenizer = tokenizer.build_tokenizer(tok_config["target"])

        if build_state:
            self._prev_src_tokenizer = build_state["src_tokenizer"]
            self._prev_tgt_tokenizer = build_state["tgt_tokenizer"]

            build_state["src_tokenizer"] = self._src_tokenizer
            build_state["tgt_tokenizer"] = self._tgt_tokenizer

        self._postprocess_only = build_state['postprocess_only']


    def _preprocess(self, tu_batch):
        tu_batch = self._set_tokenizers(tu_batch, self._src_tokenizer, self._tgt_tokenizer)
        return tu_batch


    def _postprocess(self, tu_batch):
        # Tokenization from 'postprocess' field applies current tokenization in postprocess.
        if self._postprocess_only:
            src_tokenizer = self._src_tokenizer
            tgt_tokenizer = self._tgt_tokenizer
        # Tokenization from 'preprocess' field applies previous tokenization in postprocess.
        else:
            src_tokenizer = self._prev_src_tokenizer
            tgt_tokenizer = self._prev_tgt_tokenizer
        tu_batch = self._set_tokenizers(tu_batch, src_tokenizer, tgt_tokenizer)
        return tu_batch


    def _set_tokenizers(self, tu_batch, src_tokenizer, tgt_tokenizer):
        tu_list, meta_batch = tu_batch

        # Set tokenizers for TUs.
        for tu in tu_list :
            for part in tu.parts:
                part.src_tok = (src_tokenizer, None)
                part.tgt_tok = (tgt_tokenizer, None)

        return tu_list, meta_batch


@register_operator("alignment")
class Aligner(Operator):

    @staticmethod
    def is_applied_for(process_type):
        return process_type == ProcessType.TRAINING

    def __init__(self, align_config, process_type, build_state):
        self._align_config = align_config
        self._aligner = None
        self._write_alignment = self._align_config.get('write_alignment', False)

    def _preprocess(self, tu_batch):
        tu_list, meta_batch = tu_batch
        if self._process_type == ProcessType.TRAINING:
            meta_batch['write_alignment'] = self._write_alignment
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
