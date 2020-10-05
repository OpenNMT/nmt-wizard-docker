# coding: utf-8
import six
import abc
import copy
import collections
import time
from itertools import chain

from nmtwizard.logger import get_logger
from nmtwizard.config import merge_config

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

def build_operator(operator_config,
                   global_config,
                   process_type,
                   state,
                   index,
                   override_label=None):
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

    name = operator_params.pop("name", "%s_%d" % (operator_type, index))
    return name, operator_cls(operator_params, process_type, state)


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
                i,
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

        for i, (op_name, op) in enumerate(self._ops):
            if ops_profile is not None:
                start = time.time()

            tu_batch = op(tu_batch, self._process_type)

            if ops_profile is not None:
                end = time.time()
                ops_profile[op_name] += end - start

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

    def __init__(self, criteria=None):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        if criteria is None:
            criteria = []
        self._criteria = criteria


    @staticmethod
    def is_applied_for(process_type):
        return process_type == ProcessType.TRAINING


    def _preprocess_tu(self, tu, meta_batch):
        for c in self._criteria:
            if (c(tu)):
                return []
        return [tu]
