# coding: utf-8
import six
import abc
from itertools import chain

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import tokenizer

logger = get_logger(__name__)

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

    def __init__(self, config, preprocess_exit_step):
        self._build_state = None
        self._process_type = ProcessType.TRAINING
        self._build_preprocess_pipeline(config, preprocess_exit_step)


    def _build_preprocess_pipeline(self, config, preprocess_exit_step=None):

        self._ops = []

        preprocess_config = config.get("preprocess")
        if preprocess_config:
            for i, op in enumerate(preprocess_config):
                operator = self._build_operator(op)
                if operator:
                    self._ops.append(operator)
                if preprocess_exit_step and i == preprocess_exit_step:
                    break


    def _build_operator(self, operator_config):
        op = get_operator_type(operator_config)
        params = get_operator_params(operator_config)
        operator = None
        if op == "length_filter":
            operator = LengthFilter(params)
        elif op == "tokenization":
            operator =  Tokenizer(params, self._build_state)
        # TODO : all other operators
        else:
            # TODO : warning or error ?
            logger.warning('Unknown operator \'%s\' will be ignored.' % op)
        return operator


    def __call__(self, tu_batch):
        for i, op in enumerate(self._ops):
            tu_batch = op(tu_batch, self._process_type)
        return tu_batch


class InferencePipeline(Pipeline):

    def __init__(self, config, process_type=ProcessType.INFERENCE):

        self._process_type = process_type

        # TODO : do we really need those now ?
        self.start_state = { "src_tok_config" : None,
                             "tgt_tok_config" : None }

        # Current state of pipeline.
        # Passed to and modified by operator initializers if necessary.
        self._build_state = dict(self.start_state)

        self._build_preprocess_pipeline(config)



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
            for op in postprocess_config:
                operator = self._build_operator(op)
                if operator:
                    self._ops.append(operator)


@six.add_metaclass(abc.ABCMeta)
class Operator(object):
    """Base class for preprocessing operators."""

    def __call__(self, tu_batch, process_type):
        if self._is_enabled_for(process_type):
            tu_batch = self._apply_operator(tu_batch, process_type)
        return tu_batch

    @abc.abstractmethod
    def _apply_operator(self, tu_batch, process_type):
        raise NotImplementedError()

    def _is_enabled_for(self, process_type):
        return True


class TUOperator(Operator):
    """Base class for operations iterating on each TU in a batch."""

    def _apply_operator(self, tu_batch, process_type):
        # TU operator applies an action to each tu.
        # The action yields zero, one or more element for the new list
        tu_batch = list(chain.from_iterable(self._apply_tu_operator(tu, process_type) for tu in tu_batch))

        return tu_batch

    @abc.abstractmethod
    def _apply_tu_operator(self, tu, process_type):
        raise NotImplementedError()


class Filter(TUOperator):

    def __init__(self):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        self._criteria = []

    # TODO : make generic ?
    def _is_enabled_for(self, process_type):
        if process_type > ProcessType.TRAINING:
            return False
        return True

    def _apply_tu_operator(self, tu, process_type):
        for c in self._criteria:
            if (c(tu)):
                return []
        return [tu]


class LengthFilter(Filter):

    def __init__(self, config):

        super(LengthFilter, self).__init__()

        self._source_max = config.get('source', {}).get('max_length_char')
        self._target_max = config.get('target', {}).get('max_length_char')

        if self._source_max:
            self._criteria.append(lambda x:len(x.get_src_detok()) > self._source_max)

        if self._target_max:
            self._criteria.append(lambda x:len(x.get_tgt_detok()) > self._target_max)


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


    def _apply_operator(self, tu_batch, process_type):
        # Build tokenizers if necessary.
        if not self._src_tokenizer:
            if process_type == ProcessType.POSTPROCESS:
                if self._src_tok_config_prev:
                    self._src_tokenizer = tokenizer.build_tokenizer(self._src_tok_config_prev)
            else:
                if self._src_tok_config:
                    self._src_tokenizer = tokenizer.build_tokenizer(self._src_tok_config)

        if not self._tgt_tokenizer:
            if process_type == ProcessType.POSTPROCESS:
                if self._tgt_tok_config_prev:
                    self._tgt_tokenizer = tokenizer.build_tokenizer(self._tgt_tok_config_prev)
            else:
                if self._tgt_tok_config:
                    self._tgt_tokenizer = tokenizer.build_tokenizer(self._tgt_tok_config)

        # Set tokenizers for TUs.
        for tu in tu_batch :
            tu.set_src_tok(self._src_tokenizer)
            tu.set_tgt_tok(self._tgt_tokenizer)

        return tu_batch
