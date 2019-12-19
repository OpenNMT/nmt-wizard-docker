# coding: utf-8
import six
import abc
from itertools import chain

import tokenizer
from nmtwizard.logger import get_logger

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

@six.add_metaclass(abc.ABCMeta)
class Operator(object):
    """Base class for preprocessing operators."""

    @abc.abstractmethod
    def __call__(self, tu_batch):
        raise NotImplementedError()

    def _reverse(self):
        # Be default, operator is not reversible, unless redefined otherwise.
        return False


class Pipeline(Operator):
    """Pipeline operator for building and applying TU operators in order."""

    def __init__(self, config, preprocess_exit_step=None):
        # Current state of preprocessing pipeline.
        # Passed to and modified by operator initializers if necessary.
        self.state = { "src_tokenizer" : None,
                       "tgt_tokenizer" : None }

        self._ops = []

        if 'preprocess' in config:
            self._build_pipeline(config['preprocess'], preprocess_exit_step)

    def build_postprocess_pipeline(self, config):

        # Reverse the order of the preprocessing operators.
        # Remove operators that do not require postprocessing.
        self._reverse()

        # Add pure postprocessing operators.
        if 'postprocess' in config:
            self._build_pipeline(config['postprocess'])

    def _reverse(self):
        reversed_ops = []
        for op in self._ops:
            if op._reverse():
                reversed_ops.append(op)
        self._ops = reversed_ops

    def _build_pipeline(self, config, preprocess_exit_step):
        for i, op in enumerate(config):
            operator = self._build_operator(op)
            if operator:
                self._ops.append(operator)
            if preprocess_exit_step and i == preprocess_exit_step:
                break

    def _build_operator(self, operator_config):
        op = get_operator_type(operator_config)
        params = get_operator_params(operator_config)
        if op == "length_filter":
            return LengthFilter(params)
        if op == "tokenization":
            return Tokenizer(params, self.state)
        # TODO : all other operators
        else:
            # TODO : warning or error ?
            logger.warning('Unknown operator \'%s\' will be ignored.' % op)
            return None

    def __call__(self, tu_batch):
        for op in self._ops:
            tu_batch = op(tu_batch)
        return tu_batch


class TUOperator(Operator):
    """Base class for operations iterating on each TU in a batch."""

    def __call__(self, tu_batch):

        # TU operator applies an action to each tu.
        # The action yields zero, one or more element for the new list
        tu_batch = list(chain.from_iterable(self.apply(tu) for tu in tu_batch))

        return tu_batch

    @abc.abstractmethod
    def apply(self, tu_batch):
        raise NotImplementedError()


class Filter(TUOperator):

    def __init__(self):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        self._criteria = []

    def apply(self, tu):
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
            self._criteria.append(lambda x:len(x.src_raw) > self._source_max)

        if self._target_max:
            self._criteria.append(lambda x:len(x.tgt_raw) > self._target_max)


class Tokenizer(Operator):

    def __init__(self, tok_config, state):
        self._prev_src_tokenizer = state["src_tokenizer"]
        self._prev_tgt_tokenizer = state["tgt_tokenizer"]

        self._src_tokenizer = ('source' in tok_config and \
                              tokenizer.build_tokenizer(tok_config['source'])) or \
                              ('multi' in tok_config and \
                               tokenizer.build_tokenizer(tok_config['multi']))

        self._tgt_tokenizer = ('target' in tok_config and \
                              tokenizer.build_tokenizer(tok_config['target'])) or \
                              ('multi' in tok_config and \
                               tokenizer.build_tokenizer(tok_config['multi']))

        state["src_tokenizer"] = self._src_tokenizer
        state["tgt_tokenizer"] = self._tgt_tokenizer


    def __call__(self, tu_batch):

        # Reset tokenization parameters
        for tu in tu_batch :
            tu.reset_src_tok(self._src_tokenizer)
            tu.reset_tgt_tok(self._tgt_tokenizer)

        return tu_batch

    def _reverse(self):
        tmp_src_tokenizer = self._prev_src_tokenizer
        tmp_tgt_tokenizer = self._prev_tgt_tokenizer

        self._src_tokenizer = self._prev_src_tokenizer
        self._tgt_tokenizer = self._prev_tgt_tokenizer

        self._prev_src_tokenizer = tmp_src_tokenizer
        self._prev_tgt_tokenizer = tmp_tgt_tokenizer
