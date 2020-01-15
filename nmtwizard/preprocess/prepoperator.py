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


class Pipeline(object):
    """Pipeline for building and applying pre/postprocess operators in order."""

    def __init__(self, config, process_type, preprocess_exit_step):
        # Current state of preprocessing pipeline.
        # Passed to and modified by operator initializers if necessary.
        self.start_state = { "src_tokenizer" : None,
                             "tgt_tokenizer" : None }

        self._build_state = dict(self.start_state)

        self._ops = []

        preprocess_config = config.get("preprocess")
        if preprocess_config:
            for i, op in enumerate(preprocess_config):
                operator = self._build_operator(op, process_type)
                if operator and operator.is_applied():
                    self._ops.append(operator)
                if preprocess_exit_step and i == preprocess_exit_step:
                    break

        if process_type=="postprocess":
            self._ops = reversed(self._ops)

            self.start_state, self._build_state = self._build_state, self.start_state

            # Add pure postprocessing operators.
            postprocess_config = config.get("postprocess")
            if postprocess_config:
                for op in postprocess_config:
                    op = op.copy()
                    # Explicitely mark as a postprocess operator.
                    op['postprocess'] = True
                    operator = self._build_operator(op, process_type)
                    if operator and operator.is_applied():
                        self._ops.append(operator)


    def _build_operator(self, operator_config, process_type):
        op = get_operator_type(operator_config)
        params = get_operator_params(operator_config)
        operator = None
        if op == "length_filter":
            operator = LengthFilter(params, process_type)
        elif op == "tokenization":
            operator =  Tokenizer(params, process_type, self._build_state)
        # TODO : all other operators
        else:
            # TODO : warning or error ?
            logger.warning('Unknown operator \'%s\' will be ignored.' % op)
        return operator

    def __call__(self, tu_batch):
        for op in self._ops:
            tu_batch = op(tu_batch)
        return tu_batch


@six.add_metaclass(abc.ABCMeta)
class Operator(object):
    """Base class for preprocessing operators."""

    @abc.abstractmethod
    def __call__(self, tu_batch):
        raise NotImplementedError()

    def is_applied(self):
        # Operator is applied by default, unless _is_applied property is explicitely set to False.
        if hasattr(self, "_is_applied"):
            return self._is_applied
        return True


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

    def __init__(self, process_type):
        # TODO: Sub-criteria for source_detok, target_detok, source_tok, target_tok, or both with alignment ?
        self._criteria = []
        if process_type=="inference" or process_type=="postprocess":
            self._is_applied = False

    def apply(self, tu):
        for c in self._criteria:
            if (c(tu)):
                return []
        return [tu]


class LengthFilter(Filter):

    def __init__(self, config, process_type):

        super(LengthFilter, self).__init__(process_type)

        if self.is_applied():
            self._source_max = config.get('source', {}).get('max_length_char')
            self._target_max = config.get('target', {}).get('max_length_char')

            if self._source_max:
                self._criteria.append(lambda x:len(x.get_src_detok()) > self._source_max)

            if self._target_max:
                self._criteria.append(lambda x:len(x.get_tgt_detok()) > self._target_max)


class Tokenizer(Operator):

    def __init__(self, tok_config, process_type, state):
        src_tokenizer = ('source' in tok_config and \
                         tokenizer.build_tokenizer(tok_config['source'])) or \
                         ('multi' in tok_config and \
                          tokenizer.build_tokenizer(tok_config['multi']))

        tgt_tokenizer = ('target' in tok_config and \
                         tokenizer.build_tokenizer(tok_config['target'])) or \
                         ('multi' in tok_config and \
                          tokenizer.build_tokenizer(tok_config['multi']))

        if process_type=="postprocess":
            self._src_tokenizer = state["src_tokenizer"]
            self._tgt_tokenizer = state["tgt_tokenizer"]
        else:
            self._src_tokenizer = src_tokenizer
            self._tgt_tokenizer = tgt_tokenizer

        state["src_tokenizer"] = src_tokenizer
        state["tgt_tokenizer"] = tgt_tokenizer


    def __call__(self, tu_batch):

        # Reset tokenization parameters
        for tu in tu_batch :
            if self._src_tokenizer:
                tu.set_src_tok(self._src_tokenizer)
            if self._tgt_tokenizer:
                tu.set_tgt_tok(self._tgt_tokenizer)
        return tu_batch
