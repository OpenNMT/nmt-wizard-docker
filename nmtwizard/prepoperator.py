# coding: utf-8
import six
import abc
import os
import collections
from itertools import chain

from nmtwizard import tokenizer
from nmtwizard import tu
from nmtwizard.logger import get_logger

logger = get_logger(__name__)

@six.add_metaclass(abc.ABCMeta)
class Operator(object):
    """Base class for preprocessing opertators."""

    @abc.abstractmethod
    def __call__(self, tu_batch):
        raise NotImplementedError()


class PreprocessingPipeline(Operator):

    def __init__(self, config):
        self._ops = []
        if 'preprocess' in config:
            self._build_pipeline(config['preprocess'])

    def _build_pipeline(self, preprocess_config):
        for i, op in enumerate(preprocess_config):
            operator = self._build_operator(i, op)
            if operator:
                self.add(operator)

    def _build_operator(self, step, operator_config):
        if not "op" in operator_config:
            raise RuntimeError('Step %d in \'preprocess\' doesn\'t have mandatory \'op\' option.' % step)
        if operator_config["op"] == "length_filter":
            return LengthFilter(operator_config)
        # TODO : all other operators
        else:
            # TODO : warning or error ?
            logger.warning('Unknown operator \'%s\' will be ignored.' % operator_config["op"])
            return None

    def add(self, op):
        self._ops.append(op)

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


class FileLoader(object):

    def __init__(self, f, batch_size = 0):

        # TODO V2: multiple src and tgt
        self._file = f

        self._batch_size = batch_size
        self._current_line = 0

    def __call__(self):

        tu_batch = []
        while True:
            del tu_batch[:] # TODO V2: Check memory usage on a big corpus
            # Read sampled lines from all files and build TUs.
            batch_line = 0
            while (batch_line < self._batch_size and self._current_line < self._file.lines_count):
                src_line = self._file.files[0].readline().strip()
                tgt_line = self._file.files[1].readline().strip()
                if (self._current_line in self._file.random_sample):
                    while self._file.random_sample[self._current_line] and \
                          batch_line < self._batch_size:
                        tu_batch.append(tu.TranslationUnit(src_line, tgt_line))
                        batch_line += 1
                        self._file.random_sample[self._current_line] -= 1
                self._current_line += 1
            if not tu_batch:
                return

            yield tu_batch


@six.add_metaclass(abc.ABCMeta)
class Consumer(object):
    """Base class for using preprocess results."""

    def __init__(self, result_dir):
        self._result_dir = result_dir

    @abc.abstractmethod
    def __call__(self, tu_batch):
        raise NotImplementedError()

    def open_files(self, f):
        pass

    def close_files(self):
        for _, obj in self.__dict__.items():
            if hasattr(obj, "close") and callable(obj.close) and \
               hasattr(obj, "closed") and not obj.closed:
                obj.close()

    def finalize(self, config):
        pass


class SubwordLearner(Consumer):

    def __init__(self, config, result_dir):

        super(SubwordLearner, self).__init__(result_dir)

        self._subword_learners = {}

        opt_multi = config.get('tokenization', {}).get('multi', {}).get('build_subword')
        opt_source = config.get('tokenization', {}).get('source', {}).get('build_subword')
        opt_target = config.get('tokenization', {}).get('target', {}).get('build_subword')

        if opt_multi:
            self._subword_learners['multi'] = tokenizer.make_subword_learner(opt_multi, result_dir)
        if opt_source:
            self._subword_learners['source'] = tokenizer.make_subword_learner(opt_source, result_dir)
        if opt_target:
            self._subword_learners['target'] = tokenizer.make_subword_learner(opt_target, result_dir)


    def __call__(self, tu_batch):
        # Feed lines to subword learners.
        # TODO V2 : feed tokenized lines, individual tokens ?
        # TODO V2 : undo all placeholder annotation for subword processing
        for tu in tu_batch :
            if 'source' in self._subword_learners:
                self._subword_learners['source']['learner'].ingest(tu.get_src_detok())
            if 'target' in self._subword_learners:
                self._subword_learners['target']['learner'].ingest(tu.get_tgt_detok())
            if 'multi' in self._subword_learners:
                self._subword_learners['multi']['learner'].ingest(tu.get_src_detok())
                self._subword_learners['multi']['learner'].ingest(tu.get_tgt_detok())


    def finalize(self, config):
        # Learn subword models and write them to files.
        for side, learner in self._subword_learners.items():
            name =  config['tokenization'][side]['build_subword']['name'] \
                    if 'name' in config['tokenization'][side]['build_subword'] \
                    else 'model'

            subword_type = self._subword_learners[side]['subword_type']
            size = self._subword_learners[side]['size']

            if side == 'multi' :
                out_file = os.path.join(self._result_dir, \
                                        "joint_%s_%s-%d.%s_%s" % \
                                        (subword_type, name, size, config['source'], config['target']) )
                config['tokenization']['source'][subword_type+"_model_path"] = out_file
                config['tokenization']['target'][subword_type+"_model_path"] = out_file
            else :
                out_file = os.path.join(self._result_dir, \
                                        "%s_%s-%d.%s" % (subword_type, name, size, config[side]))
                config['tokenization'][side][subword_type+"_model_path"] = out_file

            self._subword_learners[side]['learner'].learn(out_file)


class VocabularyBuilder(Consumer):

    def __init__(self, config, result_dir):

        super(VocabularyBuilder, self).__init__(result_dir)

        self._vocabularies = {}
        self._sums = {}

        opt_multi = config.get('tokenization', {}).get('multi', {}).get('build_vocabulary')
        opt_source = config.get('tokenization', {}).get('source', {}).get('build_vocabulary')
        opt_target = config.get('tokenization', {}).get('target', {}).get('build_vocabulary')

        if opt_multi:
            self._vocabularies['multi'] = collections.defaultdict(int)
            self._sums['multi'] = 0
        if opt_source:
            self._vocabularies['source'] = collections.defaultdict(int)
            self._sums['source'] = 0
        if opt_target:
            self._vocabularies['target'] = collections.defaultdict(int)
            self._sums['target'] = 0


    def __call__(self, tu_batch):

        # TODO V2 : feed tokenized words ?
        # TODO : remove value for placeholders
        for tu in tu_batch :
            if 'source' in self._vocabularies:
                for token in tu.get_src_tok():
                    self._vocabularies['source'][token] += 1
                    self._sums['source'] += 1
            if 'target' in self._vocabularies:
                for token in tu.get_tgt_tok():
                    self._vocabularies['target'][token] += 1
                    self._sums['target'] += 1
            if 'multi' in self._vocabularies:
                for token in tu.get_src_tok():
                    self._vocabularies['multi'][token] += 1
                    self._sums['multi'] += 1
                for token in tu.get_tgt_tok():
                    self._vocabularies['multi'][token] += 1
                    self._sums['multi'] += 1

    def _prune(self, vocabulary, sorted_vocabulary, size, min_frequency):
        real_size = len(sorted_vocabulary)

        if min_frequency :
            for t in reversed(sorted_vocabulary):
                if vocabulary[t] < min_frequency :
                    real_size -= 1
                else:
                    break

        return min(real_size, size)


    def finalize(self, config):

        for side, vocabulary in self._vocabularies.items():
            name =  config['tokenization'][side]['build_vocabulary']['name'] \
                    if 'name' in config['tokenization'][side]['build_vocabulary'] \
                    else 'vocab'

            # Size option is mandatory, already checked it.
            size = config['tokenization'][side]['build_vocabulary']['size']

            min_frequency = config['tokenization'][side]['build_vocabulary']['min-frequency'] \
                            if 'min-frequency' in config['tokenization'][side]['build_vocabulary'] \
                            else 0

            added_size = 0

            # Merge previously created vocabulary.
            vocab_to_merge = config['tokenization'][side]['build_vocabulary']['merge'] \
                            if 'merge' in config['tokenization'][side]['build_vocabulary'] \
                            else None

            if vocab_to_merge and os.path.isfile(vocab_to_merge):
                with open(vocab_to_merge, 'r') as f:
                    header = True
                    for l in f:
                        if header and l[0] == '#':
                            continue
                        header = False
                        w = l.strip().split(' ')[0]
                        if w :
                            # Set heaviest frequency on tokens from vocabulary to merge.
                            vocabulary[w] = float("inf")
                            added_size += 1

            # Add extra tokens from a list.
            vocab_to_add = config['tokenization'][side]['build_vocabulary']['add'] \
                           if 'add' in config['tokenization'][side]['build_vocabulary'] \
                           else []

            for w in vocab_to_add:
                vocabulary[w] = float("inf")
                added_size += 1

            if added_size > size :
                raise RuntimeError('The size of extra tokens from \'merge\' and \'add\' (%d) cannot be bigger than than the required vocabulary size (%d)' % (added_size, size))

            # Find out the real vocabulary size.
            sorted_vocabulary = sorted(vocabulary, key=vocabulary.get, reverse=True)

            real_size = self._prune(vocabulary, sorted_vocabulary, size, min_frequency)

            # Write to file.
            if side == 'multi' :
                out_file = os.path.join(self._result_dir, \
                                        "joint_%s-%d.%s_%s" % \
                                        (name, real_size, config['source'], config['target']))
                config['tokenization']['source']['vocabulary'] = out_file
                config['tokenization']['target']['vocabulary'] = out_file

            else :
                out_file = os.path.join(self._result_dir, "%s-%d.%s" % (name, real_size, config[side]))
                config['tokenization'][side]['vocabulary'] = out_file

            with open(out_file, 'w') as vocab_file :
                for i in range(real_size):
                    w = sorted_vocabulary[i]
                    vocab_file.write("%s %s\n" % (w, vocabulary[w]/float(self._sums[side])))

            # TODO V2 : header with configuration ?
            # TODO V2 : deal with placeholders


class FileWriter(Consumer):

    def open_files(self, f):
        # TODO V2 : multiple files
        # TODO V2 : do we output ALL the files that we take as input ?
        src = os.path.join(self._result_dir, os.path.basename(f.files[0].name))
        self._src_file_out = open(src, 'w')

        tgt = os.path.join(self._result_dir, os.path.basename(f.files[1].name))
        self._tgt_file_out = open(tgt, 'w')

    def __call__(self, tu_batch):
        # Write lines to files from TUs
        for tu in tu_batch :
            # TODO V2 : Write preprocessed instead of raw
            self._src_file_out.write("%s\n" % tu.get_src_detok())
            self._tgt_file_out.write("%s\n" % tu.get_tgt_detok())


def make_consumer(config, result_dir, result):

    if result == 'subword':
        return SubwordLearner(config, result_dir)

    if result == 'vocabulary':
        return VocabularyBuilder(config, result_dir)

    # Default is write to file.
    return FileWriter(result_dir)


class Tokenizer(Operator):

    def __init__(self, tok_config):
        self._src_tokenizer = ('source' in tok_config and \
                              tokenizer.build_tokenizer(tok_config['source'])) or \
                              ('multi' in tok_config and \
                               tokenizer.build_tokenizer(tok_config['multi']))

        self._tgt_tokenizer = ('target' in tok_config and \
                              tokenizer.build_tokenizer(tok_config['target'])) or \
                              ('multi' in tok_config and \
                               tokenizer.build_tokenizer(tok_config['multi']))

    def __call__(self, tu_batch):

        # Reset tokenization parameters
        for tu in tu_batch :
            tu.reset_src_tok(self._src_tokenizer)
            tu.reset_tgt_tok(self._tgt_tokenizer)

        return tu_batch
