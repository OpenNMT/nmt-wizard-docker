# coding: utf-8
import six
import abc
import os

from nmtwizard import tokenizer
from nmtwizard import tu


@six.add_metaclass(abc.ABCMeta)
class Operator(object):
    """Base class for preprocessing opertators."""

    @abc.abstractmethod
    def __call__(self, tu_batch):
        raise NotImplementedError()


class PreprocessingPipeline(Operator):

    def __init__(self):
        self._ops = []

    def add(self, op):
        self._ops.append(op)

    def __call__(self, tu_batch):
        for op in self._ops:
            tu_batch = op(tu_batch)
        return tu_batch


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

        opt_multi = config.get('tokenization', {}).get('multi', {}).get('subword')
        opt_source = config.get('tokenization', {}).get('source', {}).get('subword')
        opt_target = config.get('tokenization', {}).get('target', {}).get('subword')

        if opt_multi:
            self._subword_learners['multi'] = tokenizer.make_subword_learner(opt_multi, result_dir)
        if opt_source:
            self._subword_learners['source'] = tokenizer.make_subword_learner(opt_source, result_dir)
        if opt_target:
            self._subword_learners['target'] = tokenizer.make_subword_learner(opt_target, result_dir)


    def __call__(self, tu_batch):
        # Feed lines to subword learners.
        # TODO V2 : feed tokenized lines ?
        # TODO V2 : undo all placeholder annotation for subword processing
        for tu in tu_batch :
            if 'source' in self._subword_learners:
                self._subword_learners['source']['learner'].ingest(tu.src_raw)
            if 'target' in self._subword_learners:
                self._subword_learners['target']['learner'].ingest(tu.tgt_raw)
            if 'multi' in self._subword_learners:
                self._subword_learners['multi']['learner'].ingest(tu.src_raw)
                self._subword_learners['multi']['learner'].ingest(tu.tgt_raw)


    def finalize(self, config):
        # Learn subword models and write them to files.
        for side, learner in self._subword_learners.items():
            name =  config['tokenization'][side]['subword']['name'] \
                    if 'name' in config['tokenization'][side]['subword'] \
                    else 'model'

            subword_type = self._subword_learners[side]['subword_type']
            size = self._subword_learners[side]['size']

            if side == 'multi' :
                out_file = os.path.join(self._result_dir, "joint_" + subword_type + \
                                        "_" + name + "-" + str(size) + "." + \
                                        config['source'] + "_" + config['target'])
            else :
                out_file = os.path.join(self._result_dir, subword_type + \
                                        "_" + name + "-" + str(size) + "." + \
                                        config[side])
            self._subword_learners[side]['learner'].learn(out_file)

            config['tokenization'][side][subword_type+"_model_path"] = out_file

class VocabularyBuilder(Consumer):

    def __init__(self, config, result_dir):

        super(VocabularyBuilder, self).__init__(result_dir)

        self._vocabularies = {}
        self._sums = {}

        opt_multi = config.get('tokenization', {}).get('multi')
        opt_source = config.get('tokenization', {}).get('source')
        opt_target = config.get('tokenization', {}).get('target')

        if opt_multi:
            self._vocabularies['multi'] = {}
            self._sums['multi'] = 0
        if opt_source:
            self._vocabularies['source'] = {}
            self._sums['source'] = 0
        if opt_target:
            self._vocabularies['target'] = {}
            self._sums['target'] = 0


    def __call__(self, tu_batch):

        # TODO V2 : feed tokenized words ?
        # TODO : remove value for placeholders
        for tu in tu_batch :
            if 'source' in self._vocabularies:
                for token in tu.src_raw.split():
                    self._vocabularies['source'][token] = \
                        self._vocabularies['source'].get(token, 0) + 1
                    self._sums['source'] += 1
            if 'target' in self._vocabularies:
                for token in tu.tgt_raw.split():
                    self._vocabularies['target'][token] = \
                        self._vocabularies['target'].get(token, 0) + 1
                    self._sums['target'] += 1
            if 'multi' in self._vocabularies:
                for token in tu.src_raw.split():
                    self._vocabularies['multi'][token] = \
                        self._vocabularies['multi'].get(token, 0) + 1
                    self._sums['multi'] += 1
                for token in tu.tgt_raw.split():
                    self._vocabularies['multi'][token] = \
                        self._vocabularies['multi'].get(token, 0) + 1
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
            name =  config['tokenization'][side]['vocabulary']['name'] \
                    if 'name' in config['tokenization'][side]['vocabulary'] \
                    else 'vocab'

            # Size option is mandatory, already checked it.
            size = config['tokenization'][side]['vocabulary']['size']

            min_frequency = config['tokenization'][side]['vocabulary']['min-frequency'] \
                            if 'min-frequency' in config['tokenization'][side]['vocabulary'] \
                            else 0

            # Find out the real vocabulary size.
            sorted_vocabulary = sorted(vocabulary, key=vocabulary.get, reverse=True)

            real_size = self._prune(vocabulary, sorted_vocabulary, size, min_frequency)

            # Write to file.
            if side == 'multi' :
                out_file = os.path.join(self._result_dir, "joint_" + name + \
                                        "-" + str(real_size) + "." + \
                                        config['source'] + "_" + config['target'])

            else :
                out_file = os.path.join(self._result_dir, name + \
                                        "-" + str(real_size) + "." + \
                                        config[side])

            with open(out_file, 'w') as vocab_file :
                for i in range(real_size):
                    w = sorted_vocabulary[i]
                    vocab_file.write("%s %s\n" % (w, vocabulary[w]/float(self._sums[side])))

            # TODO V2 : use "path" instead
            config['tokenization'][side]['vocabulary'] = out_file

            # TODO V2 : header with configuration ?
            # TODO V2 : deal with placeholders
            # TODO : "merge" and "add" options


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
            self._src_file_out.write("%s\n" % tu.src_raw)
            self._tgt_file_out.write("%s\n" % tu.tgt_raw)


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

        for tu in tu_batch :
            if self._src_tokenizer:
                tu.src_raw = tokenizer.tokenize(self._src_tokenizer, tu.src_raw)

            if self._tgt_tokenizer:
                tu.tgt_raw = tokenizer.tokenize(self._tgt_tokenizer, tu.tgt_raw)

        return tu_batch
