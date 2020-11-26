# coding: utf-8
import six
import abc
import os
import collections
import itertools

import pyonmttok

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import tokenizer

logger = get_logger(__name__)

@six.add_metaclass(abc.ABCMeta)
class Consumer(object):
    """Base class for using preprocess results."""

    def __init__(self):
        self._num_samples = 0

    @property
    def num_samples(self):
        return self._num_samples

    def __call__(self, outputs):
        self._num_samples += len(outputs[0])
        self._consume(outputs)

    @abc.abstractmethod
    def _consume(self, outputs):
        raise NotImplementedError()

    def finalize(self):
        pass


class MultiConsumer(Consumer):
    """A consumer wrapping multiple consumers."""

    def __init__(self, consumers=None):
        super().__init__()
        if consumers is None:
            consumers = []
        self._consumers = consumers

    def add(self, consumer):
        self._consumers.append(consumer)

    def _consume(self, outputs):
        for consumer in self._consumers:
            consumer(outputs)

    def finalize(self):
        for consumer in self._consumers:
            consumer.finalize()


class OpsProfileLogger(Consumer):
    """A consumer that reduces operators profile information and logs the result."""

    def __init__(self):
        super().__init__()
        self._global_profile = collections.defaultdict(float)

    def _consume(self, outputs):
        _, batch_meta = outputs
        profile = batch_meta.get('ops_profile')
        if not profile:
            return
        for name, value in profile.items():
            self._global_profile[name] += value

    def finalize(self):
        if not self._global_profile:
            return
        total_time = sum(self._global_profile.values())
        self._global_profile["all"] = total_time
        sorted_profile = sorted(
            self._global_profile.items(),
            key=lambda item: item[1],
            reverse=True)
        logger.info("Summary of operators execution CPU time:")
        for name, value in sorted_profile:
            logger.info(
                "\t%s: %.3f s (%.1f%%)",
                name,
                value,
                (value / total_time) * 100)


class FilterSummaryLogger(Consumer):
    """A consumer that reduces operators filter information and logs the result."""

    def __init__(self):
        super().__init__()
        self._summary = collections.defaultdict(int)

    def _consume(self, outputs):
        _, batch_meta = outputs
        summary = batch_meta.get("filter_summary")
        if not summary:
            return
        for name, value in summary.items():
            self._summary[name] += value

    def finalize(self):
        if not self._summary:
            return
        logger.info(
            "Summary of filtered sentences (%d sentences dropped in total):",
            sum(self._summary.values()))
        sorted_summary = sorted(
            self._summary.items(),
            key=lambda item: item[1],
            reverse=True)
        for name, value in sorted_summary:
            logger.info("\t%s dropped %d sentences", name, value)


def _ingest_tokens(subword_learner, tokens):
    for part in tokens:
        for token in part:
            subword_learner.ingest_token(token)

def _build_subword_learner(tok_config, result_dir, ref_tok_config=None):
    subword_config = tok_config.get('build_subword')
    if subword_config is None:
        return {}
    if ref_tok_config is None:
        ref_tok_config = tok_config
    subword_info = tokenizer.make_subword_learner(
        subword_config,
        result_dir,
        tokenizer=tokenizer.build_tokenizer(ref_tok_config))
    return subword_info

def _build_vocabulary_counters(config):
    vocab_config = config.get('build_vocabulary')
    if vocab_config is None:
        return {}
    return {
        "tokens": collections.defaultdict(int),
        "total": 0,
    }


class SubwordLearner(Consumer):
    """SubwordLearner class stores, learns and writes subword models."""

    def __init__(self, config, result_dir, tok_step):
        super().__init__()
        self._config = config
        self._result_dir = result_dir
        self._tok_step = tok_step

        tok_config = config['preprocess'][tok_step]
        source_config = tok_config['source']
        target_config = tok_config['target']
        shared_config = tok_config.get('multi', {})

        # The subword learner needs to be aware of the tokenizer annotations to properly
        # ignore them. We assume for a shared subword model that the source and target
        # tokenizers use the same type of annotations, and pass the source tokenization
        # config when building the shared learner.
        shared_subword_info = _build_subword_learner(shared_config, result_dir, source_config)
        if shared_subword_info:
            self._source_subword_info = shared_subword_info
            self._target_subword_info = shared_subword_info
        else:
            self._source_subword_info = _build_subword_learner(source_config, result_dir)
            self._target_subword_info = _build_subword_learner(target_config, result_dir)


    def _consume(self, outputs):
        source_learner = self._source_subword_info.get('learner')
        target_learner = self._target_subword_info.get('learner')
        if source_learner is None and target_learner is None:
            return

        for output in outputs[0]:
            if source_learner is not None:
                _ingest_tokens(source_learner, output.src)
            if target_learner is not None:
                _ingest_tokens(target_learner, output.tgt)


    def finalize(self):
        config = self._config
        if not self._source_subword_info and not self._target_subword_info:
            return

        if self._source_subword_info is self._target_subword_info:
            all_subword_info = [('multi', self._source_subword_info)]
        else:
            all_subword_info = []
            if self._source_subword_info:
                all_subword_info.append(('source', self._source_subword_info))
            if self._target_subword_info:
                all_subword_info.append(('target', self._target_subword_info))

        tok_config = config['preprocess'][self._tok_step]

        # Learn subword models and write them to files.
        for side, subword_info in all_subword_info:
            name =  tok_config[side]['build_subword']['name'] \
                    if 'name' in tok_config[side]['build_subword'] \
                    else 'model'+str(self._tok_step)

            subword_type = subword_info['subword_type']
            size = subword_info['size']

            if side == 'multi' :
                out_file = os.path.join(self._result_dir, \
                                        "joint_%s_%s-%d.%s_%s" % \
                                        (subword_type, name, size, config['source'], config['target']) )
                tok_config['source'][subword_type+"_model_path"] = out_file
                tok_config['target'][subword_type+"_model_path"] = out_file
            else :
                out_file = os.path.join(self._result_dir, \
                                        "%s_%s-%d.%s" % (subword_type, name, size, config[side]))
                tok_config[side][subword_type+"_model_path"] = out_file

            subword_info['learner'].learn(out_file)


class VocabularyBuilder(Consumer):
    """VocabularyBuilder class stores, learns and writes vocabularies."""

    def __init__(self, config, result_dir, tok_step):
        super().__init__()
        self._config = config
        self._result_dir = result_dir
        self._tok_step = tok_step
        self._tokens_to_add = {'source':set(), 'target':set()}

        tok_config = config['preprocess'][tok_step]
        source_config = tok_config['source']
        target_config = tok_config['target']
        shared_config = tok_config.get('multi', {})

        shared_counters = _build_vocabulary_counters(shared_config)
        if shared_counters:
            self._source_counters = shared_counters
            self._target_counters = shared_counters
        else:
            self._source_counters = _build_vocabulary_counters(source_config)
            self._target_counters = _build_vocabulary_counters(target_config)


    def _consume(self, outputs):
        if not self._source_counters and not self._target_counters:
            return

        outputs, meta = outputs

        if 'tokens_to_add' in meta:
            if 'source' in meta['tokens_to_add']:
                self._tokens_to_add['source'].update(meta['tokens_to_add']['source'])
            if 'target' in meta['tokens_to_add']:
                self._tokens_to_add['target'].update(meta['tokens_to_add']['target'])

        for output in outputs:
            for token in itertools.chain.from_iterable(output.src):
                self._source_counters['tokens'][token] += 1
                self._source_counters['total'] += 1
            for token in itertools.chain.from_iterable(output.tgt):
                self._target_counters['tokens'][token] += 1
                self._target_counters['total'] += 1


    def _prune(self, sorted_vocabulary, size, min_frequency):
        real_size = len(sorted_vocabulary)

        if min_frequency :
            for t, f in reversed(sorted_vocabulary):
                if f < min_frequency :
                    real_size -= 1
                else:
                    break

        return min(real_size, size)


    def finalize(self):
        config = self._config
        if not self._source_counters and not self._target_counters:
            return

        tok_config = config['preprocess'][self._tok_step]

        if self._source_counters is self._target_counters:
            vocabularies = [('multi', self._source_counters)]
        else:
            vocabularies = []
            if self._source_counters:
                vocabularies.append(('source', self._source_counters))
            if self._target_counters:
                vocabularies.append(('target', self._target_counters))

        for side, counters in vocabularies:
            vocabulary = counters['tokens']

            total_size = counters['total']
            name =  tok_config[side]['build_vocabulary']['name'] \
                    if 'name' in tok_config[side]['build_vocabulary'] \
                    else 'vocab'+str(self._tok_step)

            # Size option is mandatory, already checked it.
            size = tok_config[side]['build_vocabulary']['size']

            min_frequency = tok_config[side]['build_vocabulary']['min-frequency'] \
                            if 'min-frequency' in tok_config[side]['build_vocabulary'] \
                            else 0

            added_size = 0

            # Merge previously created vocabulary.
            vocab_to_merge = tok_config[side]['build_vocabulary']['merge'] \
                             if 'merge' in tok_config[side]['build_vocabulary'] \
                             else None

            if vocab_to_merge and os.path.isfile(vocab_to_merge):
                for w in tokenizer.vocabulary_iterator(vocab_to_merge):
                    if w :
                        # Set heaviest frequency on tokens from vocabulary to merge.
                        vocabulary[w] = float("inf")
                        added_size += 1

            # Add extra tokens from a list.
            vocab_to_add = tok_config[side]['build_vocabulary']['add'] \
                           if 'add' in tok_config[side]['build_vocabulary'] \
                           else []

            for w in vocab_to_add:
                vocabulary[w] = float("inf")
                added_size += 1

            if added_size > size :
                raise RuntimeError('The size of extra tokens from \'merge\' and \'add\' (%d) cannot be bigger than than the required vocabulary size (%d)' % (added_size, size))

            # Add tokens added by operators, such as extra numbered placeholders that might not all be present in the sampled data.
            if side == 'multi' :
                tokens_to_add = self._tokens_to_add['source'].union(self._tokens_to_add['target'])
            else :
                tokens_to_add = self._tokens_to_add[side]

            for ph in tokens_to_add:
                vocabulary[ph] = float("inf")

            # First add placeholders to vocabulary.
            sorted_vocabulary = [ item for item in vocabulary.items() if pyonmttok.is_placeholder(item[0]) ]

            # Then add everything else in frequency order.
            sorted_vocabulary.extend( sorted([item for item in vocabulary.items() if not pyonmttok.is_placeholder(item[0])], key=lambda k_v: k_v[1], reverse=True) )

            # Find out the real vocabulary size.
            real_size = self._prune(sorted_vocabulary, size, min_frequency)

            # Write to file.
            if side == 'multi' :
                out_file = os.path.join(self._result_dir, \
                                        "joint_vocab_%s-%d.%s_%s" % \
                                        (name, real_size, config['source'], config['target']))
                tok_config['source']['vocabulary_path'] = out_file
                tok_config['target']['vocabulary_path'] = out_file

            else :
                out_file = os.path.join(self._result_dir, "vocab_%s-%d.%s" % (name, real_size, config[side]))
                tok_config[side]['vocabulary_path'] = out_file

            with open(out_file, 'w') as vocab_file :
                # Add header with configuration
                vocab_file.write("# Generated by buildvocab\n")
                vocab_file.write("# CONFIG: {} \n".format(self._config))
                for i in range(real_size):
                    w, f = sorted_vocabulary[i]
                    vocab_file.write("%s %s\n" % (w, f/float(total_size)))


class FileWriter(Consumer):
    """FileWriter writes pre/postprocessed TUs into files at inference."""

    def __init__(self, output_file, postprocess):
        # A basic file writer only opens one file.
        # In preprocess, it is used to store preprocessed source.
        # In postprocess, it is used to store postprocessed target.
        # TODO V2 : multiple files
        super().__init__()
        self._output_file = output_file
        self._postprocess = postprocess
        self._file = None
        self.metadata = []


    def __enter__(self):
        self._file = open(self._output_file, 'w')
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
        self._file = None


    def _consume(self, outputs):
        # Write lines to files from TUs
        for output in outputs[0]:
            # Postprocess.
            if self._postprocess:
                self._file.write("%s\n" % output)
            # Preprocess.
            else:
                for part in output.src:
                    part = " ".join(part)
                    self._file.write("%s\n" % part)
                self.metadata.append(output.metadata)


class SamplerFileWriter(Consumer):
    """SamplerFileWriter writes pre/postprocessed TUs into files at training using SamplerFile object."""

    def __init__(self, config, result_dir, exit_step, summary):
        super().__init__()
        self._config = config
        self._result_dir = result_dir
        self._summary = summary
        self._tokens_to_add = {'source':set(), 'target':set()}
        self._src_suffix = config['source']
        self._tgt_suffix = config['target']


    def _consume(self, outputs):
        outputs, meta = outputs
        if 'tokens_to_add' in meta:
            if 'source' in meta['tokens_to_add']:
                self._tokens_to_add['source'].update(meta['tokens_to_add']['source'])
            if 'target' in meta['tokens_to_add']:
                self._tokens_to_add['target'].update(meta['tokens_to_add']['target'])

        basename = meta["base_name"]
        src_path = os.path.join(self._result_dir, basename + "." + self._src_suffix)
        tgt_path = os.path.join(self._result_dir, basename + "." + self._tgt_suffix)
        align_path = os.path.join(self._result_dir, basename + ".align")
        write_alignment = meta.get('write_alignment', False)

        file_summary = self._summary[basename]
        line_filtered = file_summary.setdefault("linefiltered", 0)
        if line_filtered == 0:
            # When the batch is coming from a file we did not see yet, clear any existing
            # output files.
            open(src_path, "w").close()
            open(tgt_path, "w").close()
            if write_alignment:
                open(align_path, "w").close()

        file_summary["linefiltered"] += len(outputs)

        # Write lines to file from TUs
        with open(src_path, "a") as src_file, open(tgt_path, "a") as tgt_file:
            for output in outputs:
                src_tokens = output.src
                if isinstance(src_tokens, list):
                    for part in src_tokens :
                        part = " ".join(part)
                        src_file.write("%s\n" % part)
                else:
                    src_file.write("%s\n" % output.src)

                tgt_tokens = output.tgt
                if isinstance(tgt_tokens, list):
                    for part in tgt_tokens :
                        part = " ".join(part)
                        tgt_file.write("%s\n" % part)
                else:
                    tgt_file.write("%s\n" % output.tgt)

        if write_alignment:
            with open(align_path, "a") as align_file:
                for output in outputs:
                    alignment = output.alignment
                    if alignment :
                        for part in alignment:
                            part = " ".join(sorted("%s-%s" % tup for tup in part))
                            align_file.write("%s\n" % part)


    def finalize(self):
        config = self._config
        summary = self._summary
        if self._tokens_to_add['source'] or self._tokens_to_add['target'] :
            if 'tokens_to_add' not in summary:
                summary['tokens_to_add'] = {}
            if 'source' not in summary['tokens_to_add']:
                summary['tokens_to_add']['source'] = []
            if 'target' not in summary['tokens_to_add']:
                summary['tokens_to_add']['target'] = []
            source_set = set(summary['tokens_to_add']['source'])
            source_set.update(self._tokens_to_add['source'])
            summary['tokens_to_add']['source'] = list(source_set)
            target_set = set(summary['tokens_to_add']['target'])
            target_set.update(self._tokens_to_add['target'])
            summary['tokens_to_add']['target'] = list(target_set)
