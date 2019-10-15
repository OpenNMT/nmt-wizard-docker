"""Functions for corpus preprocessing."""

import os

from nmtwizard.logger import get_logger
from nmtwizard.sampler import sample
from nmtwizard import prepoperator
from nmtwizard import tokenizer

logger = get_logger(__name__)

def _generate_models(config, corpus_dir, data_dir, option):

    opt_multi = config.get('tokenization', {}).get('multi', {}).get(option)
    opt_source = config.get('tokenization', {}).get('source', {}).get(option)
    opt_target = config.get('tokenization', {}).get('target', {}).get(option)

    if not opt_multi and not (opt_source and opt_target):
        logger.warning('No \'' + option + '\' option specified, exit without preprocessing.')
        return

    if (opt_multi and opt_source) or \
       (opt_multi and opt_target) :
        raise RuntimeError('Cannot specify \'' + option + '\' for both \'multi\' and either \'source\' or \'target\'.')

    # Generate preprocessed sentences and feed them to subword learners or to vocabularies.
    generate_preprocessed_data(config, corpus_dir, data_dir, option)


def generate_vocabularies(config, corpus_dir, data_dir):

    # TODO V2 : change this when tokenization will be part of preprocessing pipeline.
    if 'tokenization' not in config:
        raise RuntimeError('No \'tokenization\' field in configuration, \
                            cannot build the vocabularies.)')

    if ('source' not in config['tokenization'] or \
        'target' not in config['tokenization']) and \
        'multi' not in config['tokenization'] :
        raise RuntimeError('\'tokenization\' field should contain \
                           either both \'source\' and \'target\' fields \
                           or \'multi\' field.')

    # Check there isn't already a vocabulary.
    for side in config['tokenization']:
        if config['tokenization'][side].get('vocabulary', {}).get('path'):
            raise RuntimeError('Cannot build vocabulary if \'%s\' vocabulary path is already specified.' % side)
        if not config['tokenization'][side].get('vocabulary', {}).get('size'):
            raise RuntimeError('\'size\' option is mandatory to build vocabulary for \'%s\'.' % side)

    _generate_models(config, corpus_dir, data_dir, 'subword')

    _generate_models(config, corpus_dir, data_dir, 'vocabulary')

    # TODO V2 : we don't need to return tokenization ?
    return {}, config['tokenization']


def generate_preprocessed_data(config, corpus_dir, data_dir, result='preprocess'):

    # TODO V2 : annotations
    # TODO V2 : file-specific rules/extra

    # For backward compatibility with old relative path configurations.
    train_dir = 'train'
    if 'data' in config :
        if 'train_dir' in config['data']:
            train_dir = config['data']['train_dir']
    else :
        logger.warning('No \'data\' field in configuration, \
                        default corpus directory and all corpora are used.)')

    # Default data path.
    data_path = os.path.join(corpus_dir, train_dir)

    num_samples = None
    summary = None
    metadata = None

    # If some sampling OR preprocessing is applied, change result directory.
    if 'data' in config or 'preprocess' in config:

        result_dir = os.path.join(data_dir, result)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        if not os.path.isdir(result_dir):
            raise RuntimeError('%s is not a directory' % result_dir)
        logger.info('Generating data to %s', result_dir)

        # Sample files and write information to a special file structure.
        all_files, summary, metadata = sample(config, data_path)

        num_samples = 0
        consumer = prepoperator.make_consumer(config, result_dir, result)
        for f in all_files:
            lines_filtered = 0
            if f.lines_kept :

                # Default batch size is the whole sample size.
                batch_size = f.lines_kept
                if 'preprocess' in config and 'batch_size' in config['preprocess'] :
                    batch_size = config['preprocess']['batch_size']

                # Loader : load selected lines into batches.
                loader = prepoperator.FileLoader(f, batch_size)

                # Preprocessor : preprocess lines in batch.
                pipeline = prepoperator.PreprocessingPipeline()
                # TODO V2 : Initialize FILE-SPECIFIC preprocessor pipeline
                # if 'preprocess' in config:
                # pipeline.add(buildPreprocessPipeline(config['preprocess']))
                # TODO V2 : ultimately, tokenization should be part of the preprocess pipeline
                if 'tokenization' in config:
                    pipeline.add(prepoperator.Tokenizer(config['tokenization']))

                # Consumer : action after preprocessing.
                # * write lines to file, if simple preprocessing.
                # * feed to subword learner, if building subword model.
                # * add words to vocabulary, if building vocabulary.
                consumer.open_files(f)

                for tu_batch in loader():
                    tu_batch = pipeline(tu_batch)
                    consumer(tu_batch)
                    lines_filtered += len(tu_batch)
                    # TODO V2 : parallelization
                f.close_files()

                consumer.close_files()

            consumer.finalize(config)

            if lines_filtered != f.lines_kept:
                num_samples += lines_filtered
                summary[f.base_name]["lines_filtered"] = lines_filtered
            else:
                num_samples += f.lines_kept
                summary[f.base_name]["lines_filtered"] = f.lines_kept

        data_path = result_dir

    return data_path, train_dir, num_samples, summary, metadata
