"""Tokenization utilities."""

import os
import six
import shutil
import pyonmttok

def build_tokenizer(args):
    """Builds a tokenizer based on user arguments."""
    local_args = {}
    for k, v in six.iteritems(args):
        if isinstance(v, six.string_types):
            local_args[k] = v.encode('utf-8')
        else:
            local_args[k] = v
    mode = local_args['mode']
    del local_args['mode']
    if 'vocabulary' in local_args:
        del local_args['vocabulary']
    if 'build_subword' in local_args:
        del local_args['build_subword']
    if 'build_vocabulary' in local_args:
        del local_args['build_vocabulary']

    return pyonmttok.Tokenizer(mode, **local_args)

def tokenize_file(tokenizer, input, output):
    """Tokenizes an input file."""
    if not tokenizer:
        shutil.copy(input, output)
    else:
        tokenizer.tokenize_file(input, output)

def detokenize_file(tokenizer, input, output):
    """Detokenizes an input file."""
    if not tokenizer:
        shutil.copy(input, output)
    else:
        tokenizer.detokenize_file(input, output)

def tokenize_directory(input_dir,
                       output_dir,
                       src_tokenizer,
                       tgt_tokenizer,
                       src_suffix,
                       tgt_suffix):
    """Tokenizes all files in input_dir into output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for f in files:
        if f.endswith(src_suffix):
            tokenizer = src_tokenizer
        elif f.endswith(tgt_suffix):
            tokenizer = tgt_tokenizer
        else:
            continue
        input_file = os.path.join(input_dir, f)
        output_file = os.path.join(output_dir, f)
        tokenize_file(tokenizer, input_file, output_file)

def tokenize(tokenizer, text):
    words,_ = tokenizer.tokenize(text)
    output = " ".join(words)
    return output

def make_subword_learner(subword_config, subword_dir):

    if 'params' not in subword_config:
        raise RuntimeError('Parameter field \'params\' should be specified for subword model learning.')
    params = subword_config['params']

    if 'type' not in subword_config:
        raise RuntimeError('\'type\' field should be specified for subword model learning.')
    subword_type = subword_config['type']

    if 'vocab_size' not in params :
        raise RuntimeError('\'vocab_size\' should be specified for subword model learning.')
    size = params['vocab_size']

    learner = None
    if (subword_type == "bpe"):
        min_frequency = params['min-frequency'] if 'min-frequency' in params else 0
        total_symbols = params['total_symbols'] if 'total_symbols' in params else False
        # If no tokenizer is specified, the default tokenizer is space mode.
        learner = pyonmttok.BPELearner(symbols=size, min_frequency=min_frequency, total_symbols=total_symbols)
    elif (subword_type == "sp"):
        learner = pyonmttok.SentencePieceLearner(**params)
    else:
        raise RuntimeError('Invalid subword type : \'%s\'.' % subword_type)

    return { "learner": learner, "subword_type": subword_type, "size": size }
