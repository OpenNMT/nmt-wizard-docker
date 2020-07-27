# coding: utf-8
"""Tokenization utilities."""

import os
import shutil
import pyonmttok

joiner_marker = "￭"

def build_tokenizer(args):
    """Builds a tokenizer based on user arguments."""
    args = args.copy()
    args.pop('vocabulary', None)
    args.pop('build_subword', None)
    args.pop('build_vocabulary', None)
    if not args:
        return None
    return pyonmttok.Tokenizer(**args)

def tokenize_file(tokenizer, input_file, output_file):
    """Tokenizes an input file."""
    if not tokenizer:
        shutil.copy(input_file, output_file)
    else:
        tokenizer.tokenize_file(input_file, output_file)

def detokenize_file(tokenizer, input_file, output_file):
    """Detokenizes an input file."""
    if not tokenizer:
        shutil.copy(input_file, output_file)
    else:
        tokenizer.detokenize_file(input_file, output_file)

def tokenize_directory(input_dir,
                       output_dir,
                       src_tokenizer,
                       tgt_tokenizer,
                       src_suffix,
                       tgt_suffix):
    """Tokenizes all files in input_dir into output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        if not os.path.isfile(input_file):
            continue
        if filename.endswith(src_suffix):
            tokenizer = src_tokenizer
        elif filename.endswith(tgt_suffix):
            tokenizer = tgt_tokenizer
        else:
            continue
        output_file = os.path.join(output_dir, filename)
        tokenize_file(tokenizer, input_file, output_file)

def tokenize(tokenizer, text):
    words,_ = tokenizer.tokenize(text)
    output = " ".join(words)
    return output

def make_subword_learner(subword_config, subword_dir, tokenizer=None):
    params = subword_config.get('params')
    if params is None:
        raise ValueError('\'params\' field should be specified for subword model learning.')
    subword_type = subword_config.get('type')
    if subword_type is None:
        raise ValueError('\'type\' field should be specified for subword model learning.')
    vocab_size = params.get('vocab_size')
    if vocab_size is None:
        raise ValueError('\'vocab_size\' parameter should be specified for subword model learning.')

    if subword_type == "bpe":
        learner = pyonmttok.BPELearner(
            tokenizer=tokenizer,
            symbols=vocab_size,
            min_frequency=params.get('min-frequency', 0),
            total_symbols=params.get('total_symbols', False))
    elif subword_type == "sp":
        learner = pyonmttok.SentencePieceLearner(tokenizer=tokenizer, **params)
    else:
        raise ValueError('Invalid subword type : \'%s\'.' % subword_type)

    return {
        "learner": learner,
        "subword_type": subword_type,
        "size": vocab_size
    }
