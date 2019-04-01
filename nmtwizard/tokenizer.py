"""Tokenization utilities."""

import os
import six
import shutil
import gzip

def build_tokenizer(args):
    """Builds a tokenizer based on user arguments."""
    import pyonmttok
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
    return pyonmttok.Tokenizer(mode, **local_args)

def tokenize_file(tokenizer, input, output):
    """Tokenizes an input file."""
    if not tokenizer:
        shutil.copy(input, output)
    else:
        open_func = open
        if input.endswith(".gz"):
            open_func = gzip.open
        with open_func(input, 'rb') as input_file, open(output, 'w') as output_file:
            for line in input_file:
                line = line.strip().decode('utf-8', 'ignore').encode('utf-8')
                tokens, _ = tokenizer.tokenize(line)
                output_file.write(' '.join(tokens))
                output_file.write('\n')

def detokenize_file(tokenizer, input, output):
    """Detokenizes an input file."""
    open_func = open
    if output.endswith(".gz"):
        open_func = gzip.open
    with open(input, 'r') as input_file, open_func(output, 'w') as output_file:
        for line in input_file:
            if tokenizer:
                text = tokenizer.detokenize(line.strip().split())
            else:
                text = line.strip()
            output_file.write(text)
            output_file.write('\n')

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
