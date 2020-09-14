# coding: utf-8
"""Tokenization utilities."""

import pyonmttok

joiner_marker = "￭"

_ALLOWED_TOKENIZER_ARGS = set([
    "bpe_dropout",
    "bpe_model_path",
    "case_feature",
    "case_markup",
    "joiner",
    "joiner_annotate",
    "joiner_new",
    "mode",
    "no_substitution",
    "preserve_placeholders",
    "preserve_segmented_tokens",
    "segment_alphabet",
    "segment_alphabet_change",
    "segment_case",
    "segment_numbers",
    "sp_alpha",
    "sp_model_path",
    "sp_nbest_size",
    "spacer_annotate",
    "spacer_new",
    "support_prior_joiners",
    "vocabulary_path",
    "vocabulary_threshold",
])

def build_tokenizer(args):
    """Builds a tokenizer based on user arguments."""
    args = {name:value for name, value in args.items() if name in _ALLOWED_TOKENIZER_ARGS}
    if not args:
        return None
    return pyonmttok.Tokenizer(**args)

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
