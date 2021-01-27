from nmtwizard.preprocess import prepoperator

@prepoperator.register_operator("length_filter")
class LengthFilter(prepoperator.Filter):

    def __init__(self, config, process_type, build_state):
        source_config = _get_side_config(config, 'source')
        target_config = _get_side_config(config, 'target')

        filters = []
        filters.extend(_get_side_filters(
            source_config,
            lambda tu: tu.src_detok,
            lambda tu: tu.src_tok.tokens[0]))
        filters.extend(_get_side_filters(
            target_config,
            lambda tu: tu.tgt_detok,
            lambda tu: tu.tgt_tok.tokens[0]))


        min_words_ratio = config.get('min_words_ratio')
        if min_words_ratio is not None:
            filters.append(lambda tu: (
                len(tu.src_tok.tokens[0]) / len(tu.tgt_tok.tokens[0]) < min_words_ratio))

        max_words_ratio = config.get('max_words_ratio')
        if max_words_ratio is not None:
            filters.append(lambda tu: (
                len(tu.src_tok.tokens[0]) / len(tu.tgt_tok.tokens[0]) > max_words_ratio))

        super(LengthFilter, self).__init__(filters)


def _get_side_config(config, side):
    config = config.get(side, {})
    # Filter empty sentences by default.
    config.setdefault('min_words', 1)
    return config

def _get_side_filters(config, chars_fn, words_fn):
    filters = []

    max_chars = config.get('max_characters')
    if max_chars is not None:
        filters.append(lambda tu: len(chars_fn(tu)) > max_chars)

    max_words = config.get('max_words')
    if max_words is not None:
        filters.append(lambda tu: len(words_fn(tu)) > max_words)

    min_words = config.get('min_words')
    if min_words is not None:
        filters.append(lambda tu: len(words_fn(tu)) < min_words)

    return filters
