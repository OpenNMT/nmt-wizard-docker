from nmtwizard.preprocess import prepoperator


@prepoperator.register_operator("length_filter")
class LengthFilter(prepoperator.Filter):
    @classmethod
    def _config_schema(cls):
        schema = super(LengthFilter, cls)._config_schema()

        length_mono_block = {
            "type": "object",
            "properties": {
                "lang": {"type": "string"},
                "max_characters": {"type": "integer", "minimum": 0},
                "max_words": {"type": "integer", "minimum": 0},
                "min_words": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        }
        schema["properties"].update(
            {
                "source": length_mono_block,
                "target": length_mono_block,
                "min_words_ratio": {"type": "number"},
                "max_words_ratio": {"type": "number"},
                "min_num_words_for_ratio": {"type": "integer"},
            }
        )
        return schema

    def __init__(self, config, process_type, build_state):
        source_config = _get_side_config(config, "source")
        target_config = _get_side_config(config, "target")
        self._verbose = config.get("verbose", False)

        filters = []
        filters.extend(
            _get_side_filters(
                source_config,
                lambda tu: tu.src_detok,
                lambda tu: tu.src_tok.tokens[0],
                self._verbose,
            )
        )
        filters.extend(
            _get_side_filters(
                target_config,
                lambda tu: tu.tgt_detok,
                lambda tu: tu.tgt_tok.tokens[0],
                self._verbose,
            )
        )

        min_words_ratio = config.get("min_words_ratio")
        min_num_words_for_ratio = config.get("min_num_words_for_ratio", 0)

        if min_words_ratio is not None:
            message_min_words_ratio = "Inferior to min word length ratio (%.2f) (Src length : %d Tgt length : %d Ratio : %.2f)"
            filters.append(
                lambda tu: (
                    len(tu.src_tok.tokens[0]) >= min_num_words_for_ratio
                    and len(tu.tgt_tok.tokens[0]) >= min_num_words_for_ratio
                    and len(tu.src_tok.tokens[0]) / len(tu.tgt_tok.tokens[0])
                    < min_words_ratio,
                    message_min_words_ratio
                    % (
                        min_words_ratio,
                        len(tu.src_tok.tokens[0]),
                        len(tu.tgt_tok.tokens[0]),
                        len(tu.src_tok.tokens[0]) / len(tu.tgt_tok.tokens[0]),
                    ),
                )
            )

        max_words_ratio = config.get("max_words_ratio")
        if max_words_ratio is not None:
            message_max_words_ratio = "Exceeds max word length ratio (%.2f) (Src length : %d Tgt length : %d Ratio : %.2f)"
            filters.append(
                lambda tu: (
                    len(tu.src_tok.tokens[0]) >= min_num_words_for_ratio
                    and len(tu.tgt_tok.tokens[0]) >= min_num_words_for_ratio
                    and len(tu.src_tok.tokens[0]) / len(tu.tgt_tok.tokens[0])
                    > max_words_ratio,
                    message_max_words_ratio
                    % (
                        max_words_ratio,
                        len(tu.src_tok.tokens[0]),
                        len(tu.tgt_tok.tokens[0]),
                        len(tu.src_tok.tokens[0]) / len(tu.tgt_tok.tokens[0]),
                    ),
                )
            )

        super(LengthFilter, self).__init__(filters)


def _get_side_config(config, side):
    config = config.get(side, {})
    # Filter empty sentences by default.
    config.setdefault("min_words", 1)
    return config


def _get_side_filters(config, chars_fn, words_fn, verbose):
    filters = []

    max_chars = config.get("max_characters")
    if max_chars is not None:
        message_max_chars = f"Longer than max chars ({max_chars})"
        filters.append(lambda tu: (len(chars_fn(tu)) > max_chars, message_max_chars))

    max_words = config.get("max_words")
    if max_words is not None:
        message_max_words = f"Longer than max words ({max_words})"
        filters.append(lambda tu: (len(words_fn(tu)) > max_words, message_max_words))

    min_words = config.get("min_words")
    if min_words is not None:
        message_min_words = f"Shorter than min words ({min_words})"
        filters.append(lambda tu: (len(words_fn(tu)) < min_words, message_min_words))

    return filters
