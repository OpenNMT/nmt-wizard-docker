import itertools
import collections


from nmtwizard.preprocess import prepoperator


@prepoperator.register_operator("parentheses")
class ParenthesesFilter(prepoperator.Filter):
    @classmethod
    def _config_schema(cls):
        schema = super(ParenthesesFilter, cls)._config_schema()

        schema["properties"].update(
            {
                "side": {"type": "string", "enum": ["source", "target", "both"]},
                "type": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            }
        )

        schema["required"] = ["side"]

        return schema

    def __init__(self, config, process_type, build_state):
        side = config.get("side")
        self._remove_src = side == "source" or side == "both"
        self._remove_tgt = side == "target" or side == "both"

        self._parentheses_types = {("(", ")")}

        parentheses_types = config.get("type")
        if parentheses_types:
            for par in parentheses_types:
                self._parentheses_types.add(tuple(par))

        filters = [self._filter_parentheses]

        super(ParenthesesFilter, self).__init__(filters)

    def _discover_parentheses(self, tokens):
        replacements = collections.defaultdict(list)
        for parentheses_type in self._parentheses_types:
            opening, closing = parentheses_type
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if closing in tok:
                    return None
                if opening in tok:
                    for j in range(i + 1, len(tokens)):
                        tok_after = tokens[j]
                        if closing in tok_after:
                            joiner_marker = "￭"
                            repl = []
                            if tok.startswith(joiner_marker) and tok_after.endswith(
                                joiner_marker
                            ):
                                repl.append(joiner_marker)
                            replacements[parentheses_type].append((i, j - i + 1, repl))
                            i = j
                            break
                        if j == len(tokens) - 1:  # Didn't find a pair
                            return None
                        for par in itertools.chain(
                            *self._parentheses_types
                        ):  # Found a nested/mismatched parenthesis after
                            if par in tok_after:
                                return None
                i += 1
        return replacements

    def _filter_parentheses(self, tu):
        src_tokens = tu.src_tok.tokens[0]
        src_replacements = self._discover_parentheses(src_tokens)
        if src_replacements is None:  # Unbalanced or nested parentheses in source
            return True

        tgt_tokens = tu.tgt_tok.tokens[0]
        tgt_replacements = self._discover_parentheses(tgt_tokens)
        if tgt_replacements is None:  # Unbalanced or nested parentheses in target
            return True

        src_replacements_to_keep = []
        tgt_replacements_to_keep = []
        for parentheses_type in self._parentheses_types:
            src_repl = src_replacements[parentheses_type]
            tgt_repl = tgt_replacements[parentheses_type]
            length_src_repl = len(src_repl)
            length_tgt_repl = len(tgt_repl)
            if length_src_repl != length_tgt_repl and (
                length_src_repl > 1 or length_tgt_repl > 1
            ):  # Unabalanced source/target
                return True

            if self._remove_src and length_src_repl == 1 and length_tgt_repl == 0:
                src_replacements_to_keep.append(src_repl[0])

            if self._remove_tgt and length_tgt_repl == 1 and length_src_repl == 0:
                tgt_replacements_to_keep.append(tgt_repl[0])

        src_replacements_to_keep.sort(key=lambda tup: tup[0])
        tgt_replacements_to_keep.sort(key=lambda tup: tup[0])

        for repl in reversed(src_replacements_to_keep):
            tu.replace_tokens_side("source", repl)

        for repl in reversed(tgt_replacements_to_keep):
            tu.replace_tokens_side("target", repl)

        return False
