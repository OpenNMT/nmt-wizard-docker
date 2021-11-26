import random
import copy
import os

from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess.tu import TokReplace
import fasttext


@prepoperator.register_operator("noise")
class Noise(prepoperator.TUOperator):
    @classmethod
    def _config_schema(cls):
        schema = super(Noise, cls)._config_schema()

        noise_block = {
            "lang": {"type": "string"},
            "drop_word_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "duplicate_word_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "swap_word_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "substitute_word_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "word_embedding_file": {"type": "string"},
            "drop_space_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "insert_space_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "drop_char_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "duplicate_char_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "swap_char_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "substitute_char_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "add_marker": {"type": "boolean"},
        }
        schema["properties"].update(
            {
                "source": {
                    "type": "object",
                    "properties": noise_block,
                    "additionalProperties": False,
                }
            }
        )
        schema["properties"].update(noise_block)

        return schema

    @staticmethod
    def is_applied_for(process_type):
        return process_type == prepoperator.ProcessType.TRAINING

    def __init__(self, config, *args):
        source_config = config.get("source")
        if source_config:
            config = source_config
        self._drop_word_prob = config.get("drop_word_prob", 0)
        self._duplicate_word_prob = config.get("duplicate_word_prob", 0)
        self._swap_word_prob = config.get("swap_word_prob", 0)
        self._substitute_word_prob = config.get("substitute_word_prob", 0)
        # TODO: SharedState builder ?
        self._word_embedding_file = config.get("word_embedding_file")
        self._word_embedding_model = None
        if self._word_embedding_file is not None:
            if not os.path.isfile(self._word_embedding_file):
                raise ValueError(
                    "Word embedding file doesn't exist: %s"
                    % (self._word_embedding_file)
                )
            self._word_embedding_model = fasttext.load_model(self._word_embedding_file)
        self._drop_space_prob = config.get("drop_space_prob", 0)
        self._insert_space_prob = config.get("insert_space_prob", 0)
        self._drop_char_prob = config.get("drop_char_prob", 0)
        self._duplicate_char_prob = config.get("duplicate_char_prob", 0)
        self._swap_char_prob = config.get("swap_char_prob", 0)
        self._substitute_char_prob = config.get("substitute_char_prob", 0)
        self._add_marker = config.get("add_marker", 0)

    def _preprocess_tu(self, tu, *args):
        original_tokens = copy.deepcopy(tu.src_tok.token_objects)
        tu = self._apply_space_insertion_noise(tu)
        src_tok = tu.src_tok
        tokens = src_tok.token_objects
        new_tokens = [self._apply_word_noise(tokens[0])]
        tu.src_tok = (src_tok.tokenizer, new_tokens)
        if self._add_marker and new_tokens != original_tokens:
            tu.replace_tokens_side("source", (0, 0, ["｟mrk_noisy｠"]))
        return [tu]

    def _apply_space_insertion_noise(self, tu):
        src_tok = tu.src_tok
        tokens = src_tok.token_objects[0]
        added_spaces = 0
        for pos, token in enumerate(tokens):
            if not token.is_placeholder():
                if (
                    self._insert_space_prob > 0
                    and random.random() <= self._insert_space_prob
                    and len(token) > 1
                ):
                    new_space_index = random.randint(1, len(token) - 1)
                    first_part_surface = token.surface[0:new_space_index]
                    second_part_surface = token.surface[new_space_index:]
                    tu.replace_tokens_side(
                        "source",
                        (
                            pos + added_spaces,
                            1,
                            [first_part_surface, second_part_surface],
                        ),
                    )
                    added_spaces += 1
        return tu

    def _apply_word_noise(self, tokens):
        new_tokens = []
        for token in tokens:
            if not token.is_placeholder():
                if self._drop_word_prob > 0 and random.random() <= self._drop_word_prob:
                    continue
                # TODO : joiners
                elif self._duplicate_word_prob > 0 and random.random() <= self._duplicate_word_prob:
                    new_tokens.extend([token, token])
                    continue
                elif len(new_tokens) > 0 and self._swap_word_prob > 0 and random.random() <= self._swap_word_prob:
                    new_tokens.insert(-1, token)
                    continue
                elif self._word_embedding_model is not None and self._substitute_word_prob > 0 and random.random() <= self._substitute_word_prob and all(c.isalpha() for c in token.surface):
                    nearest_neighbors = self._word_embedding_model.get_nearest_neighbors(token.surface, k=5) # TODO : define k as an option
                    nearest_neighbors = [nn[1] for nn in nearest_neighbors if all(c.isalpha() for c in nn[1])]
                    token.surface = random.choice(nearest_neighbors)
                    new_tokens.append(token)
                    continue
                elif (
                    self._drop_space_prob > 0
                    and random.random() <= self._drop_space_prob
                ):
                    token.join_left = True

                if (
                    self._drop_char_prob > 0
                    or self._duplicate_char_prob > 0
                    or self._swap_char_prob > 0
                    or self._substitute_char_prob > 0
                ):
                    token.surface = self._apply_character_noise(token.surface)
            if len(token.surface) != 0:  # Delete token if empty.
                new_tokens.append(token)
        return new_tokens

    @staticmethod
    def get_neighbor_keys_on_qwerty(key):
        lines = "qwertyuiop", "asdfghjkl", "zxcvbnm"
        line_index, index = [(i, l.find(key)) for i, l in enumerate(lines) if key in l][
            0
        ]
        lines = lines[line_index - 1 : line_index + 2] if line_index else lines[0:2]
        return [
            line[index + i]
            for line in lines
            for i in [-1, 0, 1]
            if len(line) > index + i and line[index + i] != key and index + i >= 0
        ]

    def _apply_character_noise(self, cur_surface):
        new_surface = ""
        i = 0
        while i < len(cur_surface):
            if self._drop_char_prob > 0 and random.random() <= self._drop_char_prob:
                pass
            elif (
                self._duplicate_char_prob > 0
                and random.random() <= self._duplicate_char_prob
            ):
                new_surface += cur_surface[i] * 2
            elif (
                self._swap_char_prob > 0
                and i + 1 < len(cur_surface)
                and random.random() <= self._swap_char_prob
            ):
                new_surface += cur_surface[i + 1]
                new_surface += cur_surface[i]
                i += 1
            elif (
                self._substitute_char_prob > 0
                and random.random() <= self._substitute_char_prob
                and cur_surface[i].isalpha()
            ):
                neighbors = self.get_neighbor_keys_on_qwerty(cur_surface[i])
                new_surface += random.choice(neighbors)
            else:
                new_surface += cur_surface[i]
            i += 1
        return new_surface
