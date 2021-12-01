import random
import copy
import os
import logging

import pyonmttok

from nmtwizard.logger import get_logger
from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess.tu import TokReplace
import fasttext

logger = get_logger(__name__)


class Noiser:
    def __init__(self, config):
        self._drop_word_prob = config.get("drop_word_prob", 0)
        self._duplicate_word_prob = config.get("duplicate_word_prob", 0)
        self._swap_word_prob = config.get("swap_word_prob", 0)
        substitute_word_config = config.get("substitute_word", None)
        self._substitute_word_prob = 0
        if substitute_word_config:
            self._substitute_word_prob = substitute_word_config.get("prob", 0)
            if self._substitute_word_prob:
                word_embedding_file = substitute_word_config.get("word_embedding_file")
                self._word_embedding_model = None
                if word_embedding_file is not None:
                    if not os.path.isfile(word_embedding_file):
                        raise ValueError(
                            "Word embedding file doesn't exist: %s"
                            % (word_embedding_file)
                        )
                    self._word_embedding_model = fasttext.load_model(
                        word_embedding_file
                    )
                    self._nn = substitute_word_config.get("nearest_neighbors_num")
        self._drop_space_prob = config.get("drop_space_prob", 0)
        self._insert_space_prob = config.get("insert_space_prob", 0)
        self._drop_char_prob = config.get("drop_char_prob", 0)
        self._duplicate_char_prob = config.get("duplicate_char_prob", 0)
        self._swap_char_prob = config.get("swap_char_prob", 0)
        self._substitute_char_prob = config.get("substitute_char_prob", 0)

    def apply_noise(self, tokens):
        new_tokens = []
        for token in tokens:
            if not token.is_placeholder():
                if self._drop_word_prob > 0 and random.random() <= self._drop_word_prob:
                    continue
                # TODO : joiners
                elif (
                    self._duplicate_word_prob > 0
                    and random.random() <= self._duplicate_word_prob
                ):
                    new_tokens.extend([token, token])
                    continue
                elif (
                    len(new_tokens) > 0
                    and self._swap_word_prob > 0
                    and random.random() <= self._swap_word_prob
                ):
                    new_tokens.insert(-1, token)
                    continue
                elif (
                    self._substitute_word_prob > 0
                    and self._word_embedding_model is not None
                    and random.random() <= self._substitute_word_prob
                    and all(c.isalpha() for c in token.surface)
                ):
                    nearest_neighbors = (
                        self._word_embedding_model.get_nearest_neighbors(
                            token.surface, k=self._nn
                        )
                    )
                    nearest_neighbors = [
                        nn[1]
                        for nn in nearest_neighbors
                        if all(c.isalpha() for c in nn[1])
                    ]
                    if nearest_neighbors:
                        token.surface = random.choice(nearest_neighbors)
                    new_tokens.append(token)
                    continue
                elif (
                    self._insert_space_prob > 0
                    and random.random() <= self._insert_space_prob
                    and len(token) > 1
                ):
                    new_space_index = random.randint(1, len(token) - 1)
                    first_part_surface = token.surface[0:new_space_index]
                    second_part_surface = token.surface[new_space_index:]
                    token.surface = first_part_surface
                    second_part_token = pyonmttok.Token(token)
                    second_part_token.surface = second_part_surface
                    new_tokens.extend([token, second_part_token])
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
                    token.surface = self.apply_character_noise(token.surface)
            if len(token.surface) != 0:  # Delete token if empty.
                new_tokens.append(token)
        return new_tokens

    def apply_noise_batch(self, tokens_batch):
        new_tokens_batch = []
        for tokens in tokens_batch:
            new_tokens_batch.append(self.apply_noise(tokens))
        return new_tokens_batch

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

    def apply_character_noise(self, cur_surface):
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


@prepoperator.register_operator("noise")
class Noise(prepoperator.Operator):
    @classmethod
    def _config_schema(cls):
        schema = super(Noise, cls)._config_schema()

        noise_block = {
            "lang": {"type": "string"},
            "drop_word_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "duplicate_word_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "swap_word_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "substitute_word": {
                "properties": {
                    "prob": {"type": "number", "minimum": 0, "maximum": 1},
                    "word_embedding_file": {"type": "string"},
                    "nearest_neighbors_num": {"type": "integer"},
                },
                "type": "object",
                "additionalProperties": False,
            },
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

    def get_shared_classes():
        return [Noiser]

    @staticmethod
    def get_shared_builders(config, process_type):
        # Only build noiser as shared object for word substitution with embeddings
        word_emb = config.get("substitute_word", {}).get("word_embedding_file")
        if word_emb:
            return {"noiser": (Noiser, (config,))}
        else:
            return None

    def __init__(self, config, process_type, build_state, shared_state=None):
        source_config = config.get("source")
        if source_config:
            config = source_config
        self._noiser = shared_state.get("noiser") if shared_state else None
        if not self._noiser:
            self._noiser = Noiser(config)
        self._add_marker = config.get("add_marker", 0)

    def _preprocess(self, tu_batch):
        tu_list, meta_batch = tu_batch

        src_tokens = []
        src_detok = []
        for tu in tu_list:
            src_tok = tu.src_tok
            src_tokens.append(src_tok.token_objects[0])
            src_detok.append(tu.src_detok)

        src_tokens_noisy = self._noiser.apply_noise_batch(src_tokens)

        for detok, tok_noisy, tu in zip(src_detok, src_tokens_noisy, tu_list):
            src_tok = tu.src_tok
            tu.src_tok = (src_tok.tokenizer, [tok_noisy])
            new_detok = tu.src_detok
            if detok != new_detok:
                if self._add_marker:
                    tu.replace_tokens_side("source", (0, 0, ["｟mrk_noisy｠"]))
                log_level = logging.INFO if self._verbose else logging.DEBUG
                if logger.isEnabledFor(log_level):
                    logger.info(
                        "'%s' operator modifies source in preprocess.\nSRC BEFORE : %s\nSRC AFTER  : %s",
                        self.name,
                        detok,
                        new_detok,
                    )

        return tu_list, meta_batch
