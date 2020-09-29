import random

from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess.tu import TokReplace

@prepoperator.register_operator("noise")
class Noise(prepoperator.TUOperator):

    @staticmethod
    def is_applied_for(process_type):
        return process_type == prepoperator.ProcessType.TRAINING

    def __init__(self, config, *args):
        self._drop_word_prob = config.get("drop_word_prob", 0)
        self._drop_space_prob = config.get("drop_space_prob", 0)
        self._drop_char_prob = config.get("drop_char_prob", 0)
        self._duplicate_char_prob = config.get("duplicate_char_prob", 0)
        self._swap_char_prob = config.get("swap_char_prob", 0)

    def _preprocess_tu(self, tu, *args):
        tokens = tu.src_tok.token_objects

        index_to_delete = self._apply_word_noise(tokens[0])
        if index_to_delete:
            self._delete_tokens(tu, index_to_delete)
        else:
            # Invalidate detokenized representation because some token surface may have changed.
            tu.src_detok = None

        return [tu]

    def _delete_tokens(self, tu, index_to_delete):
        for i, index in enumerate(index_to_delete):
            replacement = TokReplace(start_tok_idx=index - i, tok_num=1, new_tokens=[])
            tu.replace_tokens(src_replace=replacement)

    def _apply_word_noise(self, tokens):
        index_to_delete = []
        for i, token in enumerate(tokens):
            if self._drop_word_prob > 0 and random.random() <= self._drop_word_prob:
                token.surface = ""
            elif self._drop_space_prob > 0 and random.random() <= self._drop_space_prob:
                token.join_left = True

            if not token.is_placeholder():
                token.surface = self._apply_character_noise(token.surface)
            if len(token.surface) == 0:  # Delete token if empty.
                index_to_delete.append(i)
        return index_to_delete

    def _apply_character_noise(self, cur_surface):
        new_surface = ""
        i = 0
        while i < len(cur_surface):
            if self._drop_char_prob > 0 and random.random() <= self._drop_char_prob:
                pass
            elif self._duplicate_char_prob > 0 and random.random() <= self._duplicate_char_prob:
                new_surface += cur_surface[i] * 2
            elif (self._swap_char_prob > 0
                  and i + 1 < len(cur_surface)
                  and random.random() <= self._swap_char_prob):
                new_surface += cur_surface[i + 1]
                new_surface += cur_surface[i]
                i += 1
            else:
                new_surface += cur_surface[i]
            i += 1
        return new_surface
