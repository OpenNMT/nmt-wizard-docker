from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess import tokenizer

@prepoperator.register_operator("tokenization")
class Tokenizer(prepoperator.MonolingualOperator):

    @property
    def _detok(self):
        return False

    def _build_processor(self, config, side, build_state):
        current_tokenizer = tokenizer.build_tokenizer(config)
        previous_tokenizer = None
        if build_state:
            if side == "source":
                previous_tokenizer = build_state["src_tokenizer"]
                build_state["src_tokenizer"] = current_tokenizer
            else:
                previous_tokenizer = build_state["tgt_tokenizer"]
                build_state["tgt_tokenizer"] = current_tokenizer
        if self._process_type == prepoperator.ProcessType.POSTPROCESS and not self._postprocess_only:
            return previous_tokenizer
        else:
            return current_tokenizer


    def _apply_processor(self, tokenizer, src_tok):
        return (tokenizer, None)
