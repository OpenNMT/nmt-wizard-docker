import tempfile

from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess import tokenizer


@prepoperator.register_operator("tokenization")
class Tokenizer(prepoperator.MonolingualOperator):
    @property
    def _detok(self):
        return False

    def _build_process(self, config, side, build_state):
        # Disable subword regularization in inference.
        if self.process_type != prepoperator.ProcessType.TRAINING:
            config["bpe_dropout"] = 0
            config["sp_nbest_size"] = 0
            config["sp_alpha"] = 0

        if config.get("restrict_subword_vocabulary", False):
            vocabulary_path = build_state.get(
                "src_vocabulary" if side == "source" else "tgt_vocabulary"
            )
            if vocabulary_path is None:
                raise ValueError(
                    "restrict_subword_vocabulary is set but no vocabulary is set"
                )

            # The open source Tokenizer does not accept the custom vocabulary format
            # produced by build_vocab so we create a temporary vocabulary with a simpler
            # format.
            with tempfile.NamedTemporaryFile(mode="w") as vocab_file:
                for token in tokenizer.load_vocabulary(vocabulary_path):
                    vocab_file.write("%s\n" % token)
                vocab_file.flush()
                config["vocabulary_path"] = vocab_file.name
                current_tokenizer = tokenizer.build_tokenizer(config)
        else:
            current_tokenizer = tokenizer.build_tokenizer(config)

        previous_tokenizer = None
        if build_state:
            if side == "source":
                previous_tokenizer = build_state["src_tokenizer"]
                build_state["src_tokenizer"] = current_tokenizer
            else:
                previous_tokenizer = build_state["tgt_tokenizer"]
                build_state["tgt_tokenizer"] = current_tokenizer
        if (
            self.process_type == prepoperator.ProcessType.POSTPROCESS
            and not self._postprocess_only
        ):
            return previous_tokenizer
        return current_tokenizer

    def _apply_process(self, tokenizer, src_tok):
        return (tokenizer, None)
