import os
import tempfile

from nmtwizard.preprocess import prepoperator
from nmtwizard.preprocess import tokenizer


@prepoperator.register_operator("tokenization")
class Tokenizer(prepoperator.MonolingualOperator):
    @classmethod
    def _config_schema(cls):
        schema = super(Tokenizer, cls)._config_schema()

        tokenization_block = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["aggressive", "conservative", "char", "space", "none"],
                },
                "no_substitution": {"type": "boolean"},
                "case_feature": {"type": "boolean"},
                "case_markup": {"type": "boolean"},
                "soft_case_regions": {"type": "boolean"},
                "lang": {"type": "string"},
                "bpe_model_path": {"type": "string", "is_file": True},
                "bpe_dropout": {"type": "number"},
                "sp_model_path": {"type": "string", "is_file": True},
                "sp_nbest_size": {"type": "integer"},
                "sp_alpha": {"type": "number"},
                "joiner_annotate": {"type": "boolean"},
                "joiner": {"type": "string"},
                "joiner_new": {"type": "boolean"},
                "spacer_annotate": {"type": "boolean"},
                "spacer_new": {"type": "boolean"},
                "preserve_placeholders": {"type": "boolean"},
                "preserve_segmented_tokens": {"type": "boolean"},
                "support_prior_joiners": {"type": "boolean"},
                "segment_case": {"type": "boolean"},
                "segment_numbers": {"type": "boolean"},
                "segment_alphabet": {"type": "array", "items": {"type": "string"}},
                "segment_alphabet_change": {"type": "boolean"},
                "restrict_subword_vocabulary": {"type": "boolean"},
                "build_vocabulary": {"type": "object"},
                "build_subword": {"type": ["object", "null"]},
            },
            "additionalProperties": False,
        }

        schema["properties"].update(
            {
                "source": {**tokenization_block, "required": ["mode"]},
                "target": {**tokenization_block, "required": ["mode"]},
                "multi": tokenization_block,
            }
        )
        return schema

    @property
    def _detok(self):
        return False

    @property
    def _apply_in_postprocess(self):
        return True

    def _build_process(self, config, side, build_state):
        # Disable subword regularization in inference.
        if not self.process_type.training:
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
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False
            ) as vocab_file:
                for token in tokenizer.load_vocabulary(vocabulary_path):
                    vocab_file.write("%s\n" % token)
                vocab_file.flush()
                config["vocabulary_path"] = vocab_file.name
                current_tokenizer = tokenizer.build_tokenizer(config)
            os.unlink(vocab_file.name)
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
        if self.process_type.postprocess and not self._postprocess_only:
            return previous_tokenizer
        return current_tokenizer

    def _apply_process(self, tokenizer, tok):
        return (tokenizer, None)
