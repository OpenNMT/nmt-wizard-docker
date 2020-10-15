import math

from nmtwizard.preprocess import prepoperator

@prepoperator.register_operator("align_perplexity_filter")
class AlignPerplexityFilter(prepoperator.Filter):

    def __init__(self, config, process_type, build_state):
        super().__init__([])
        self._hard_threshold = config.get("hard_threshold")
        self._percent_threshold = config.get("percent_threshold")

        if self._hard_threshold is not None:
            self._lower = self._hard_threshold.get("lower")
            self._upper = self._hard_threshold.get("upper")

        elif self._percent_threshold is not None:
            self._lower = self._percent_threshold.get("lower", 0)
            self._upper = self._percent_threshold.get("upper", 0)

    def _preprocess(self, tu_batch):
        if self._hard_threshold is None and self._percent_threshold is None:
            return tu_batch

        tu_list, meta_batch = tu_batch
        batch_size = len(tu_list)
        perplexity = list(map(_compute_perplexity, tu_list))
        new_tu_list = []

        if self._hard_threshold is not None:
            # Filter TUs on perplexity value.
            for tu, perplexity in zip(tu_list, perplexity):
                if ((self._lower is None or perplexity >= self._lower)
                    and (self._upper is None or perplexity <= self._upper)):
                    new_tu_list.append(tu)

        elif self._percent_threshold is not None:
            # Remove the worst $lower percent and the best $upper percent perplexity values.
            keep_ids = range(batch_size)
            keep_ids = list(sorted(keep_ids, key=lambda i: perplexity[i]))  # From best to worst.
            worst_to_remove = int(self._lower * batch_size)
            best_to_remove = int(self._upper * batch_size)
            if worst_to_remove != 0:
                keep_ids = keep_ids[:-worst_to_remove]
            if best_to_remove != 0:
                keep_ids = keep_ids[best_to_remove:]
            for i in sorted(keep_ids):
                new_tu_list.append(tu_list[i])

        return new_tu_list, meta_batch


def _compute_perplexity(tu):
    # Compute the average source/target perplexity.
    fwd, bwd = _get_log_probs(tu)
    src_size = len(tu.src_tok.token_objects[0])
    tgt_size = len(tu.tgt_tok.token_objects[0])
    return (math.exp(-fwd / src_size) + math.exp(-bwd / tgt_size)) / 2

def _get_log_probs(tu):
    log_probs = tu.alignment_log_probs
    if log_probs is None:
        raise ValueError("Alignment log probs are not set")
    return log_probs[0]
