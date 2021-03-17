import math

from nmtwizard.preprocess import prepoperator

@prepoperator.register_operator("align_perplexity_filter")
class AlignPerplexityFilter(prepoperator.Filter):

    def __init__(self, config, process_type, build_state):
        super().__init__([])
        self._hard_threshold = config.get("hard_threshold")
        self._percent_threshold = config.get("percent_threshold")

        if self._hard_threshold is not None:
            self._lower = _get_hard_threshold(self._hard_threshold, "lower")
            self._upper = _get_hard_threshold(self._hard_threshold, "upper")
            if self._lower is not None and self._upper is not None and self._upper <= self._lower:
                raise ValueError("align_perplexity_filter: hard threshold 'upper' should be "
                                 "greater than 'lower'")

        elif self._percent_threshold is not None:
            self._lower = _get_percent_threshold(self._percent_threshold, "lower")
            self._upper = _get_percent_threshold(self._percent_threshold, "upper")
            total_removed = self._lower + self._upper
            if total_removed >= 1:
                raise ValueError("align_perplexity_filter: percent threshold values will filter "
                                 "all sentences (lower=%.2f and upper=%.2f mean %.2f%% of sentences "
                                 "will be filtered)" % (self._lower, self._upper, total_removed * 100))

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
            keep_ids = list(sorted(keep_ids, key=lambda i: perplexity[i], reverse=True))  # From best to worst.
            worst_to_remove = int(self._lower * batch_size)
            best_to_remove = int(self._upper * batch_size)
            if worst_to_remove != 0:
                keep_ids = keep_ids[:-worst_to_remove]
            if best_to_remove != 0:
                keep_ids = keep_ids[best_to_remove:]
            for i in sorted(keep_ids):
                new_tu_list.append(tu_list[i])

        return new_tu_list, meta_batch


def _get_hard_threshold(config, field):
    value = config.get(field)
    if value is not None and value > 0:
        raise ValueError("align_perplexity_filter: perplexity values range from "
                         "-inf (worst perplexity) to 0 (best perplexity), but hard "
                         "threshold '%s' is set to %.2f" % (field, value))
    return value

def _get_percent_threshold(config, field):
    value = config.get(field, 0)
    if value < 0 or value >= 1:
        raise ValueError("align_perplexity_filter: percent threshold should be between "
                         "0 (included) and 1 (excluded), but '%s' is set to %.2f" % (field, value))
    return value

def _compute_perplexity(tu):
    # Compute the average source/target perplexity.
    fwd, bwd = _get_log_probs(tu)

    src_size = len(tu.src_tok.tokens[0])
    tgt_size = len(tu.tgt_tok.tokens[0])

    min_size = min(src_size, tgt_size) or 1

    return math.log((math.exp(fwd / min_size) + math.exp(bwd / min_size)) / 2)

def _get_log_probs(tu):
    log_probs = tu.alignment_log_probs
    if log_probs is None:
        raise ValueError("Alignment log probs are not set")
    return log_probs[0]
