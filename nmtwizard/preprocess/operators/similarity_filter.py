import random
import math

from nmtwizard.preprocess import prepoperator

@prepoperator.register_operator("similarity_filter")
class SimilarityFilter(prepoperator.Filter):

    def __init__(self, config, process_type, build_state):
        threshold = config.get("threshold", 0)
        mode = config.get("mode")
        factor = config.get("factor", 1)

        if mode is None:
            raise ValueError("Missing mode field in similarity configuration")
        if mode not in ("hard", "soft_linear", "soft_sigmoid"):
            raise ValueError("Invalid mode %s in similarity configuration" % mode)

        def _filter(tu):
            annotations = tu.annotations
            if annotations is None:
                return False
            similarity = annotations.get("similarity")
            if similarity is None:
                return False
            v = float(similarity)
            norm_v = ((v - threshold) * factor + 1) / 2
            if mode == "hard":
                p = 0.5
            else:
                p = random.random()
                if mode == "soft_sigmoid":
                    norm_v = 1 / (1 + math.exp(-norm_v))
            if p > norm_v:
                if self._verbose:
                    return (True, f"Similarity score {norm_v} lower than {p}")
                return True
            return False

        super().__init__([_filter])
