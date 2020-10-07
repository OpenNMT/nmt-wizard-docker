import os
import systran_align

from nmtwizard.preprocess import prepoperator

@prepoperator.register_operator("alignment")
class Aligner(prepoperator.Operator):

    @staticmethod
    def is_applied_for(process_type):
        return process_type == prepoperator.ProcessType.TRAINING

    def __init__(self, align_config, process_type, build_state):
        self._align_config = align_config
        self._aligner = None
        self._write_alignment = self._align_config.get('write_alignment', False)

    def _preprocess(self, tu_batch):
        tu_list, meta_batch = tu_batch
        if self.process_type == prepoperator.ProcessType.TRAINING:
            meta_batch['write_alignment'] = self._write_alignment
        self._build_aligner()
        tu_list = self._set_aligner(tu_list)
        return tu_list, meta_batch


    def _build_aligner(self):
        if not self._aligner and self._align_config:
            # TODO : maybe add monotonic alignment ?
            forward_probs_path=self._align_config.get('forward', {}).get('probs')
            backward_probs_path=self._align_config.get('backward', {}).get('probs')
            if forward_probs_path and backward_probs_path:
                if not os.path.exists(forward_probs_path) or not os.path.isfile(forward_probs_path):
                    raise ValueError("Forward probs file for alignment doesn't exist: %s" % forward_probs_path)
                if not os.path.exists(backward_probs_path) or not os.path.isfile(backward_probs_path):
                    raise ValueError("Backward probs file for alignment doesn't exist: %s" % backward_probs_path)
                self._aligner = systran_align.Aligner(forward_probs_path, backward_probs_path)
            else:
                self._aligner = None

    def _set_aligner(self, tu_list):
        # Set aligner for TUs.
        for tu in tu_list :
            tu.set_aligner(self._aligner)
        return tu_list
