import os
import systran_align

from nmtwizard.preprocess import prepoperator

@prepoperator.register_operator("alignment")
class Aligner(prepoperator.Operator):

    @staticmethod
    def is_applied_for(process_type):
        return process_type == prepoperator.ProcessType.TRAINING

    # Alignment models can take several GB in memory so we need to share an Aligner
    # instance across workers.

    @staticmethod
    def get_shared_classes():
        return [systran_align.Aligner]

    @staticmethod
    def get_shared_builders(config, process_type):
        forward_probs_path = config.get('forward', {}).get('probs')
        backward_probs_path = config.get('backward', {}).get('probs')
        if forward_probs_path is None or backward_probs_path is None:
            return None
        if not os.path.isfile(forward_probs_path):
            raise ValueError("Forward probs file for alignment doesn't exist: %s" % (
                forward_probs_path))
        if not os.path.isfile(backward_probs_path):
            raise ValueError("Backward probs file for alignment doesn't exist: %s" % (
                backward_probs_path))
        return {"aligner": (systran_align.Aligner, (forward_probs_path, backward_probs_path))}

    def __init__(self, align_config, process_type, build_state, shared_state=None):
        self._aligner = shared_state['aligner'] if shared_state else None
        self._write_alignment = align_config.get('write_alignment', False)

    def _preprocess(self, tu_batch):
        tu_list, meta_batch = tu_batch
        if self.process_type == prepoperator.ProcessType.TRAINING:
            meta_batch['write_alignment'] = self._write_alignment
        for tu in tu_list :
            tu.set_alignment(self._aligner)
        return tu_list, meta_batch
