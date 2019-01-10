import os
import six
import re
import time
import random

from nmtwizard.utility import Utility
from nmtwizard.logger import get_logger
from nmtwizard.utility import resolve_environment_variables
from nmtwizard.storage import StorageClient

from similarity import main

logger = get_logger(__name__)

class SimilarityUtility(Utility):

    def __init__(self):
        super(SimilarityUtility, self).__init__()

    @property
    def name(self):
        return "similarity"

    def exec_function(self, args):
        assert len(args) and (args[0] == "train" or args[0] == "apply"), "invalid parameters for similarity module"
        if args[0] == "train":
            assert "-trn" in args, "missing `-trn` parameters in training mode"
        else:
            assert "-tst" in args and "-mdir" in args, "missing `-tst` or `-mdir` parameters in inference mode"
            assert "-output" in args, "missing `-output` parameter in inference mode"
        new_args = []
        file_parameters = ['-trn', '-dev', '-src_tok', '-tgt_tok', '-src_emb', '-tgt_emb', '-tst', '-mdir']
        idx = 1
        output = None
        local_output = None
        while idx < len(args):
            val = args[idx]
            val = resolve_environment_variables(val)
            new_args.append(val)
            if val in file_parameters:
                assert idx+1 < len(args), "missing value for parameters `%s`" % val
                nextval = args[idx+1]
                inputs = nextval.split(',')
                local_inputs = []
                for input in inputs:
                    local_input = os.path.join(self._data_dir, self._storage.split(input)[-1])
                    print "--", input, local_input
                    if val == '-mdir':
                        self._storage.get_directory(input, local_input)
                    else:
                        self._storage.get_file(input, local_input)
                    local_inputs.append(local_input)
                new_args.append(','.join(local_inputs))
                idx += 1
            elif val == '-output':
                assert idx+1 < len(args), "missing value for parameters `%s`" % val
                output = args[idx+1]
                if output == '-':
                    new_args.append(output)
                else:
                    local_output = os.path.join(self._data_dir, self._storage.split(output)[-1])
                    new_args.append(local_output)
                idx += 1
            idx += 1
        print new_args
        main(['similarity.py'] + new_args)
        if local_output is not None:
            self._storage.push(local_output, output)


if __name__ == '__main__':
    SimilarityUtility().run()
