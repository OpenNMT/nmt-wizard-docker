import os
import six
import re
import requests
import time
import random

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger

logger = get_logger(__name__)

class TestFramework(Framework):

    def __init__(self):
        super(TestFramework, self).__init__(support_multi_training_files=True)

    def train_multi_files(self,
                          config,
                          data_dir,
                          src_vocab_info,
                          tgt_vocab_info,
                          model_path=None,
                          num_samples=None,
                          samples_metadata=None,
                          gpuid=0):
        options = self._get_training_options(
            config,
            data_dir,
            model_path=model_path,
            num_samples=num_samples,
            samples_metadata=samples_metadata,
            gpuid=gpuid)

        duration = config['options'].get('duration', 10)
        while duration > 0:
            print("LOG MESSAGE BLOCK - remaining %d", duration)
            time.sleep(5)
            duration -= 5

        model_file = os.path.join(self._output_dir,"model")
        with open(model_file, "w") as f:
            cmapping = range(0,26)
            random.shuffle(cmapping)
            f.write("".join([chr(c+65) for c in cmapping]))

        return {"model": model_file}

    def train(self, *arg, **kwargs):
        raise NotImplementedError()

    def trans(self, config, model_path, input_file, output_file, gpuid=0):
        model_file = os.path.join(model_path, 'model')
        options = self._get_translation_options(
            config, model_file, input_file=input_file, output_file=output_file, gpuid=gpuid)

        with open(model_file) as f:
            map_alphabet = f.read()
        with open(input_file) as f_in, open(output_file, "w") as f_out:
            for l in f_in:
                for idx, c in enumerate(l):
                    c = l[idx]
                    if c >= 'A' and c <= 'Z':
                        c = map_alphabet[ord(c)-ord('A')]
                    elif c >= 'a' and c <= 'z':
                        c = chr(ord(map_alphabet[ord(c)-ord('a')])+ord('a')-ord('A'))
                    l = l[:idx] + c + l[idx+1:]
                f_out.write(l)

    def serve(self, *arg, **kwargs):
        raise NotImplementedError('serving is not supported yet for test framework')

    def forward_request(self, *arg, **kwargs):
        raise NotImplementedError()

    def release(self, config, model_path, gpuid=0):
        raise NotImplementedError('release is not supported yet for test framework')

    def _get_training_options(self,
                              config,
                              data_dir,
                              model_path=None,
                              num_samples=None,
                              samples_metadata=None,
                              gpuid=0):
        options = config["options"]
        options.update(config["options"].get("common", {}))
        options.update(config["options"].get("train", {}))
        return options

    def _get_translation_options(self, config, model_file, input_file=None, output_file=None, gpuid=0):
        options = {}
        options.update(config["options"].get("common", {}))
        options.update(config["options"].get("trans", {}))
        return options

if __name__ == "__main__":
    TestFramework().run()
