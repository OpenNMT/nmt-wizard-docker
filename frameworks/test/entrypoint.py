import os
import six
import re
import requests
import time

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import pick_free_port, TranslationOutput
from nmtwizard.utils import run_cmd

logger = get_logger(__name__)


class TestFramework(Framework):

    def __init__(self):
        super(TestFramework, self).__init__(support_multi_training_files=True)

    def train_multi_files(self,
                          config,
                          data_dir,
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

        time.sleep(config.get("duration", 10))

        model_file = os.path.join(self._output_dir,"model")
        with open(model_file, "w") as f:
            charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            length = 26
            random_bytes = os.urandom(length)
            len_charset = len(charset)
            indices = [int(len_charset * (ord(byte) / 256.0)) for byte in random_bytes]
            f.write("".join([charset[index] for index in indices]))

        return {"model": model_file}

    def train(self, *arg, **kwargs):
        raise NotImplementedError()

    def trans(self, config, model_path, input, output, gpuid=0):
        model_file = os.path.join(model_path, 'model')
        options = self._get_translation_options(
            config, model_file, input=input, output=output, gpuid=gpuid)

        with open(model_file) as f:
            map_alphabet = f.read()
        with open(input) as f_in, open(output, "w") as f_out:
            for l in f_in:
                for idx in range(len(l)):
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

    def _get_translation_options(self, config, model_file, input=None, output=None, gpuid=0):
        options = {}
        options.update(config["options"].get("common", {}))
        options.update(config["options"].get("trans", {}))
        return options

if __name__ == "__main__":
    TestFramework().run()
