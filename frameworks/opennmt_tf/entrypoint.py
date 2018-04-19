import os
import copy
import shutil
import six

import opennmt as onmt
import tensorflow as tf

from opennmt.config import load_model

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger

logger = get_logger(__name__)


class OpenNMTTFFramework(Framework):

    def train(self,
              config,
              src_file,
              tgt_file,
              model_path=None,
              gpuid=0):
        model_dir, model = self._load_model(
            model_type=config['options'].get('model_type'),
            model_file=config['options'].get('model'),
            model_path=model_path)
        run_config = copy.deepcopy(config['options']['config'])
        run_config['model_dir'] = model_dir
        for k, v in six.iteritems(run_config['data']):
            run_config['data'][k] = self._convert_vocab(v)
        run_config['data']['train_features_file'] = src_file
        run_config['data']['train_labels_file'] = tgt_file
        if 'train_steps' not in run_config['train']:
            run_config['train']['single_pass'] = True
            run_config['train']['train_steps'] = None
        if 'sample_buffer_size' not in run_config['train']:
            run_config['train']['sample_buffer_size'] = -1
        onmt.Runner(model, run_config).train()
        return self._list_model_files(model_dir)

    def trans(self, config, model_path, input, output, gpuid=0):
        model_dir, model = self._load_model(
            model_type=config['options'].get('model_type'),
            model_file=config['options'].get('model'),
            model_path=model_path)
        run_config = copy.deepcopy(config['options']['config'])
        run_config['model_dir'] = model_dir
        for k, v in six.iteritems(run_config['data']):
            run_config['data'][k] = self._convert_vocab(v)
        onmt.Runner(model, run_config).infer(input, predictions_file=output)

    def _convert_vocab(self, vocab_file):
        converted_vocab_file = os.path.join(self._data_dir, os.path.basename(vocab_file))
        with open(vocab_file, "rb") as vocab, open(converted_vocab_file, "wb") as converted_vocab:
            converted_vocab.write(b"<blank>\n")
            converted_vocab.write(b"<s>\n")
            converted_vocab.write(b"</s>\n")
            for line in vocab:
                converted_vocab.write(line)
        return converted_vocab_file

    def _load_model(self, model_type=None, model_file=None, model_path=None):
        """Returns the model directory and the model instances.

        If model_path is not None, the model files are copied in the current
        working directory ${WORKSPACE_DIR}/output/model/.
        """
        model_dir = os.path.join(self._output_dir, "model")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        if model_path is not None:
            for filename in os.listdir(model_path):
                path = os.path.join(model_path, filename)
                if os.path.isfile(path):
                    shutil.copy(path, model_dir)
        model = load_model(model_dir, model_file=model_file, model_name=model_type)
        return model_dir, model

    def _list_model_files(self, model_dir):
        """Lists the files that should be bundled in the model package."""
        latest = tf.train.latest_checkpoint(model_dir)
        objects = {
            "checkpoint": os.path.join(model_dir, "checkpoint"),
            "model_description.pkl": os.path.join(model_dir, "model_description.pkl")
        }
        for filename in os.listdir(model_dir):
            path = os.path.join(model_dir, filename)
            if os.path.isfile(path) and path.startswith(latest):
                objects[filename] = path
        return objects


if __name__ == '__main__':
    OpenNMTTFFramework().run()
