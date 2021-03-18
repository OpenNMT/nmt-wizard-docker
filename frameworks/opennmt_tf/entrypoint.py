import os
import shutil

# Import logger before TensorFlow to register the global config and avoid duplicated logs.
from nmtwizard.logger import get_logger

import tensorflow as tf

import opennmt

from nmtwizard.framework import Framework
from nmtwizard import utils
from nmtwizard import serving

logger = get_logger(__name__)
tf.get_logger().setLevel(logger.level)

_V1_SAVED_MODEL_DIR = '1'
_SAVED_MODEL_DIR = 'saved_model'


class OpenNMTTFFramework(Framework):

    def train(self,
              config,
              src_file,
              tgt_file,
              src_vocab_info,
              tgt_vocab_info,
              align_file=None,
              example_weights_file=None,
              model_path=None,
              gpuid=0):
        if model_path is None or tf.train.latest_checkpoint(model_path) is None:
            prev_src_vocab = None
            prev_tgt_vocab = None
        else:
            prev_src_vocab = src_vocab_info.previous
            prev_tgt_vocab = tgt_vocab_info.previous

        runner = self._build_runner(
            config,
            src_vocab=prev_src_vocab or src_vocab_info.current,
            tgt_vocab=prev_tgt_vocab or tgt_vocab_info.current,
            src_file=src_file,
            tgt_file=tgt_file,
            align_file=align_file,
            example_weights_file=example_weights_file,
            model_path=model_path)

        if prev_src_vocab or prev_tgt_vocab:
            previous_model_dir = runner.model_dir
            runner.update_vocab(
                os.path.join(self._output_dir, 'new_vocab_checkpoint'),
                src_vocab=src_vocab_info.current if prev_src_vocab else None,
                tgt_vocab=tgt_vocab_info.current if prev_tgt_vocab else None)
            shutil.rmtree(previous_model_dir)

        output_dir, summary = runner.train(
            num_devices=utils.count_devices(gpuid),
            return_summary=True,
            fallback_to_cpu=False,
        )
        return _list_checkpoint_files(output_dir), summary

    def trans(self, config, model_path, input, output, gpuid=0):
        runner = self._build_runner(config, model_path=model_path)
        runner.infer(input, predictions_file=output)

    def release(self, config, model_path, optimization_level=None, gpuid=0):
        export_dir = os.path.join(self._output_dir, _SAVED_MODEL_DIR)
        runner = self._build_runner(config, model_path=model_path)
        runner.export(export_dir)
        return {os.path.basename(export_dir): export_dir}

    def serve(self, config, model_path, gpuid=0):
        v1_export_dir = os.path.join(model_path, _V1_SAVED_MODEL_DIR)
        if os.path.exists(v1_export_dir):
            raise ValueError('SavedModel exported with OpenNMT-tf 1.x are no longer supported. '
                             'They include ops from tf.contrib which is not included in '
                             'TensorFlow 2.x binaries. To upgrade automatically, you can release '
                             'or serve from a OpenNMT-tf 1.x training checkpoint.')
        export_dir = os.path.join(model_path, _SAVED_MODEL_DIR)
        translate_fn = tf.saved_model.load(export_dir).signatures['serving_default']
        return None, translate_fn

    def forward_request(self, model_info, inputs, outputs=None, options=None):
        translate_fn = model_info

        tokens, lengths = utils.pad_lists(inputs, padding_value='')
        outputs = translate_fn(
            tokens=tf.constant(tokens, dtype=tf.string),
            length=tf.constant(lengths, dtype=tf.int32))

        batch_predictions = outputs['tokens'].numpy()
        batch_lengths = outputs['length'].numpy()
        batch_log_probs = outputs['log_probs'].numpy()

        batch_outputs = []
        for predictions, lengths, log_probs in zip(
                batch_predictions, batch_lengths, batch_log_probs):
            outputs = []
            for prediction, length, log_prob in zip(predictions, lengths, log_probs):
                prediction = prediction[:length].tolist()
                prediction = [token.decode('utf-8') for token in prediction]
                score = float(log_prob)
                outputs.append(serving.TranslationOutput(prediction, score=score))
            batch_outputs.append(outputs)
        return batch_outputs

    def _map_vocab_entry(self, index, token, vocab):
        if index == 0:
            vocab.write('<blank>\n')
            vocab.write('<s>\n')
            vocab.write('</s>\n')
        vocab.write('%s\n' % token)

    def _build_runner(self,
                      config,
                      src_vocab=None,
                      tgt_vocab=None,
                      src_file=None,
                      tgt_file=None,
                      align_file=None,
                      example_weights_file=None,
                      model_path=None):
        model_dir = os.path.join(self._output_dir, 'model')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        # Copy checkpoint files into the temporary model dir.
        if model_path is not None:
            checkpoint_files = _list_checkpoint_files(model_path)
            for filename, path in checkpoint_files.items():
                shutil.copy(path, os.path.join(model_dir, filename))

        # Prepare vocabulary if not already done.
        if src_vocab is None:
            src_vocab = self._convert_vocab(config['vocabulary']['source']['path'])
        if tgt_vocab is None:
            tgt_vocab = self._convert_vocab(config['vocabulary']['target']['path'])

        options = config['options']
        run_config = _build_run_config(
            options.get('config'),
            model_dir,
            src_vocab,
            tgt_vocab,
            src_file=src_file,
            tgt_file=tgt_file,
            align_file=align_file,
            example_weights_file=example_weights_file)
        model = opennmt.load_model(
            model_dir,
            model_file=options.get('model'),
            model_name=options.get('model_type'),
        )
        return opennmt.Runner(
            model,
            run_config,
            auto_config=options.get('auto_config', False),
            mixed_precision=options.get('mixed_precision', False),
        )


def _build_run_config(config,
                      model_dir,
                      src_vocab,
                      tgt_vocab,
                      src_file=None,
                      tgt_file=None,
                      align_file=None,
                      example_weights_file=None):
    """Builds the final configuration for OpenNMT-tf."""
    config = opennmt.convert_to_v2_config(config) if config else {}
    config['model_dir'] = model_dir

    data = config.setdefault('data', {})
    data['source_vocabulary'] = src_vocab
    data['target_vocabulary'] = tgt_vocab
    if src_file is not None:
        data['train_features_file'] = src_file
    if tgt_file is not None:
        data['train_labels_file'] = tgt_file
    if align_file is not None and os.path.exists(align_file):
        data['train_alignments'] = align_file
        params = config.setdefault('params', {})
        params.setdefault('guided_alignment_type', 'ce')
    if example_weights_file is not None and os.path.exists(example_weights_file):
        data['example_weights'] = example_weights_file

    train = config.setdefault('train', {})
    train.setdefault('sample_buffer_size', -1)
    # No need to keep multiple checkpoints as only the last one will be pushed.
    train.setdefault('save_checkpoints_steps', None)
    if train.setdefault('average_last_checkpoints', 0) == 0:
        train['keep_checkpoint_max'] = 1
    if train.setdefault('max_step', None) is None:
        # Force a single pass if the number of training steps in unspecified.
        train['single_pass'] = True

    return config

def _list_checkpoint_files(model_dir):
    """Lists the checkpoint files that should be bundled in the model package."""
    latest = tf.train.latest_checkpoint(model_dir)
    if latest is None:
        return {}
    objects = {
        'checkpoint': os.path.join(model_dir, 'checkpoint'),  # Checkpoint state file.
    }
    for filename in os.listdir(model_dir):
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path) and path.startswith(latest):
            objects[filename] = path
    return objects


if __name__ == '__main__':
    OpenNMTTFFramework().run()
