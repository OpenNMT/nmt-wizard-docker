import os
import copy
import shutil
import six

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard import utils
from nmtwizard import serving

logger = get_logger(__name__)

import opennmt as onmt
import tensorflow as tf

import grpc

from grpc.framework.interfaces.face.face import ExpirationError
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from opennmt.config import load_model
from opennmt.utils import checkpoint


class OpenNMTTFFramework(Framework):

    def __init__(self, *args, **kwargs):
        super(OpenNMTTFFramework, self).__init__(*args, **kwargs)
        tf.logging.set_verbosity(logger.level)

    def train(self,
              config,
              src_file,
              tgt_file,
              src_vocab_info,
              tgt_vocab_info,
              align_file=None,
              model_path=None,
              gpuid=0):
        if src_vocab_info['changed'] or tgt_vocab_info['changed']:
            model_path = checkpoint.update_vocab(
                model_path,
                os.path.join(self._output_dir, 'new_vocab_checkpoint'),
                src_vocab_info['model'],
                tgt_vocab_info['model'],
                new_src_vocab=src_vocab_info['current'] if src_vocab_info['changed'] else None,
                new_tgt_vocab=tgt_vocab_info['current'] if tgt_vocab_info['changed'] else None,
                mode='replace',
                session_config=tf.ConfigProto(
                    device_count={'GPU': 0}))
        model_dir, model = self._load_model(
            model_type=config['options'].get('model_type'),
            model_file=config['options'].get('model'),
            model_path=model_path)
        run_config = copy.deepcopy(config['options'].get('config', {}))
        run_config['model_dir'] = model_dir
        if 'data' not in run_config:
            run_config['data'] = {}
        if 'train' not in run_config:
            run_config['train'] = {}
        run_config['data']['source_words_vocabulary'] = src_vocab_info['current']
        run_config['data']['target_words_vocabulary'] = tgt_vocab_info['current']
        run_config['data']['train_features_file'] = src_file
        run_config['data']['train_labels_file'] = tgt_file
        if align_file is not None and os.path.exists(align_file) :
            run_config['data']['train_alignments'] = align_file
        if 'train_steps' not in run_config['train']:
            run_config['train']['single_pass'] = True
            run_config['train']['train_steps'] = None
        if 'sample_buffer_size' not in run_config['train']:
            run_config['train']['sample_buffer_size'] = -1
        runner = onmt.Runner(
            model,
            run_config,
            num_devices=utils.count_devices(gpuid),
            auto_config=config['options'].get('auto_config', False))
        runner.train()
        return self._list_model_files(model_dir)

    def trans(self, config, model_path, input, output, gpuid=0):
        runner = self._make_predict_runner(config, model_path)
        runner.infer(input, predictions_file=output)

    def release(self, config, model_path, gpuid=0):
        export_dir = self._export_model(config, model_path)
        return {'1': export_dir}  # TensorFlow Serving expects a version number (here we use 1).

    def serve(self, config, model_path, gpuid=0):
        # Start a new tensorflow_model_server instance.
        batching_parameters = self._generate_batching_parameters(config.get('serving'))
        port = serving.pick_free_port()
        model_name = '%s%s' % (config['source'], config['target'])
        cmd = ['tensorflow_model_server',
               '--port=%d' % port,
               '--model_name=%s' % model_name,
               '--model_base_path=%s' % model_path,
               '--enable_batching=true',
               '--batching_parameters_file=%s' % batching_parameters]
        process = utils.run_cmd(cmd, background=True)
        info = {'port': port, 'model_name': model_name}
        return process, info

    def forward_request(self, batch_inputs, info, timeout=None):
        channel = grpc.insecure_channel("localhost:%s" % info['port'])
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        max_length = max(len(src) for src in batch_inputs)
        tokens, lengths = utils.pad_lists(batch_inputs, padding_value='', max_length=max_length)
        batch_size = len(lengths)

        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = info['model_name']
        predict_request.inputs['tokens'].CopyFrom(
            tf.make_tensor_proto(tokens, shape=(batch_size, max_length)))
        predict_request.inputs['length'].CopyFrom(
            tf.make_tensor_proto(lengths, shape=(batch_size,)))

        try:
            future = stub.Predict.future(predict_request, timeout)
            result = future.result()
        except ExpirationError as e:
            logger.error('%s', e)
            return None

        lengths = tf.make_ndarray(result.outputs['length'])
        predictions = tf.make_ndarray(result.outputs['tokens'])
        log_probs = tf.make_ndarray(result.outputs['log_probs'])

        batch_outputs = []
        for hypotheses, length, log_prob in zip(predictions, lengths, log_probs):
            outputs = []
            for i, prediction in enumerate(hypotheses):
                prediction_length = length[i] - 1  # Ignore </s>.
                prediction = prediction[0:prediction_length].tolist()
                prediction = [tf.compat.as_text(pred) for pred in prediction]
                score = float(log_prob[i]) / prediction_length
                outputs.append(serving.TranslationOutput(prediction, score=score))
            batch_outputs.append(outputs)
        return batch_outputs

    def _register_vocab(self, config, data_config):
        data_config['source_words_vocabulary'] = self._convert_vocab(
            config['tokenization']['source']['vocabulary'])
        data_config['target_words_vocabulary'] = self._convert_vocab(
            config['tokenization']['target']['vocabulary'])
        return data_config

    def _map_vocab_entry(self, index, token, vocab):
        if index == 0:
            vocab.write(b"<blank>\n")
            vocab.write(b"<s>\n")
            vocab.write(b"</s>\n")
        vocab.write(b"%s\n" % token)

    def _generate_batching_parameters(self, serving_config):
        if serving_config is None:
            serving_config = {}
        parameters_path = os.path.join(self._output_dir, 'batching_parameters.txt')
        with open(parameters_path, 'wb') as parameters_file:
            parameters_file.write(b'max_batch_size { value: %d }\n' % (
                serving_config.get('max_batch_size', 100000)))  # Handled by the wrapper.
            parameters_file.write(b'batch_timeout_micros { value: %d }\n' % (
                serving_config.get('batch_timeout_micros', 0)))
            parameters_file.write(b'pad_variable_length_inputs: true\n')
        return parameters_path

    def _make_predict_runner(self, config, model_path):
        model_dir, model = self._load_model(
            model_type=config['options'].get('model_type'),
            model_file=config['options'].get('model'),
            model_path=model_path)
        run_config = copy.deepcopy(config['options']['config'])
        run_config['model_dir'] = model_dir
        if 'data' not in run_config:
            run_config['data'] = {}
        run_config['data'] = self._register_vocab(config, run_config['data'])
        return onmt.Runner(model, run_config)

    def _export_model(self, config, model_path):
        # Hide GPU when exporting the model.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        runner = self._make_predict_runner(config, model_path)
        export_dir = runner.export()
        if visible_devices is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        return export_dir

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
        # Only keep the latest checkpoint in the state proto.
        tf.train.update_checkpoint_state(model_dir, os.path.basename(latest))
        objects = {
            "checkpoint": os.path.join(model_dir, "checkpoint"),
            "model_description.py": os.path.join(model_dir, "model_description.py")
        }
        for filename in os.listdir(model_dir):
            path = os.path.join(model_dir, filename)
            if os.path.isfile(path) and path.startswith(latest):
                objects[filename] = path
        return objects


if __name__ == '__main__':
    OpenNMTTFFramework().run()
