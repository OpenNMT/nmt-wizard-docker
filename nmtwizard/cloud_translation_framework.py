import abc
import time
import six
import io

from nmtwizard.framework import Framework
from nmtwizard.serving import TranslationOutput


def _batch_iter(iterable, size):
    batch = []
    for x in iterable:
        batch.append(x.strip())
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@six.add_metaclass(abc.ABCMeta)
class CloudTranslationFramework(Framework):

    def __init__(self):
        super(CloudTranslationFramework, self).__init__(stateless=True)

    def supported_languages(self):
        return None

    @abc.abstractmethod
    def translate_batch(self, batch, source_lang, target_lang):
        raise NotImplementedError()

    def _check_lang(self, lang):
        supported_languages = self.supported_languages()
        if supported_languages is not None and lang not in supported_languages:
            raise ValueError('unsupported language: %s' % lang)

    def send_request(self, request_fn, max_retry=5, retry_delay=5):
        retry = 0
        while retry < max_retry:
            r = request_fn()
            if r.status_code == 429:
                retry += 1
                time.sleep(5)
            else:
                break
        if r.status_code != 200:
            raise RuntimeError('Error status %d: %s' % (r.status_code, r.text))
        return r.json()

    def trans(self, config, model_path, input, output, gpuid=0):
        self._check_lang(config['source'])
        self._check_lang(config['target'])
        with io.open(input, mode='r', encoding='utf-8') as input_file, \
             io.open(output, mode='w', encoding='utf-8') as output_file:
            for batch in _batch_iter(input_file, 10):
                translations = self.translate_batch(
                    batch, config['source'], config['target'])
                for translation in translations:
                    output_file.write(translation)
                    output_file.write(u'\n')

    def train(self, *args, **kwargs):
        raise NotImplementedError('This framework can only be used for translation')

    def release(self, *arg, **kwargs):
        raise NotImplementedError('This framework does not require a release step')

    def serve(self, config, model_path, gpuid=0):
        self._check_lang(config['source'])
        self._check_lang(config['target'])
        return None, {'source': config['source'], 'target': config['target']}

    def forward_request(self, model_info, inputs, outputs=None, options=None):
        return [[TranslationOutput(translation)] for translation in self.translate_batch(
            inputs, model_info['source'], model_info['target'])]

    def _preprocess_input(self, state, input):
        return input

    def _postprocess_output(self, state, output):
        return output
