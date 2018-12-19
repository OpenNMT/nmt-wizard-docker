import abc
import six

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

    def trans(self, config, model_path, input, output, gpuid=0):
        self._check_lang(config['source'])
        self._check_lang(config['target'])
        with open(input, 'rb') as input_file, open(output, 'wb') as output_file:
            for batch in _batch_iter(input_file, 10):
                translations = self.translate_batch(
                    batch, config['source'], config['target'])
                for translation in translations:
                    output_file.write(translation.encode('utf-8') + '\n')

    def train(self, *args, **kwargs):
        raise NotImplementedError('This framework can only be used for translation')

    def release(self, *arg, **kwargs):
        raise NotImplementedError('This framework does not require a release step')

    def serve(self, config, model_path, gpuid=0):
        self._check_lang(config['source'])
        self._check_lang(config['target'])
        return None, {'source': config['source'], 'target': config['target']}

    def forward_request(self, batch_inputs, info, timeout=None):
        return [[TranslationOutput(translation)] for translation in self.translate_batch(
            batch_inputs, info['source'], info['target'])]

    def _preprocess_input(self, state, input):
        return input

    def _postprocess_output(self, state, output):
        return output
