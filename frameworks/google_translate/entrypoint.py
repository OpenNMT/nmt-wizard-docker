from google.cloud import translate

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import TranslationOutput
import os

logger = get_logger(__name__)


class GoogleTranslateFramework(Framework):

    def __init__(self):
        super(GoogleTranslateFramework, self).__init__(stateless=True)
        credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        assert isinstance(credentials, str), "missing credentials"
        if credentials.startswith('{'):
            with open('/root/Gateway-Translate-API.json', 'w') as f:
                f.write(credentials)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/root/Gateway-Translate-API.json'

    def trans(self, config, model_path, input, output, gpuid=0):
        with open(input, 'rb') as fi, open(output, 'wb') as fo:
            lines = fi.readlines()
            translations = translate_list(
                lines, source_language=config['source'], target_language=config['target'])
            for translation in translations:
                fo.write(translation.encode('utf-8'))

    def train(self, *args, **kwargs):
        raise NotImplementedError("This framework can only be used for translation")

    def release(self, *arg, **kwargs):
        raise NotImplementedError('This framework does not require a release step')

    def serve(self, config, model_path, gpuid=0):
        return None, {'source': config['source'], 'target': config['target']}

    def forward_request(self, batch_inputs, info, timeout=None):
        return [[TranslationOutput(translation)] for translation in translate_list(
            batch_inputs,
            source_language=info['source'],
            target_language=info['target'])]

    def _preprocess_input(self, state, input):
        return input

    def _postprocess_output(self, state, source, target):
        return target


def translate_list(texts, source_language=None, target_language=None):
    translate_client = translate.Client()
    i = 0
    while i < len(texts):
        nexti = i + 50
        if nexti > len(texts):
            nexti = len(texts)
        logger.debug('Translating range [%d:%d]', i, nexti)
        translation = translate_client.translate(
            texts[i:nexti],
            source_language=source_language,
            target_language=target_language,
            format_='text')
        for trans in translation:
            yield trans['translatedText']
        i = nexti


if __name__ == "__main__":
    GoogleTranslateFramework().run()
