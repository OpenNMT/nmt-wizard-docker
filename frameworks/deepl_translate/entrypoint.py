import requests
import os

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


class DeepLTranslateFramework(CloudTranslationFramework):

    def __init__(self):
        super(DeepLTranslateFramework, self).__init__()
        self._credentials = os.getenv('DEEPL_CREDENTIALS')
        if self._credentials is None:
            raise ValueError("missing credentials")

    def translate_batch(self, batch, source_lang, target_lang):
        params = {
            "text": batch,
            "source_lang": source_lang.upper(),
            "target_lang": target_lang.upper(),
            "split_sentences": 0,
            "auth_key": self._credentials
        }

        url = 'https://api.deepl.com/v2/translate'
        result = self.send_request(lambda: requests.get(url, params=params))
        for trans in result['translations']:
            yield trans['text']

    def supported_languages(self):
        return ['en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'pl', 'ru']


if __name__ == "__main__":
    DeepLTranslateFramework().run()
