import requests
import os
import json
import time

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

        retry = 0
        while retry < 10:
            r = requests.get("https://api.deepl.com/v2/translate", params=params)
            if r.status_code == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        if r.status_code != 200 or 'translations' not in r.json():
            raise RuntimeError('incorrect result from \'translate\' service: %s' % r.text)
        for trans in r.json()['translations']:
            yield trans['text']

    def supported_languages(self):
        return ['en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'pl', 'ru']


if __name__ == "__main__":
    DeepLTranslateFramework().run()
