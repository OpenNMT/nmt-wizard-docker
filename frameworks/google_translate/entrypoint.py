import os
import time

from google.cloud import translate

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


class GoogleTranslateFramework(CloudTranslationFramework):

    def __init__(self):
        super(GoogleTranslateFramework, self).__init__()
        credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials is None:
            raise ValueError("missing credentials")
        if credentials.startswith('{'):
            credential_path = os.path.join(self._tmp_dir, 'Gateway-Translate-API.json')
            with open(credential_path, 'w') as f:
                f.write(credentials)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        self._client = translate.Client()
        self._char_counts = 0
        self._GOOGLE_LIMIT_SIZE = 100000
        self._GOOGLE_LIMIT_TIME = 100

    def get_char_counts(self, batch):
        counts = 0
        for line in batch:
            counts += len(line)
        return counts

    def translate_batch(self, batch, source_lang, target_lang):
        current_batch_char = self.get_char_counts(batch)
        if self._char_counts + current_batch_char > self._GOOGLE_LIMIT_SIZE:
            self._char_counts = current_batch_char
            print("Exceeding the Google API limit %d, sleep %d seconds ..." % (self._GOOGLE_LIMIT_SIZE, self._GOOGLE_LIMIT_TIME))
            time.sleep(self._GOOGLE_LIMIT_TIME)
            time.sleep(1)
        else:
            self._char_counts += current_batch_char

        translation = self._client.translate(
            batch,
            source_language=source_lang,
            target_language=target_lang,
            format_='text')
        for trans in translation:
            yield trans['translatedText']


if __name__ == "__main__":
    GoogleTranslateFramework().run()
