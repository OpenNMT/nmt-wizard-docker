import os
import time

from google.cloud import translate

from nmtwizard.cloud_translation_framework import CloudTranslationFramework
from nmtwizard.logger import get_logger

logger = get_logger(__name__)


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
        self.max_retry = 5
        self._GOOGLE_LIMIT_TIME = 100

    def translate_batch(self, batch, source_lang, target_lang):
        translation = None
        retry = 0
        while retry < self.max_retry:
            try:
                translation = self._client.translate(
                    batch,
                    source_language=source_lang,
                    target_language=target_lang,
                    format_='text')
            except Exception as e:
                if e.code == 403 and "User Rate Limit Exceeded" in e.message:
                    logger.warning("Exceeding the Google API limit, retrying in %d seconds ..." % self._GOOGLE_LIMIT_TIME)
                    time.sleep(self._GOOGLE_LIMIT_TIME)
                    retry += 1
                    continue
                else:
                    raise
            break

        for trans in translation:
            yield trans['translatedText']


if __name__ == "__main__":
    GoogleTranslateFramework().run()
