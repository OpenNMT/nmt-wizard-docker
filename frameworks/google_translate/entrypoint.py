import os

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

    def translate_batch(self, batch, source_lang, target_lang):
        translation = self._client.translate(
            batch,
            source_language=source_lang,
            target_language=target_lang,
            format_='text')
        for trans in translation:
            yield trans['translatedText']


if __name__ == "__main__":
    GoogleTranslateFramework().run()
