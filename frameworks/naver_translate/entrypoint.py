import os
import requests

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


naver_lang_dict_map = {
  'zh':'zh-CN',
  'en':'en',
  'fr':'fr',
  'id':'id',
  'ko':'ko',
  'es':'es',
  'th':'th',
  'zt':'zh-TW',
  'vi':'vi'
}


class NaverTranslateFramework(CloudTranslationFramework):

    def __init__(self):
        super(NaverTranslateFramework, self).__init__()
        self._appid = os.getenv('NAVER_CLIENT_ID')
        self._key = os.getenv('NAVER_SECRET')
        if self._appid is None:
            raise ValueError("missing app id")
        if self._key is None:
            raise ValueError("missing key")

    def translate_batch(self, batch, source_lang, target_lang):
        url = 'https://naveropenapi.apigw.ntruss.com/nmt/v1/translation'
        data = {
            'source': naver_lang_dict_map[source_lang.lower()],
            'target': naver_lang_dict_map[target_lang.lower()],
            'text': '\n'.join(batch)
        }
        headers = {
            'X-NCP-APIGW-API-KEY-ID': self._appid,
            'X-NCP-APIGW-API-KEY': self._key
        }

        result = self.send_request(lambda: requests.post(url, data=data, headers=headers))
        yield result['message']['result']['translatedText']

    def supported_languages(self):
        return ['zh', 'en', 'fr', 'id', 'ko', 'es', 'th', 'zt', 'vi']


if __name__ == "__main__":
    NaverTranslateFramework().run()
