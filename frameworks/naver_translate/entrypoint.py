import os
import json
import urllib2

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
        from_lang = naver_lang_dict_map[source_lang.lower()]
        to_lang = naver_lang_dict_map[target_lang.lower()]
        encText = urllib2.quote('\n'.join(batch))
        data = "source=%s&target=%s&text=%s" % (from_lang, to_lang, encText)

        request = urllib2.Request("https://naveropenapi.apigw.ntruss.com/nmt/v1/translation")
        request.add_header("X-NCP-APIGW-API-KEY-ID", self._appid)
        request.add_header("X-NCP-APIGW-API-KEY", self._key)

        retry = 0
        while retry < 10:
            r = urllib2.urlopen(request, data=data.encode("utf-8"))
            rescode = r.getcode()
            if rescode == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.load(r)
        if rescode != 200 or 'result' not in results['message']:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        yield results['message']['result']['translatedText']

    def supported_languages(self):
        return ['zh', 'en', 'fr', 'id', 'ko', 'es', 'th', 'zt', 'vi']


if __name__ == "__main__":
    NaverTranslateFramework().run()
