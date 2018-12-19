import os
import json
import time
import httplib
import hashlib
import hmac
import base64
import urllib
import random
import sys
import binascii

from nmtwizard.cloud_translation_framework import CloudTranslationFramework


entrypoint = "tmt.na-siliconvalley.tencentcloudapi.com"
# if you are inside mainland China, pleas use this server
#entrypoint = "tmt.tencentcloudapi.com"


class TencentTranslateFramework(CloudTranslationFramework):

    def __init__(self):
        super(TencentTranslateFramework, self).__init__()
        self._appid = os.getenv('TENCENT_SecretId')
        self._key = os.getenv('TENCENT_SecretKey')
        if self._appid is None:
            raise ValueError("missing app id")
        if self._key is None:
            raise ValueError("missing key")
        self.httpClient = httplib.HTTPSConnection(entrypoint)

    def __del__(self):
        if self.httpClient:
          self.httpClient.close()

    def translate_batch(self, batch, source_lang, target_lang):
        # Tencent API does not support translating multi lines in one request
        for line in batch:
            yield self._translate_line(line, source_lang, target_lang)

    def _translate_line(self, line, source_lang, target_lang):
        action = 'TextTranslate'
        region = 'na-siliconvalley'
        timestamp = int(time.time())
        version = '2018-03-21'
        project_id = '0'
        signature_method = 'HmacSHA256'

        query = line
        from_lang = source_lang.lower()
        to_lang = target_lang.lower()
        salt = random.randint(1, sys.maxsize)

        myurl_pre = 'Action='+action+'&Nonce='+str(salt)+'&ProjectId='+project_id+'&Region='+region+'&SecretId='+self._appid+'&SignatureMethod='+signature_method+'&Source='+from_lang+'&SourceText='
        myurl_suf = '&Target='+to_lang+'&Timestamp='+str(timestamp)+'&Version='+version
        myurl_get = 'POST'+entrypoint+"/?"+myurl_pre+query+myurl_suf

        sign = _sign_request(self._key, myurl_get, signature_method)
        myurl_request = myurl_pre+urllib.quote(query)+myurl_suf+'&Signature='+urllib.quote(sign)

        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'accept': "application/json"
        }

        retry = 0
        while retry < 10:
            self.httpClient.request("POST", "/", myurl_request, headers)
            r = self.httpClient.getresponse()
            if r.status == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.loads(r.read())
        if r.status != 200 or 'TargetText' not in results['Response']:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        return results['Response']['TargetText']

    def supported_languages(self):
        return [
            'de',
            'en',
            'es',
            'fr',
            'id',
            'it',
            'ja',
            'ko',
            'ms',
            'pt',
            'ru',
            'th',
            'tr',
            'vi',
            'zh'
        ]


def _sign_request(secretKey, signStr, signMethod):
    if sys.version_info[0] > 2:
        signStr = bytes(signStr, 'utf-8')
        secretKey = bytes(secretKey, 'utf-8')

    digestmod = None
    if signMethod == 'HmacSHA256':
        digestmod = hashlib.sha256
    elif signMethod == 'HmacSHA1':
        digestmod = hashlib.sha1
    else:
        raise NotImplementedError("signMethod invalid",
                                  "signMethod only support (HmacSHA1, HmacSHA256)")

    hashed = hmac.new(secretKey, signStr, digestmod)
    base64 = binascii.b2a_base64(hashed.digest())[:-1]

    if sys.version_info[0] > 2:
        base64 = base64.decode()

    return base64


if __name__ == "__main__":
    TencentTranslateFramework().run()
