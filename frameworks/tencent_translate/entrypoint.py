from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger
from nmtwizard.serving import TranslationOutput
import os
import re
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

logger = get_logger(__name__)

supportedLangRe = re.compile(r"^(zh|en|fr|de|id|it|ja|ko|ms|pt|ru|es|th|tr|vi)$")
entrypoint = "tmt.na-siliconvalley.tencentcloudapi.com"
# if you are inside mainland China, pleas use this server
#entrypoint = "tmt.tencentcloudapi.com"

class TencentTranslateFramework(Framework):

    def __init__(self):
        super(TencentTranslateFramework, self).__init__(stateless=True)
        self._appid = os.getenv('TENCENT_SecretId')
        self._key = os.getenv('TENCENT_SecretKey')
        assert isinstance(self._appid, str), "missing app id"
        assert isinstance(self._key, str), "missing key"
        self.httpClient = httplib.HTTPSConnection(entrypoint)
        
    def __del__(self):
        if self.httpClient:
          self.httpClient.close()
          
    def trans(self, config, model_path, input, output, gpuid=0):
        assert supportedLangRe.match(config['source']), "unsupported language: %s" % config['source']
        assert supportedLangRe.match(config['target']), "unsupported language: %s" % config['target']
        with open(input, 'rb') as fi, open(output, 'wb') as fo:
            lines = fi.readlines()
            translations = translate_list(
                self.httpClient,
                self._appid,
                self._key,
                lines, source_language=config['source'], target_language=config['target'])
            for translation in translations:
                fo.write(translation.encode('utf-8') + '\n')

    def train(self, *args, **kwargs):
        raise NotImplementedError("This framework can only be used for translation")

    def release(self, *arg, **kwargs):
        raise NotImplementedError('This framework does not require a release step')

    def serve(self, config, model_path, gpuid=0):
        return None, {'source': config['source'], 'target': config['target']}

    def forward_request(self, batch_inputs, info, timeout=None):
        return [[TranslationOutput(translation)] for translation in translate_list(
            self.httpClient,
            self._appid,
            self._key,
            batch_inputs,
            source_language=info['source'],
            target_language=info['target'])]

    def _preprocess_input(self, state, input):
        return input

    def _postprocess_output(self, state, source, target):
        return target


def translate_list(httpClient, appid, secretKey, texts, source_language, target_language):
    
    i = 0
    while i < len(texts):
        # Tencent API does not support to translate multi lines in one request
        nexti = i + 1
        if nexti > len(texts):
            nexti = len(texts)
        logger.info('Translating range [%d:%d]', i, nexti)
        
        action = 'TextTranslate'
        region = 'na-siliconvalley'
        timestamp = int(time.time())
        version = '2018-03-21'
        pojectid = '0'
        SignatureMethod = 'HmacSHA256'
        
        query = ''.join(texts[i:nexti])
        fromLang = source_language.lower()
        toLang = target_language.lower()
        salt = random.randint(1, sys.maxsize)

        myurl_pre = 'Action='+action+'&Nonce='+str(salt)+'&ProjectId='+pojectid+'&Region='+region+'&SecretId='+appid+'&SignatureMethod='+SignatureMethod+'&Source='+fromLang+'&SourceText='
        myurl_suf = '&Target='+toLang+'&Timestamp='+str(timestamp)+'&Version='+version
        myurl_get = 'POST'+entrypoint+"/?"+myurl_pre+query+myurl_suf

        sign = sign_request(secretKey, myurl_get, SignatureMethod)
        myurl_request = myurl_pre+urllib.quote(query)+myurl_suf+'&Signature='+urllib.quote(sign)

        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'accept': "application/json"
            }  
            
        retry = 0
        while retry < 10:
            httpClient.request("POST", "/", myurl_request, headers)
            r = httpClient.getresponse()
            if r.status == 429:
                retry += 1
                time.sleep(5)
            else:
                break

        results = json.loads(r.read())
        if r.status != 200 or 'TargetText' not in results['Response']:
            raise RuntimeError('incorrect result from \'translate\' service: %s' % results)
        yield results['Response']['TargetText']
        i = nexti

def sign_request(secretKey, signStr, signMethod):
    if sys.version_info[0] > 2:
        signStr = bytes(signStr, 'utf-8')
        secretKey = bytes(secretKey, 'utf-8')

    digestmod = None
    if signMethod == 'HmacSHA256':
        digestmod = hashlib.sha256
    elif signMethod == 'HmacSHA1':
        digestmod = hashlib.sha1
    else:
        raise NotImplementedError("signMethod invalid", "signMethod only support (HmacSHA1, HmacSHA256)")

    hashed = hmac.new(secretKey, signStr, digestmod)
    base64 = binascii.b2a_base64(hashed.digest())[:-1]

    if sys.version_info[0] > 2:
        base64 = base64.decode()

    return base64
    
if __name__ == "__main__":
    TencentTranslateFramework().run()
