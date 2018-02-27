from google.cloud import translate

from nmtwizard.framework import Framework
from nmtwizard.logger import get_logger

logger = get_logger(__name__)


class GoogleTranslateFramework(Framework):

    def __init__(self):
        super(GoogleTranslateFramework, self).__init__(stateless=True)

    def trans(self, config, model_path, input, output, gpuid=0):
        translate_client = translate.Client()
        with open(input) as fi, open(output, "w") as fo:
            lines = fi.readlines()
            i = 0
            while i < len(lines):
                nexti = i + 50
                if nexti > len(lines):
                    nexti = len(lines)
                logger.debug('Translating range [%d:%d]', i, nexti)
                translation = translate_client.translate(
                    lines[i:nexti],
                    source_language=config['source'],
                    target_language=config['target'],
                    format_='text')
                for trans in translation:
                    fo.write(trans['translatedText'].encode('utf-8'))
                i = nexti

    def train(self, *args, **kwargs):
        raise NotImplementedError("This framework can only be used for translation")


if __name__ == "__main__":
    GoogleTranslateFramework().run()
