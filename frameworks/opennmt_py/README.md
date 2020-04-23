# OpenNMT-py framework

This framework is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/).

[Preprocessing](https://opennmt.net/OpenNMT-py/options/preprocess.html), [training](https://opennmt.net/OpenNMT-py/options/train.html), and [translation](https://opennmt.net/OpenNMT-py/options/translate.html) options specific to OpenNMT-py can be configured in the `options` block of the configuration.

Example:

```json
{
    "options": {
        "config": {
            "preprocess": {
            },
            "train": {
                "batch_size": 64,
                "optim": "sgd",
                "dropout": 0.3,
                "learning_rate": 1.0,
                "src_word_vec_size": 512,
                "tgt_word_vec_size": 512,
                "encoder_type": "rnn",
                "decoder_type": "rnn",
                "layers": 2,
                "enc_layers": 2,
                "dec_layers": 2,
                "rnn_size": 512
            },
            "trans": {
                "replace_unk": true
            }
        }
    }
}
```
