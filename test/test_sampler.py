import pytest

from nmtwizard.preprocess import generate_preprocessed_data

def test_sampler(tmpdir):
    config = {
        "source": "en",
        "target": "de",
        "data": {
            "sample": 5000,
            "sample_dist": [
                {
                    "path": str(pytest.config.rootdir.join('corpus/train')),
                    "distribution": [
                        ["generic", 1],
                        ["specific", 5.2],
                        ["news.*pattern", "*1"],
                        [".*something", 1]
                    ]
                }
            ]
        }
    }

    data_path, train_dir, num_samples, summary, metadata = \
        generate_preprocessed_data(config, "", str(tmpdir))
    assert num_samples == 5000
    assert summary['news_pattern.']['linesampled'] == 3000
    assert summary['generic_corpus.']['linesampled'] == 215
    assert summary['generic_added.']['linesampled'] == 108
    assert summary['corpus_specific1.']['linesampled'] == 479
    assert summary['corpus_specific2.']['linesampled'] == 1198
    assert summary['IT.']['linesampled'] == 0

    # check unique sampling with undersampling
    with open(str(tmpdir.join("preprocess/corpus_specific2.en")), 'rb') as f :
        rf = f.readlines()
        rf_list = [l.strip() for l in rf]
        assert len(rf_list) == len(set(rf_list))

    # check unique sampling with oversampling
    with open(str(tmpdir.join("preprocess/generic_added.en")), 'rb') as f :
        rf_dict = {}
        for line in f :
            if line in rf_dict:
                rf_dict[line] += 1
            else:
                rf_dict[line] = 1
        for l, c in rf_dict.items() :
            assert 2 <= c <= 3


    config["data"]["sample_dist"] = [
        {
            "path": str(pytest.config.rootdir.join('corpus/train')),
            "distribution": [
                ["generic", 1],
                ["corpus", 1],
            ]
        }
    ]

    with pytest.raises(RuntimeError, match=r"matches more than one rule"):
        generate_preprocessed_data(config, "", str(tmpdir))

    # TODO :
    # multiple blocks : should pattern count be global ?
    # blocks with same pattern
