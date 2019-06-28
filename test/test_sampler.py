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
    assert summary['news_pattern.']['lines_sampled'] == 3000
    assert summary['generic_corpus.']['lines_sampled'] >= 215
    assert summary['generic_added.']['lines_sampled'] >= 107
    assert summary['corpus_specific1.']['lines_sampled'] >= 479
    assert summary['corpus_specific2.']['lines_sampled'] >= 1198
    assert summary['IT.']['lines_sampled'] == 0

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
