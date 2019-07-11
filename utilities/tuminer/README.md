# TU Miner Utility

## Build docker


```bash
(within nmt-wizard-docker top directory)
docker build -t nmtwizard/tuminer ${PWD} -f utilities/tuminer/Dockerfile
```


## Run docker

### Command line options ###

```bash
usage: entrypoint.py [-h] [-s STORAGE_CONFIG] [-t TASK_ID] [-i IMAGE]
                     [-b BEAT_URL] [-bi BEAT_INTERVAL]
                     [--statistics_url STATISTICS_URL] [-ms MODEL_STORAGE]
                     [-msr MODEL_STORAGE_READ] [-msw MODEL_STORAGE_WRITE]
                     [-c CONFIG] [-m MODEL] [-g GPUID] [--no_push] --tumode
                     {score,mine} --srclang SRCLANG --srcfile SRCFILE
                     --tgtlang TGTLANG --tgtfile TGTFILE --output OUTPUT
                     [--encoding ENCODING] [--verbose] [--threshold THRESHOLD]
                     [--bpecodes BPECODES] [--encoder ENCODER]
                     [--encoderdim ENCODERDIM]
                     [--encoderbuffersize ENCODERBUFFERSIZE]
                     [--encodermaxtokens ENCODERMAXTOKENS]

optional arguments:
  -h, --help            show this help message and exit
  ...
  -g GPUID, --gpuid GPUID
                        Comma-separated list of 1-indexed GPU identifiers (0
                        for CPU).
  ...
  --tumode {score,mine}   Tuminer mode
  --srclang SRCLANG     Source language (two-letter language code; ISO 639-1).
  --srcfile SRCFILE     Source language file.
  --tgtlang TGTLANG     Target language (two-letter language code; ISO 639-1).
  --tgtfile TGTFILE     Target language file.
  --output OUTPUT       Output file.
  --encoding ENCODING   Encoding of the input and output text files.
  --verbose             Increase output verbosity.
  --threshold THRESHOLD
                        When in `mine` mode, threshold value for mined TUs
  --bpecodes BPECODES   BPE code to be applied to both source and target
                        files. (default model provided in docker)
  --encoder ENCODER     Multi-lingual encoder to be used to encode both source
                        and target files. (default model provided in docker)
  --encoderdim ENCODERDIM
                        Encoder output dimension
  --encoderbuffersize ENCODERBUFFERSIZE
                        Encoder buffer size
  --encodermaxtokens ENCODERMAXTOKENS
                        Encoder max_token size
```

### Sample command line ###

```bash
nvidia-docker run -it \
  -v $PWD/test/corpus:/root/corpus:/corpus \
  -v /tmp/output:/output \
  nmtwizard/tuminer \
  --tumode score \
  --srclang he --srcfile /corpus/train/europarl-v7.de-en.10K.tok.de \
  --tgtlang en --tgtfile /corpus/train/europarl-v7.de-en.10K.tok.en \ 
  --output /output/europarl-v7.de-en.10K.tok.deen.tuminer-score
```

### Output format ###

#### Score mode ####

The output file will contain same number lines as the input files where each line will contain a real number.


#### Mine mode ####

In `mine` mode, the output file contains 0 or more lines of text where each line is formatted as below: 

```
(real number score)  \t  (source sentence)  \t  (target sentence)
```


#### How to interpret the score #### 
For both `score` and `mine` modes, a score is associated with a given sentence pair. 

This value typically ranges between 0 and 1.5.
However, a really bad pair of source and target sentences may produce a value below 0, and likewise a really good pair may have higher values than 1.5.

Values above 1.0 may indicate a really good translation unit pair.
If mining TUs from comparable corpora or when scoring translated texts, values above 0.7 or 0.8 may indicate that the pair is useful.



### Selecting the sentence encoder ###
The docker image contains two pre-trained multi-lingual encoders. 
The following encoder model and its associated BPE code is used by default:

```
/opt/LASER/models/bilstm.93langs.2018-12-26.pt
/opt/LASER/models/93langs.fcodes
```

If you wish to try the other model included in the docker image, you can add the following arguments to your command: 

```
--encoder /opt/LASER/models/bilstm.eparl21.2018-11-19.pt
--bpecodes /opt/LASER/models/eparl21.fcodes
```



### If there is an error ###

If your process is terminated without generating an output file (and there was no useful error message), 
it may be due to the process being killed when the sentence encoder was not able to allocate enough buffer space in memory.

The default values for the arguments `--encoderbuffersize`and `--encodermaxtokens` are 10000 and 12000, respectively,
and these values are known to work on a server-grade machine with 256G of RAM and NVIDIA GPUs with 12G of memory.

For a Macbook Pro laptop with 16G of RAM without any GPUs, scoring a 1000 pairs of sentences ran successfully by setting the argument `--encodermaxtokens` to 7500.





