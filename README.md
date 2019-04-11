[![CC BY 3.0](https://img.shields.io/badge/License-CC%20BY%203.0-blue.svg)](https://creativecommons.org/licenses/by/3.0/)

# Accurate Tempo Estimation

This repository accompanies the paper [High-Accuracy Musical Tempo Estimation using Convolutional
Neural Networks and Autocorrelation](https://underreview.org/) in order to improve reproducibility
of the reported results.

## Audio Files

Unfortunately, because of size limitations imposed by GitHub as well as copyright issues, this repository does not
contain all audio samples or extracted features. But you can download those and extract them yourself.

Download links: 

- [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) 
- [Extended Ballroom](http://anasynth.ircam.fr/home/media/ExtendedBallroom) 
- [GiantSteps MTG Tempo](https://github.com/GiantSteps/giantsteps-mtg-key-dataset)
- [Lakh MIDI Dataset (LMD) Previews](https://bit.ly/2Bl8D1J)

Should you use any of the datasets in your academic work, please cite the corresponding publications.  

## Annotations

All necessary ground truth annotations are in the [annotations](./annotations) folder. For easy parsing they are
formatted in a simple tab separated values (`.tsv`) format, with columns `id \t bpm \n`. The class
[GroundTruth](./accuratetempo/groundtruth.py) is capable of reading and interpreting these files.

**Note:** The file [ballroom.tsv](./annotations/ballroom.tsv) contains tempo annotations derived from the beat annotations
[published by Florian Krebs](https://github.com/CPJKU/BallroomAnnotations). Using the the measure
values to find corresponding beats, we computed the median *Inter Measure Interval* (IMI) and converted it to BPM.
In other words: These are not the original annotations.     

The files [train.tsv](./annotations/train.tsv) and [valid.tsv](./annotations/valid.tsv) contain a 90/10
split of annotations for MTG Tempo, Eball and LMD Tempo. The same is true for their `*_refined.tsv` variants, which
have been processed with [RLAC](#restricted-lag-autocorrelation).

## Installation

To spare yourself some pain, first create a new [conda](https://docs.conda.io/en/latest/miniconda.html)
environment, then install:

    git clone https:/github.com/xxx/accurate-tempo.git
    cd accurate-tempo
    conda env create -f environment.yml
    [conda env create -f environment_no_gpu.yml]
    source activate accuratetempo  
    python setup.py install

If you are on macOS, you will need to use the `no_gpu` variant. If you are on another OS, but don't have
a supported GPU, you might need to edit [setup.py](./setup.py) before installation to make sure you are
installing `tensorflow` and not `tensorflow-gpu`.

## Feature Extraction

Before training, you need to extract features from the audio files.
For extraction, you can use the code in [feature_extraction.py](./accuratetempo/feature_extraction.py)
or the command line script mentioned below.
The created `.joblib` files are simple dictionaries, containing strings as keys and mel spectrograms as values.

Assuming you have all audio files for the datasets mentioned above in the directories
`./lmd`, `./eball`, `./ballroom`, and `./mtg` you may run the extraction using the following command line script:

    accurate_tempo_extraction -a ./lmd ./eball ./ballroom ./mtg \
        -g annotations/ballroom.tsv annotations/train.tsv annotations/valid.tsv

**Hint:** This is not a fast process... 
    
The ground truth file is optional. If given, only files that also occur in the ground truth are added
to the created feature `.joblib` files.

## Network Training and Evaluation

The provided script will train the networks and then report accuracy scores (Acc0, Acc1, Acc2, Mape1, and Mape2)
for the test and validation datasets.

As a prerequisite, you need to create feature files as [described above](#feature-extraction). 

**Note:** Running this locally only makes sense on a computer with GPU and even then it will take very long.

Each execution of the training script will:

1. Train 7 models based on the same data ("runs")
2. Evaluate against the provided validation dataset
3. Evaluate against the provided test dataset ("ballroom")
4. Save all predictions as `.joblib` (with the models) and `.jams` files (in the job directory)
5. Store the resulting models in the model directory

**Hint:** For your convenience, this repository already contains [pre-trained models](./models/), so by pointing
the script below to `./models` as model dir, you can skip training.

To run the training/reporting, you can execute the script [training.py](./accuratetempo/training.py)
or the command line script mentioned below with the following arguments:

    --test annotations/ballroom.tsv \
    --train annotations/train.tsv \
    --valid annotations/valid.tsv \
    --features tempo_features_hop256.joblib \
    --job-dir ./job \
    --model-dir ./models \
    --log-scale \
    --augment \
    --refined \
    --classes 256 \
    --hop 256
    
After installation, you may run the training code using the following command line script:

    accurate_tempo_training [arguments]

For this to succeed you must have first extracted features (for the `--features` argument).

You can influence the setup of the training, by setting some flags:

- `--log-scale` if set, classes are mapped to tempo values logarithmically
- `--augment` if set, scale & crop augmentation is used 
- `--classes C` if set, `C` classes are used
- `--hop H` if set, the network expects spectrograms created with hop length `H`
- `--refined` if set, refined versions of `train.tsv` and `valid.tsv` are used, simply by
replacing `.tsv` with `_refined.tsv` before loading.

All output from the script will be sent both to the console and a log file in your job directory.

## Restricted Lag Autocorrelation

After installation, you will be able to refine existing annotations or estimates from the command line
using *Restricted Lag Autocorrelation* (RLAC).
All you need is the original audio files and your estimates either as [JAMS](https://github.com/marl/jams)
or as `.tsv` (`id \t BPM \n`) files.

To refine your estimates/annotations, execute:

    rlac -a AUDIO_DIR [AUDIO_DIR ...] -e ESTIMATES [ESTIMATES ...]
    
`AUDIO_DIR` should point to a directory containing mp3 or wav files, `ESTIMATES` should either
point to a `.tsv` file or a directory of `.jams` files.

The script either produces either `*_refined.tsv` files or `*_refined.jams` files. Instead of
creating new files, you can also choose to update existing `.jams` files with the flag `--update-jams`.

Note that the default settings are rather fine-grained, i.e., it takes very long to run the script.
If you like to sacrifice BPM resolution for speed, set `--hop` to a larger value like `16` (default is `1`).
Hop defines the hop length used for creating the energy/activation function used by RLAC.

## License

This repository is licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
For attribution, please cite:

> XX: High-Accuracy Musical Tempo Estimation using Convolutional Neural Networks and Autocorrelation,
> in Proceedings...,... 