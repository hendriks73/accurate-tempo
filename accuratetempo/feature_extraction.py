"""
Feature extraction code.
Creates joblib feature files out of audio files.
"""
import argparse
from os import walk
from os.path import join

import librosa
import numpy as np
from sklearn.externals import joblib

from accuratetempo.groundtruth import GroundTruth


def extract_tempo_features(file, window_length=1024, hop_fraction=2):
    """
    Extract features from a single file.

    :param file: file
    :param window_length: STFT window length
    :param hop_fraction: fraction of window_length to use as hop length (e.g. 2 -> hop_length=window_length*1/2).
    :return: features (as np.float16)
    """
    y, sr = librosa.load(file, sr=11025)
    hop_length = window_length // hop_fraction
    data = librosa.feature.melspectrogram(y=y, sr=11025, n_fft=window_length, hop_length=hop_length,
                                          power=1, n_mels=40, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    # float16 should be more than enough and saves some disk space
    return data.astype(np.float16)


def extract_features_from_folder(base_folder, valid_keys, extractor):
    """
    Reads a folder and its sub-folders, parses all ``.mp3/.wav`` files and stores
    the result in a dictionary using the file names (minus the extension or LOFI) as keys.

    :param valid_keys: valid keys (ignore files that do not resolve to one of the valid keys)
    :param base_folder: folder with ``.mp3/.wav`` files
    :return: dictionary with file names as keys
    """
    feature_dataset = {}
    for (dirpath, _, filenames) in walk(base_folder):
        for file in [f for f in filenames if f.endswith('.mp3') or f.endswith('.wav')]:
            key = file.replace('.mp3', '').replace('.wav', '').replace('.LOFI', '')
            # if we have a ground truth, limit to ids listed in the ground truth
            if valid_keys is not None and key not in valid_keys:
                continue
            features = extractor(join(dirpath, file))
            feature_dataset[key] = features
    return feature_dataset


def convert_audio_folder_to_joblib(base_folders, valid_keys, output_file, extractor):
    """
    Extract features from all audio files in the given folder and its sub-folders,
    and store them under keys equivalent to the file names (minus extension),
    store the resulting dict in ``output_file`` and return the dict.

    :param extractor: function that extracts features from a given file
    :param valid_keys: valid keys
    :param base_folders: base folders for json files
    :param output_file: joblib file
    :return: dict of keys and features
    """
    dataset = {}
    for base_folder in base_folders:
        dataset.update(extract_features_from_folder(base_folder, valid_keys, extractor))
    joblib.dump(dataset, output_file)
    return dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''This script allows extracting features from mp3 or wav files by recursively
walking a directory tree, starting with a provided base audio folder.
The features are stored in simple dictionaries ('filename_w/o_extension': spectrogram),
which in turn are stored in .joblib files.''')

    parser.add_argument(
        '-a',
        '--audio',
        nargs='+',
        help='Folder(s) containing mp3 or wav audio files. Will be read recursively, file names are used as keys.',
        required=True
    )
    parser.add_argument(
        '-g',
        '--ground-truth',
        nargs='*',
        help='Files with ground truth .tsv files. If set, only files also occurring in the truth will be read.',
        required=False
    )
    args = parser.parse_args()
    return args


def main():
    arguments = parse_arguments()

    audio_folders = arguments.audio
    if arguments.ground_truth:
        valid_keys = []
        for f in arguments.ground_truth:
            ground_truth = GroundTruth(f)
            valid_keys.extend(ground_truth.labels.keys())
        valid_keys = set(valid_keys)
    else:
        valid_keys = None

    window_length=1024

    for hop_fraction in [2, 4]:
        print('Extracting for hop_fraction {}'.format(hop_fraction))

        def tempo_extractor(file):
            return extract_tempo_features(file, window_length=window_length, hop_fraction=hop_fraction)

        convert_audio_folder_to_joblib(audio_folders,
                                       valid_keys,
                                       'tempo_features_hop{}.joblib'.format(window_length // hop_fraction),
                                       tempo_extractor)


if __name__ == '__main__':
    main()
