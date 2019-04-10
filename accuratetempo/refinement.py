"""
Refinement of BPM annotations/estimates using RLAC.
"""

import csv
import os
import sys
from math import floor, ceil
from os import walk
from os.path import join, basename

import audioread
import jams
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


def rlac(audio_file, original_bpm, hop_length=1, window_length=256, tau=0.04, print_images=False):
    """
    Attempt to refined the original tempo (in BPM) for the given audio file.
    
    :param audio_file: audio file
    :param original_bpm: original BPM value
    :param hop_length: hop length for the RMSE energy function
    :param window_length: window length for the RMSE energy function
    :param tau: tolerance around the original BPM to use for the restricted lag interval(s)
    :param print_images: print images of RLAC curve?
    :return: refined estimate in BPM
    """

    if original_bpm == 0.:
        return 0.

    # downsample
    y, sr = librosa.load(audio_file, sr=44100.)
    sr /= 4
    y = y[::4] # this is cheating, but MUCH faster

    energy = librosa.feature.rmse(y=y, frame_length=window_length, hop_length=hop_length)[0]
    energy_sr = sr / hop_length

    # min-max norm
    norm_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    hl = int(4 * energy_sr)  # 4 sec
    wl = int(10 * energy_sr)

    bpms = []
    for pos in range(0, norm_energy.shape[0]-wl, hl):
        norm_energy_window = norm_energy[pos:pos+wl]

        # now find peak in given bpm range
        lag = bpm_to_lag(original_bpm, sr=energy_sr)
        lag_shoulder = ceil(lag - bpm_to_lag(original_bpm * (1. + tau), sr=energy_sr))

        lo_lag_1 = ceil(lag + lag_shoulder)
        hi_lag_1 = floor(lag - lag_shoulder)

        lag_shoulder_6 = ceil(lag * 6 * tau)
        lo_lag_6 = ceil(lag*6 + lag_shoulder_6)
        hi_lag_6 = floor(lag*6 - lag_shoulder_6)
        ac_6 = restricted_lag_autocorrelation(norm_energy_window, hi_lag_6, lo_lag_6)
        peak_lag_6 = np.argmax(ac_6)

        lag_shoulder_8 = ceil(lag * 8 * tau)
        lo_lag_8 = ceil(lag*8 + lag_shoulder_8)
        hi_lag_8 = floor(lag*8 - lag_shoulder_8)
        ac_8 = restricted_lag_autocorrelation(norm_energy_window, hi_lag_8, lo_lag_8)
        peak_lag_8 = np.argmax(ac_8)

        if print_images:
            _print_image(audio_file, pos, norm_energy_window, hi_lag_1, lo_lag_1, hi_lag_6, lo_lag_6, hi_lag_8, lo_lag_8)

        if ac_6[peak_lag_6] > ac_8[peak_lag_8]:
            peak_lag = (peak_lag_6 + hi_lag_6) / 6.
        else:
            peak_lag = (peak_lag_8 + hi_lag_8) / 8.

        bpm = lag_to_bpm(peak_lag, sr=energy_sr)
        bpms.append(bpm)

    return np.median(bpms)


def _print_image(file, n, onset_env, hi_lag_1, lo_lag_1, hi_lag_6, lo_lag_6, hi_lag_8, lo_lag_8):
    """
    Print PDF of RLAC.
    """
    ac = librosa.autocorrelate(onset_env)

    for i in range(ac.shape[0]):
        ac[i] /= ac.shape[0]-i

    plt.rc('font', family='Times New Roman')

    ax = plt.gca()
    plt.gcf().set_size_inches(4, 2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x)//1000)))
    plt.plot(ac)
    plt.ylim(bottom=0.04)
    plt.xlim(left=0, right=len(ac))

    ac_8 = ac.copy()
    ac_8[:hi_lag_8] = 0
    ac_8[lo_lag_8:] = 0
    x_8 = hi_lag_8 + (lo_lag_8 - hi_lag_8) // 2
    plt.axvline(x=x_8, color='black', linestyle='dashed')
    plt.plot(ac_8, label='8 IBIs±4%')

    ac_6 = ac.copy()
    ac_6[:hi_lag_6] = 0
    ac_6[lo_lag_6:] = 0
    x_6 = hi_lag_6 + (lo_lag_6 - hi_lag_6) // 2
    plt.axvline(x=x_6, color='black', linestyle='dashed')
    plt.plot(ac_6, label='6 IBIs±4%')

    ac_1 = ac.copy()
    ac_1[:hi_lag_1] = 0
    ac_1[lo_lag_1:] = 0
    x_1 = hi_lag_1 + (lo_lag_1 - hi_lag_1) // 2
    plt.axvline(x=x_1, color='black', linestyle='dashed')
    plt.plot(ac_1, label='1 IBI±4%')

    plt.ylabel('Autocorrelation')
    plt.xlabel('Lag in 1,000 Frames')
    plt.legend(loc='lower right')
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/' + basename(file).replace('.wav', '_{}.pdf'.format(n)), bbox_inches='tight')
    plt.close()


def restricted_lag_autocorrelation(y, hi_lag, lo_lag):
    """
    Restricted lag autocorrelation.
    Naive implementation, not relying on FFT.

    :param y: signal
    :param hi_lag: upper boundary (in BPM, i.e. this is LESS THAN lo_lag)
    :param lo_lag: lower boundary (in BPM, i.e. this is GREATER THAN hi_lag)
    :return: lags, starting with hi_lag, i.e. not 0-based.
    """
    lags = np.zeros(lo_lag - hi_lag)
    for l in range(hi_lag, lo_lag):
        mult = y[:-l] * y[l:]
        lags[l-hi_lag] = np.mean(mult)  # mean instead of sum to normalize
    return lags


def lag_to_bpm(lag, sr=44100.):
    """
    Convert lag (in frames) to BPM.

    :param lag: lag
    :param sr: sampling rate (of autocorrelation signal)
    :return: BPM
    """
    beats_per_second = sr / lag
    beats_per_minute = beats_per_second * 60.
    return beats_per_minute


def bpm_to_lag(bpm, sr=44100.):
    """
    Convert BPM to lag (in frames).

    :param bpm: BPM
    :param sr: sampling rate (of autocorrelation signal)
    :return: lag (in frames)
    """
    beats_per_second = bpm / 60.
    lag = sr / beats_per_second
    return lag


def refine_jams(audio_dirs, jams_dirs, tau=0.04, hop=1, rmse=256, update_jam=False):
    """
    Refine estimates/annotations from jams.

    :param audio_dirs: list of directories, containing audio files
    :param jams_dirs: list of directories, containing jams files
    :param tau: tau
    :param hop: RMSE hop length
    :param rmse: RMSE window length
    :param update_jam: if ``True`` annotations are added to the existing jams files
    """
    print('Refining jams from {} based on audio files in {} using tau={}'.format(jams_dirs, audio_dirs, tau))
    audio_files = _scan_audio_files(audio_dirs)

    def create_tempo_annotation(tempo1=0.0, tempo2=0.0, confidence1=1.0, confidence2=0.0):
        tempo = jams.Annotation(namespace='tempo')
        tempo.append(time=0.0,
                     duration='nan',
                     value=tempo1,
                     confidence=confidence1)
        if tempo2 != 0.0:
            tempo.append(time=0.0,
                         duration='nan',
                         value=tempo2,
                         confidence=confidence2)
        return tempo

    for d in jams_dirs:
        for (dirpath, _, filenames) in walk(d):
            for file in [f for f in filenames if f.endswith('.jams')]:

                k = file.replace('.jams', '')
                jam_file = join(dirpath, file)
                print('Processing \'{}\' ...'.format(jam_file))

                if k in audio_files:
                    jam = jams.load(jam_file, validate=False)
                    if update_jam:
                        new_jam = jam
                    else:
                        new_jam = jams.JAMS()
                    modified = False
                    for annotation in jam.annotations['tempo']:

                        # always correct duration
                        new_jam.file_metadata.duration = audio_files[k][1]
                        if not update_jam:
                            new_jam.file_metadata.identifiers = {'file': basename(audio_files[k][0])}

                        t1 = annotation.data[0].value
                        c1 = annotation.data[0].confidence
                        new_t1 = rlac(audio_files[k][0], t1, hop, rmse, tau=tau)

                        if len(annotation.data) > 1:
                            t2 = annotation.data[1].value
                            c2 = annotation.data[1].confidence
                            new_t2 = rlac(audio_files[k][0], t2, hop, rmse, tau=tau)
                        else:
                            c2 = 0.0
                            new_t2 = 0.0

                        # create new annotations
                        new_annotation = create_tempo_annotation(tempo1=new_t1, confidence1=c1,
                                                                 tempo2=new_t2, confidence2=c2)
                        data_source = ''
                        metadata = annotation.annotation_metadata
                        if metadata.data_source is not None\
                                and len(metadata.data_source) > 0:
                            data_source = metadata.data_source + ', '
                        data_source = data_source + 'postprocessed with RLAC (tau={}, hop={}, rmse={})' \
                            .format(tau, hop, rmse)

                        new_annotation.annotation_metadata = jams.AnnotationMetadata(
                            corpus=metadata.corpus,
                            version=metadata.version,
                            curator=metadata.curator,
                            data_source=data_source,
                            annotator=metadata.annotator)

                        new_jam.annotations.append(new_annotation)
                        modified = True

                    if modified:
                        if update_jam:
                            jam.save(jam_file)
                        else:
                            if jam.file_metadata.artist is not None:
                                new_jam.file_metadata.artist = jam.file_metadata.artist
                            if jam.file_metadata.title is not None:
                                new_jam.file_metadata.title = jam.file_metadata.title
                            if jam.file_metadata.release is not None:
                                new_jam.file_metadata.release = jam.file_metadata.release
                            new_jam.save(jam_file.replace('.jams', '_refined.jams'))
                else:
                    print('Failed to find audio file for \'{}\''.format(jam_file), file=sys.stderr)


def _scan_audio_files(audio_dirs):
    """
    Create a mapping between ids (usually a filename without extension) and full
    filenames (incl. path).
    Recursively walk directories to find audio files.

    :param audio_dirs: audio dirs
    :return: dict of ids and a tuple consisting of full filenames and duration
    """

    def _create_id_file_dict(audio_dir):
        """
        Create a mapping between ids (usually a filename without extension) and full
        filenames (incl. path).

        :param audio_dir: directory containing audio files.
        :return: dict of ids and a tuple consisting of full filenames and duration
        """
        files = {}
        for (dirpath, _, filenames) in walk(audio_dir):
            for file in [f for f in filenames if f.endswith('.wav') or f.endswith('.mp3')]:
                k = file.replace('.wav', '').replace('.mp3', '').replace('.LOFI', '')
                full_file = join(dirpath, file)
                with audioread.audio_open(full_file) as audio_file:
                    files[k] = (full_file, audio_file.duration)

        return files

    print('Scanning audio files...')
    audio_files = {}
    for audio_dir in audio_dirs:
        # create id -> filename label
        audio_files.update(_create_id_file_dict(audio_dir))
    print('Found {} audio files.'.format(len(audio_files)))
    return audio_files


def refine_ground_truth(audio_dirs, ground_truth_files, tau=0.04, hop=1, rmse=256):
    """
    Attempts to refine a ground truth based on restricted lag autocorrelation (RLAC).

    :param audio_dirs: base directories where audio files can be found
    :param ground_truth_files: list of ground truth .tsv files
    :param tau: specifies search window around certain time lags (6 IBI, 8 IBI).
    :param hop: RMSE hop length
    :param rmse: RMSE window length
    """
    print('Refining ground truths {} based on audio files in {} using tau={}'.format(ground_truth_files, audio_dirs, tau))
    audio_files = _scan_audio_files(audio_dirs)

    for ground_truth_file in ground_truth_files:
        ground_truth = _read_ground_truth_file(ground_truth_file)
        corrected_ground_truth = {}
        for k,v in ground_truth.items():
            print('Processing \'{}\' ...'.format(audio_files[k][0]))
            old_bpm = v
            new_bpm = rlac(audio_files[k][0], old_bpm, hop, rmse, tau=tau, print_images=False)
            bpm = new_bpm
            corrected_ground_truth[k] = bpm
        refined_file_name = ground_truth_file.replace('.tsv', '_refined.tsv')
        _write_ground_truth_file(corrected_ground_truth, refined_file_name)
        print('Corrected {} bpm values in {}. Stored in {}'.format(len(ground_truth), ground_truth_file, refined_file_name))


def _read_ground_truth_file(file):
    """
    Read ground truth from .tsv file containing ids and BPM values.

    :param file: tsv file
    :return: dict of ids and BPM labels
    """
    labels = {}
    with open(file, mode='r', encoding='utf-8') as text_file:
        reader = csv.reader(text_file, delimiter='\t')
        for row in reader:
            id = row[0]
            bpm = float(row[1])
            labels[id] = bpm
    return labels


def _write_ground_truth_file(ground_truth, file):
    """
    Writes ground truth as .tsv file.

    :param ground_truth: dict of ids and BPM labels
    :param file: file to write to
    """
    with open(file, mode='w', encoding='utf-8') as text_file:
        writer = csv.writer(text_file, delimiter='\t')
        for k,v in ground_truth.items():
            writer.writerow([k, v])
