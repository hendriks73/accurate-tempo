"""
Encapsulates ground truth.
"""

import csv
from os.path import basename

import numpy as np

BPM_LO = 30.
BPM_HIGH = 285.
BPM_VALUES=BPM_HIGH - BPM_LO + 1.
LOG2_RANGE = np.log2(BPM_HIGH / BPM_LO)


def linear_to_label(index, nb_classes):
    return BPM_VALUES * index / nb_classes + BPM_LO


def linear_to_index(label, nb_classes):
    return round((label - BPM_LO) * nb_classes / BPM_VALUES)


def log_to_label(index, nb_classes):
    factor = (nb_classes-1) / LOG2_RANGE
    return BPM_LO * np.power(2., index/factor)


def log_to_index(label, nb_classes):
    if label == 0:
        return 0.
    factor = (nb_classes-1) / LOG2_RANGE
    return round(factor * np.log2(label/BPM_LO))


class GroundTruth:
    """
    Tempo ground truth.
    """

    def __init__(self, file, nb_classes=256, log_scale=False) -> None:
        """
        Creates a ground truth based on the given ``.tsv`` file. The file must contain the columns
        id, BPM.

        :param file: tsv file
        :param nb_classes: number of classes
        :param log_scale: use log scale (otherwise linear scale)
        """
        super().__init__()
        self.file = file
        self.nb_classes = nb_classes
        self.labels = {}
        self.labels.update(self._read_label_file(file))
        self.name = basename(file).replace('.tsv', '')
        self.log_scale = log_scale
        if log_scale:
            self.to_index = log_to_index
            self.to_label = log_to_label
        else:
            self.to_index = linear_to_index
            self.to_label = linear_to_label

    def __len__(self):
        return len(self.labels)

    def classes(self):
        return [i for i in range(self.nb_classes)]

    def get_label(self, index):
        if index < 0 or index >= self.nb_classes:
            return None
        return self.to_label(index, self.nb_classes)

    def get_index(self, label):
        index = self.to_index(label, self.nb_classes)
        # enforce valid range
        if index < 0:
            return 0
        elif index >= self.nb_classes:
            return self.nb_classes-1
        else:
            return index

    def get_index_for_key(self, key, scale_factor=1.):
        if scale_factor is None:
            scale_factor = 1.
        label = self.labels[key]
        if label is None:
            return None
        else:
            return self.get_index(label*scale_factor)

    def _read_label_file(self, file):
        labels = {}
        with open(file, mode='r', encoding='utf-8') as text_file:
            reader = csv.reader(text_file, delimiter='\t')
            for row in reader:
                id = row[0]
                bpm = float(row[1])
                labels[id] = bpm
        return labels
