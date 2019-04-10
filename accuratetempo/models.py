"""
Utility for model loading.
"""
from tensorflow.python.keras.models import load_model
import logging


class ModelLoader:
    """
    Handle for model allows passing around a (trained) model, without having to keep it in memory.
    It's simply referred to by its file name.
    """

    def __init__(self, file, name, history=None) -> None:
        super().__init__()
        self.file = file
        self.name = name
        self.history = history

    def load(self):
        logging.debug('Loading model named \'{}\' from file {}.'.format(self.name, self.file))
        return load_model(self.file)
