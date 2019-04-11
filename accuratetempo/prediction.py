"""
Predict labels with a model.
"""
import logging
import zipfile
from os import makedirs, walk
from os.path import exists, join, basename
from shutil import rmtree

import jams
import numpy as np
from sklearn.externals.joblib import dump, load
from tensorflow.python.keras import backend as K


def predict_from_models(features, input_shape, model_loaders, normalizer, ground_truth, job_dir='./job',
                        write_jams=True, overwrite=False):
    """
    Create predictions for a test or validation dataset (i.e., a ground truth) for
    multiple models (passed via model loaders).

    :param job_dir:
    :param overwrite: overwrite prediction
    :param features: dict with id :: features
    :param input_shape: network input shape
    :param log: log function
    :param model_loaders: dict with model names as keys and lists of model loaders as values (different runs)
    :param normalizer: normalization function
    :param ground_truth: ground truth to predict for
    :return: a dict for each model kind with lists of results for multiple runs
    """

    sorted_names = list(model_loaders.keys())
    sorted_names.sort()

    dataset_description = ground_truth.name
    logging.info('Predicting for dataset \'{}\'.'.format(dataset_description))

    results = {}
    for model_name in sorted_names:
        same_kind_models = model_loaders[model_name]
        same_kind_results = results[model_name] = []

        for run, model_loader in enumerate(same_kind_models):
            file = prediction_file(model_loader, dataset_description)
            if overwrite or not exists(file):
                model = model_loader.load()
                if write_jams:
                    jams_dir = join(job_dir, 'jams/jams_' + dataset_description + '_'
                                    + basename(model_loader.file).replace('.h5', ''))
                else:
                    jams_dir = None
                windowed = True
                predictions = global_predict(model, input_shape, windowed, ground_truth, features, normalizer, jams_dir)
                dump_predictions(predictions, model_loader, dataset_description)
                # don't keep all models in memory
                del model
                K.clear_session()
            else:
                predictions = load_predictions(model_loader, dataset_description)
            same_kind_results.append(predictions)
    return results


def prediction_file(model_loader, testset_description):
    """
    Standardized name for a prediction. 
    
    :param model_loader: model loader 
    :param testset_description: test set description
    :return: file name for predictions
    """
    return model_loader.file.replace('.h5', '_pred_{}.joblib'.format(testset_description))


def load_predictions(model_loader, testset_description):
    """
    Load predictions for a model and a test set.

    :param model_loader: model loader
    :param testset_description: test set description
    :return: predictions
    """
    return load(prediction_file(model_loader, testset_description))


def dump_predictions(predictions, model_loader, testset_description):
    """
    Store predictions for a model and a test set.

    :param predictions: predictions
    :param model_loader: model loader
    :param testset_description: test set description
    """
    dump(predictions, prediction_file(model_loader, testset_description))


def global_predict(model, input_shape, windowed, ground_truth, features, normalizer, jams_dir=None):
    """
    Predict global tempo values using the given model, spectrogram input shape, ground truth,
    features and normalizer function.

    :param model: model
    :param input_shape: single spectrogram shape
    :param windowed: if ``True``, predict for multiple windows
    :param ground_truth: ground truth necessary to convert indices to actual labels
    :param features: dict, mapping keys to spectrograms
    :param normalizer: function that takes a numpy array and normalizes it (needed before prediction)
    :param jams_dir: if not None, dump predictions as jams zip file into the given directory
    :return: dict, mapping keys to predicted labels
    """
    results = {}
    model_input_length = input_shape[1]
    for key in ground_truth.labels.keys():
        if key not in features:
            logging.warning('Prediction for dataset \'{}\': Spectrogram not found. Id=\'{}\''
                            .format(ground_truth.name, key))
            continue

        # make sure we don't modify the original!
        spectrogram = np.copy(features[key])

        if windowed:
            length = spectrogram.shape[1]
            samples = []

            # half overlapping
            hop_size = input_shape[1] // 2
            pos = 0
            while pos + model_input_length <= length:
                sample = spectrogram[:, pos:pos + model_input_length, :]
                if normalizer is not None:
                    sample = normalizer(sample)
                samples.append(np.reshape(sample, (1, *input_shape)))
                pos += hop_size
            X = np.vstack(samples)
        else:
            # this assumes that we can predict spectrograms of arbitrary lengths (dim=1)
            if normalizer is not None:
                spectrogram = normalizer(spectrogram)
            X = np.expand_dims(spectrogram, axis=0)

        predictions = model.predict(X, X.shape[0])
        predictions = np.sum(predictions, axis=0)
        discrete_label = ground_truth.get_label(np.argmax(predictions))
        interpolated_label = ground_truth.get_label(quad_interpol_argmax(predictions))

        results[key] = discrete_label, interpolated_label, predictions

        # dump interpolated labels as JAMS
        if jams_dir is not None:
            create_jam_file(jams_dir, key, ground_truth.name, interpolated_label)

    if jams_dir is not None:
        def zipdir(path, zip_handle):
            for root, dirs, files in walk(path):
                for file in files:
                    zip_handle.write(join(root, file), file)

        if exists(jams_dir):
            zip_file = zipfile.ZipFile(jams_dir + '.zip', 'w', zipfile.ZIP_DEFLATED)
            zipdir(jams_dir, zip_file)
            rmtree(jams_dir, ignore_errors=True)
        else:
            logging.warning('Failed to create jams zip file. Jams dir does not exist: {}'.format(jams_dir))

    return results


def quad_interpol_argmax(y):
    """
    Find argmax for quadratic interpolation around argmax of y.

    :param y: array
    :return: float (index) of interpolated max
    """
    x = np.argmax(y)
    if x == 0 or x == y.shape[0]-1:
        return x
    z = np.polyfit([x - 1, x, x + 1], [y[x - 1], y[x], y[x + 1]], 2)
    # find (float) x value for max
    argmax = -z[1] / (2. * z[0])
    return argmax


def create_jam_file(jams_dir, key, corpus, estimate):
    """
    Create a jam file for the given key, corpus, and estimate.

    :param jams_dir: dir
    :param key: file id
    :param corpus: corpus (e.g. Ballroom)
    :param estimate: tempo estimate
    """
    makedirs(jams_dir, exist_ok=True)
    if estimate < 0:
        logging.warning('Attempting to store tempo estimate for {} that is less than 0: {}'.format(key, estimate))
        estimate = 0
    tempo = jams.Annotation(namespace='tempo')
    tempo.append(time=0.0,
                 duration='nan',
                 value=estimate,
                 confidence=1.0)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version='1.0',
        data_source='')
    jam = jams.JAMS()
    jam.annotations.append(tempo)
    jam.file_metadata.duration = 5000  # bogus value to please JAMS
    jam.save(join(jams_dir, '{}.jams'.format(key)))

