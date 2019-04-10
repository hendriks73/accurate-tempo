"""
Simple, textual evaluation of trained models using the provided ground truth.
"""

import logging
import numpy as np
from tensorflow.python.keras import backend as K

from accuratetempo.prediction import load_predictions


def evaluation_reports(models, ground_truth, interpolated=True):
    """
    Evaluate predictions (must already be available via files)

    :param interpolated: if ``true``, quadratically interpolate predictions to
    achieve higher BPM resolution than we have classes
    :param models: dictionary of different kinds of models, which contains lists of models for each kind
    :param ground_truth: ground truth to evaluate against
    """
    report =  '\n\nsub-class interpolation={}\n\n'.format(interpolated)\
                    + '{:<16} | {:<4} | {:<18} | {:<18} | {:<18} | {:<19} | {:<19} | {}\n'\
                        .format('Testset', 'Runs', 'mean Acc0', 'mean Acc1', 'mean Acc2', 'Mape1', 'Mape2', 'Model') \
                    + '-------------------------------------------------------------------------------------------------------------------------------------------------------------\n'
    row_template = '{:<16} | {:>4} | {:7.2%} ({:8.3%}) | {:7.2%} ({:8.3%}) | {:7.2%} ({:8.3%}) | {:8.4%} ({:8.4%}) | {:8.4%} ({:8.4%}) | {}\n'
    sorted_names = list(models.keys())
    sorted_names.sort()

    logging.info('=== Evaluation for dataset \'{}\' ==='.format(ground_truth.name))
    for model_name in sorted_names:
        same_kind_models = models[model_name]
        accuracies = []
        for run, model_loader in enumerate(same_kind_models):
            logging.info('{}. run {}'.format(run, model_name))
            model = model_loader.load()
            predictions = load_predictions(model_loader, ground_truth.name)
            acc = global_accuracy(ground_truth, predictions, interpolated=interpolated)
            accuracies.append(np.array(acc))
            # don't keep all models in memory
            del model
            # don't keep predictions
            del predictions
            # clean up TensorFlow/Keras, ESSENTIAL, when evaluating many models.
            K.clear_session()
        for a in accuracies:
            logging.debug('acc    : {}'.format(a.tolist()))
        np_acc = np.vstack(accuracies)
        means = np.mean(np_acc, axis=0)
        stdevs = np.std(np_acc, axis=0)
        best_run_max = np.argmax(np_acc, axis=0)
        best_run_min = np.argmin(np_acc, axis=0)
        best_run = best_run_max[0:3].tolist() + best_run_min[3:5].tolist()
        logging.debug('bestrun: {}'.format(best_run))
        logging.debug('means  : {}'.format(means.tolist()))
        logging.debug('stddevs: {}'.format(stdevs.tolist()))

        report += row_template.format(ground_truth.name,
                                                    len(same_kind_models),
                                                    means[0], stdevs[0],
                                                    means[1], stdevs[1],
                                                    means[2], stdevs[2],
                                                    means[3], stdevs[3],
                                                    means[4], stdevs[4],
                                                    model_name)
    report += '-------------------------------------------------------------------------------------------------------------------------------------------------------------\n'
    logging.info(report)


def global_accuracy(ground_truth, predictions, interpolated=False):
    """
    Compute several accuracy measures for the given ground truth and predictions.

    :param ground_truth: ground truth object
    :param predictions: dict of ids and predictions (regular and interpolated)
    :param interpolated: flag, whether to use interpolated or regular predictions
    :return: list of accuracy measures
    """
    acc0_sum = 0
    acc1_sum = 0
    acc2_sum = 0

    mape1_sum = 0
    mape2_sum = 0

    count = 0
    for key in ground_truth.labels.keys():
        if key in predictions:
            if interpolated:
                predicted_label = predictions[key][1]
            else:
                predicted_label = predictions[key][0]

            true_label = ground_truth.labels[key]

            mape1_sum += mape1(true_label, predicted_label)
            mape2_sum += mape2(true_label, predicted_label)

            if same_tempo(true_label, predicted_label, tolerance=0.0):
                acc0_sum += 1
            if acc1(true_label, predicted_label):
                acc1_sum += 1
            if acc2(true_label, predicted_label):
                acc2_sum += 1

        else:
            logging.warning('No prediction for key {}'.format(key))
            pass

        count += 1
    f_count = float(count)
    acc0_result = acc0_sum / f_count
    acc1_result = acc1_sum / f_count
    acc2_result = acc2_sum / f_count

    mape1_result = mape1_sum / f_count
    mape2_result = mape2_sum / f_count

    return [acc0_result, acc1_result, acc2_result, mape1_result, mape2_result]


def same_tempo(true_value, estimated_value, factor=1., tolerance=0.04):
    """
    Compares two tempi given a factor and a tolerance.

    :param true_value: reference value
    :param estimated_value: predicted value
    :param factor: factor to multiply the predicted value with
    :param tolerance: tolerance
    :return: true or false
    """
    if tolerance is None or tolerance == 0.0:
        return np.round(estimated_value * factor) == np.round(true_value)
    else:
        return np.abs(estimated_value * factor - true_value) < true_value * tolerance


def mape1(true_value, estimate_value, factor=1.):
    """
    Mean absolute percentage error.
    See https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    :param true_value: reference value (must not be zero)
    :param estimate_value: estimated value
    :param factor: factor to multiply the predicted value with
    :return: positive float (the error)
    """
    return np.mean(np.abs((true_value - estimate_value * factor) / true_value))


def mape2(true_value, estimate_value):
    """
    Minimum of the MAPE for predicted values times {1, 2, 1/2, 3, 1/3}.

    :param true_value: reference value (must not be zero)
    :param estimate_value: estimated value
    :param factor: factor to multiply the predicted value with
    :return: positive float (the error)
    """
    return np.min(np.array([
        mape1(true_value, estimate_value),
        mape1(true_value, estimate_value, factor=2.0),
        mape1(true_value, estimate_value, factor=0.5),
        mape1(true_value, estimate_value, factor=3.0),
        mape1(true_value, estimate_value, factor=1. / 3.),
    ]))


def acc1(true_value, estimate_value, tolerance=0.04):
    """
    Accuracy 1.
    
    :param true_value: reference value (must not be zero)
    :param estimate_value: estimated value 
    :param tolerance: tolerance 
    :return: true or false
    """
    return same_tempo(true_value, estimate_value, factor=1.0, tolerance=tolerance)


def acc2(true_value, estimate_value, tolerance=0.04):
    """
    Accuracy 2.

    :param true_value: reference value (must not be zero)
    :param estimate_value: estimated value 
    :param tolerance: tolerance 
    :return: true or false
    """
    return same_tempo(true_value, estimate_value, tolerance=tolerance)\
           or same_tempo(true_value, estimate_value, factor=2., tolerance=tolerance) \
           or same_tempo(true_value, estimate_value, factor=1. / 2., tolerance=tolerance) \
           or same_tempo(true_value, estimate_value, factor=3., tolerance=tolerance) \
           or same_tempo(true_value, estimate_value, factor=1. / 3., tolerance=tolerance)
