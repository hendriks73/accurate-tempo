"""
Main entry point to train multiple models and predict on test sets.
"""

import logging
import argparse
import os
import warnings
from datetime import datetime
from os.path import join, exists, basename

import tensorflow as tf
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam

from accuratetempo.evaluation import evaluation_reports
from accuratetempo.generator import DataGenerator, normal_tempo_augmenter, identity_tempo_augmenter
from accuratetempo.groundtruth import GroundTruth
from accuratetempo.loader import create_mel_sample_loader
from accuratetempo.models import ModelLoader
from accuratetempo.network.vgg import create_vgg_like_model
from accuratetempo.normalizer import std_normalizer
from accuratetempo.prediction import predict_from_models


def train_and_predict(job_dir, model_dir, features, train, valid, test,
                      log_scale=False, augment=False, refined=False, classes=256, hop=512):
    """
    Main function to execute training and prediction.

    :param hop: hop length of features
    :param classes: number of different classes we want to use (for BPM 30-285)
    :param refined: use refined annotations during training and validation
    :param augment: augment training samples
    :param log_scale: use log scale when mapping ground truth to indices
    :param test: .tsv file with test annotations
    :param valid: .tsv file with validation annotations
    :param train: .tsv file with train annotations
    :param features: feature dictionary in .joblib format
    :param job_dir: working directory
    :param model_dir: directory to store models and predictions in
    """

    # create job_dir/model_dir sub_dirs, to avoid configuration collisions
    model_subdir = 'log_scale={}_augment={}_refined={}_classes={}_hop={}'\
        .format(log_scale, augment, refined, classes, hop)
    model_dir = join(model_dir, model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    job_dir = join(job_dir, model_subdir)
    os.makedirs(job_dir, exist_ok=True)

    results_file_name = join(job_dir, 'results-{}.txt'.format(datetime.now().strftime('%Y%m%d-%H%M%S.%f')))
    logging.basicConfig(filename=results_file_name, filemode='w', format='[%(asctime)s] %(levelname)-8s: %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S.%s', level=logging.DEBUG)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] %(levelname)-8s: %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    print('Writing results to {}'.format(results_file_name))
    logging.info('Starting train and predict. {}\n'.format(datetime.now()))

    if tf.test.gpu_device_name():
        logging.info('Default GPU: {}'.format(tf.test.gpu_device_name()))
    else:
        logging.warning("Failed to find default GPU.")

    logging.info('======================')
    logging.info(' log_scale = {}'.format(log_scale))
    logging.info(' augment   = {}'.format(augment))
    logging.info(' refined   = {}'.format(refined))
    logging.info(' classes   = {}'.format(classes))
    logging.info(' hop       = {}'.format(hop))
    logging.info('======================')

    spectrum_length = 131072 // hop
    input_shape = (40, spectrum_length, 1)
    batch_size = 32
    lr = 0.001
    epochs = 5000
    patience = 250
    runs = 7
    # model settings for dropout and number of filters
    filters = 8
    if augment:
        dropout = 0.3
    else:
        # slightly higher dropout, when augmenting to avoid overfitting
        dropout = 0.4

    normalizer = std_normalizer
    if augment:
        augmenter = normal_tempo_augmenter
    else:
        augmenter = identity_tempo_augmenter
    logging.debug('Augmenter: {}'.format(augmenter.__name__))

    if refined:
        train = train.replace('.tsv', '_refined.tsv')
        valid = valid.replace('.tsv', '_refined.tsv')

    features_file = features.format(hop)
    features = joblib.load(features_file)
    logging.debug('Loaded features from {}'.format(features_file))

    logging.debug('Input shape: {}'.format(input_shape))
    train_loader = create_mel_sample_loader(features,
                                            shape=input_shape,
                                            random_offset=True,
                                            normalizer=normalizer)
    valid_loader = create_mel_sample_loader(features,
                                            shape=input_shape,
                                            random_offset=False,
                                            normalizer=normalizer)
    logging.info('Loading ground truth...')
    train_ground_truth = GroundTruth(train, nb_classes=classes, log_scale=log_scale)
    logging.debug('Loaded {} training annotations from {} (log_scale={}).'
        .format(len(train_ground_truth.labels), train, log_scale))

    valid_ground_truth = GroundTruth(valid, nb_classes=classes, log_scale=log_scale)
    logging.debug('Loaded {} validation annotations from {} (log_scale={}).'
        .format(len(valid_ground_truth.labels), valid, log_scale))

    test_ground_truth = GroundTruth(test, nb_classes=classes, log_scale=log_scale)
    logging.debug('Loaded {} test annotations from {} (log_scale={}).'
        .format(len(test_ground_truth.labels), test, log_scale))

    if not log_scale:
        classes_per_bpm = classes // 256
        logging.debug('BPM resolution: {}'.format(1./classes_per_bpm))
    else:
        logging.debug('BPM resolution logarithmic.')

    # this code was originally meant for evaluating different
    # architectures with different parameters, hence the model dict.
    models = {}
    same_kind_models = []
    for run in range(runs):
        logging.info('Creating model for run {}.'.format(run))
        model = create_vgg_like_model(input_shape=input_shape, filters=filters, dropout=dropout)
        model_loader = training(run=run, epochs=epochs, patience=patience, batch_size=batch_size, lr=lr,
                                model=model,
                                input_shape=input_shape,
                                augmenter=augmenter,
                                train_ground_truth=train_ground_truth,
                                valid_ground_truth=valid_ground_truth,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                model_dir=model_dir,
                                job_dir=job_dir)

        same_kind_models.append(model_loader)
        K.clear_session()
    model_name = same_kind_models[0].name
    models[model_name] = same_kind_models

    # predict and evaluate for test and validation set
    predict_from_models(features, input_shape, models, normalizer, valid_ground_truth)
    predict_from_models(features, input_shape, models, normalizer, test_ground_truth)

    evaluation_reports(models, test_ground_truth)

    evaluation_reports(models, valid_ground_truth)


def training(run=0, initial_epoch=0, epochs=5000, patience=50, batch_size=32, lr=0.001, model=None,
             input_shape=(40, 256, 1), augmenter=None, train_ground_truth=None, valid_ground_truth=None,
             train_loader=None, valid_loader=None, model_dir='./', job_dir='./'):

    """
    Train a given model with the given parameters.
    If the model already exists in the file system, this function does not attempt to train,
    but simply returns a reference to the already existing model.

    :param run: run number
    :param initial_epoch: initial epoch
    :param epochs: epochs
    :param patience: early stopping patience
    :param batch_size: batch size
    :param lr: initial learning rate
    :param model: model
    :param input_shape: input shape
    :param augmenter: augmentation function
    :param train_ground_truth: train ground truth
    :param valid_ground_truth: validation ground truth
    :param train_loader: training sample loader
    :param valid_loader: validation sample loader
    :param model_dir: directory to store trained models in
    :param job_dir: working directory (e.g., for TensorBoard files)
    :return: a model loader (i.e., a fancy file name)
    """

    # and save to model_dir
    model_file = join(model_dir, 'model_{}_run={}.h5'.format(model.name, run))
    if exists(model_file):
        logging.info('Model exist, skipping training. File={}'.format(model_file))
        return ModelLoader(model_file, model.name)

    # Write an empty model file, to signal that we are on this!
    # I.e. other, parallel processed will not try to train this run of this model.
    # Yes, you've guessed correctly, this only makes sense when starting the training multiple
    # times in different processes that do not communicate. A poor man's lock file.
    open(model_file, 'a').close()

    tensorboard_dir = join(job_dir, 'tensorboard/' + basename(model_file).replace('.h5', '') + '/')
    os.makedirs(tensorboard_dir, exist_ok=True)

    checkpoint_model_file = model_file.replace('.h5', '_checkpoint.h5')

    binarizer = OneHotEncoder(sparse=False)
    binarizer.fit([[c] for c in range(train_ground_truth.nb_classes)])
    model.compile(loss='categorical_crossentropy', optimizer=(Adam(lr=lr)), metrics=['accuracy'])

    logging.debug('Run {}, {}, params={}'.format(run, model.name, model.count_params()))
    model.summary(print_fn=logging.info)

    train_generator = DataGenerator(train_ground_truth,
                                    train_loader,
                                    binarizer, batch_size=batch_size,
                                    sample_shape=input_shape, shuffle=True, augmenter=augmenter)

    if len(valid_ground_truth) < batch_size:
        valid_generator = None
        warnings.warn('The validation ground truth contains fewer samples ({}) than batch_size={}.'
                      .format(len(valid_ground_truth), batch_size) + ' Validation impossible.')
    else:
        valid_generator = DataGenerator(valid_ground_truth,
                                        valid_loader,
                                        binarizer, batch_size=batch_size,
                                        sample_shape=input_shape, shuffle=False, augmenter=None)

    def learning_rate_schedule(epoch):
        """
        Starting at 500 epochs, halve the learning rate every 250 epochs.

        :param epoch: current epoch
        """
        c = initial_epoch + 500
        if epoch < c:
            return lr
        else:
            c_epoch = epoch - c
            new_lr = lr * 0.5 ** (c_epoch / 250)  # halve every 250 epochs
            return new_lr

    model_checkpoint = ModelCheckpoint(checkpoint_model_file, monitor='val_loss')
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
                 model_checkpoint,
                 TensorBoard(log_dir=tensorboard_dir),
                 LearningRateScheduler(learning_rate_schedule, verbose=1)]

    history = model.fit_generator(train_generator,
                                  initial_epoch=initial_epoch,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_data=valid_generator)
    logging.info('Finished training for run {} after {} epochs.'.format(run, len(history.history['loss'])))
    logging.info('Best \'{}\': {}'.format(model_checkpoint.monitor, model_checkpoint.best))
    logging.debug('Run {}, lr={}, batch_size={}, epochs={}/{}, augmenter={}, model_name={}, log_scale={}'
          .format(run, lr, batch_size, len(history.history['loss']), epochs, augmenter.__name__, model.name,
                  train_ground_truth.log_scale))
    model = load_model(checkpoint_model_file)
    model.save(model_file)
    return ModelLoader(model_file, model.name, history)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''Main entry point for training and evaluating different setups.   
Both feature .joblib files and ground truth files have to be provided.
Note that seven runs are conducted.''')

    # Input Arguments
    parser.add_argument(
        '--valid',
        help='.tsv ground truth for validation',
        required=True
    )
    parser.add_argument(
        '--train',
        help='.tsv ground truth for training',
        required=True
    )
    parser.add_argument(
        '--test',
        help='.tsv ground truth for testing',
        required=True
    )
    parser.add_argument(
        '--features',
        help='Features for ALL datasets/splits.',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='job working directory',
        default='./job',
        required=True
    )
    parser.add_argument(
        '--model-dir',
        help='model directory',
        default='./models',
        required=True
    )
    parser.add_argument(
        '--log-scale',
        help='use logarithmic scaling when mapping classes to BPM values',
        action="store_true"
    )
    parser.add_argument(
        '--augment',
        help='augment training samples using scale & crop',
        action="store_true"
    )
    parser.add_argument(
        '--refined',
        help='use refined train and validation data',
        action="store_true"
    )
    parser.add_argument(
        '--classes',
        type=int,
        help='number of different classes to use (256|1024)',
        required=True
    )
    parser.add_argument(
        '--hop',
        type=int,
        help='feature hop length (256|512)',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__
    return arguments


def main():
    arguments = parse_arguments()
    train_and_predict(**arguments)


if __name__ == '__main__':
    main()
