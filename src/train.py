from src.cnn.model import Split_CNN, CNN
import tensorflow as tf
import argparse
import src.data.process_data as data
import os
import logging

def get_model(model_name, window_size):
    if model_name == 'split_cnn':
        return Split_CNN(window_size)
    elif model_name == 'cnn':
        return CNN(window_size)
    else:
        raise Exception("Model name not recognised")

def get_optimizer(optimizer, learning_rate):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        return tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        return tf.keras.optimizers.experimental.Adadelta(learning_rate=learning_rate)
    else:
        raise Exception('Optimizer not recognised')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['cnn','split_cnn'],
        help='Available models: cnn, split_cnn',
        required=True)
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs',
        default=100
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Model learning rate',
        default=0.0005
    )
    parser.add_argument(
        '--batchsize',
        type=int,
        help='Training batchsize',
        default=64
    )
    parser.add_argument(
        '--optimizer',
        choices=['adam', 'sgd', 'rmsprop', 'adadelta'],
        help='Select optimizer for model',
        default='adam'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        help='Window size for data',
        default=10
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Save path for model',
        required=True
    )
    parser.add_argument(
        '--load_model',
        action='store_true'
    )
    parser.add_argument(
        '--skip_train',
        action='store_true'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to csv files for data extraction',
        required=True
    )
    parser.add_argument(
        '--train_test_split',
        type=float,
        help='Percentage of data that should be set aside for validation',
        default=0.2
    )

    args = parser.parse_args()

    base_path = args.save_path
    save_path = os.path.join(base_path, 'ckpt')
    log_path = os.path.join(base_path, 'logs')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_path, 'run_settings.log'), level=logging.INFO)
    logging.info("Model: {}".format(args.model))
    logging.info("Optimizer: {}".format(args.optimizer))
    logging.info("Learning Rate: {}".format(args.learning_rate))
    logging.info("Training Epochs: {}".format(args.epochs))
    logging.info("Batchsize: {}".format(args.batchsize))
    logging.info("Window Size: {}".format(args.window_size))
    logging.info("="*20)
    logging.info("Test Train Split: {}".format(args.train_test_split))
    logging.info("Load Previous Model: {}".format(args.load_model))
    logging.info("Model Trained: {}".format(not args.skip_train))
    logging.info("Data Path: {}".format(args.data_path))

    if args.load_model:
        model = tf.keras.models.load_model(os.path.join(save_path, 'model.ckpt'))
    else:
        model = get_model(args.model, args.window_size)
        optimizer = get_optimizer(args.optimizer, args.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(),
            metrics=tf.keras.metrics.MeanSquaredError()
        )

    csv_data = data.get_data(args.data_path)
    samples, labels = data.window_data(csv_data, args.window_size)

    if not args.skip_train:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_path)

        model.fit(
            x=samples,
            y=labels,
            batch_size=args.batchsize,
            epochs=args.epochs,
            validation_split=args.train_test_split,
            callbacks=[tensorboard_cb]
        )

    model.save(os.path.join(save_path, 'model.ckpt'))


