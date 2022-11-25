from src.cnn.model import CNN
import tensorflow as tf
import argparse
import src.data.process_data as data

def get_model(model_name, window_size):
    if model_name == 'cnn':
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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    choices=['cnn'],
    help='Available models: cnn',
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
    help='Window size for data'
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
    default=0.4
)

args = parser.parse_args()

model = get_model(args.model, args.window_size)
optimizer = get_optimizer(args.optimizer, args.learning_rate)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=tf.keras.metrics.MeanSquaredError()
)

csv_data = data.get_data(args.data_path)
samples, labels = data.window_data(csv_data, args.window_size)

if not args.skip_train:
    model.fit(
        x=samples,
        y=labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.train_test_split
    )
