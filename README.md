# eye-see-you

## Running the models

To run a model use the command
`python -m src.train <args>`
from the base directory.

Available arguments
- `--model`: Specify what architecture should be used. Available options are `cnn` and `split_cnn`. *Required*
- `--epochs`: Number of epochs to train for. *Default=100*
- `--learning_rate`: Learning rate to use during training. Currently no learning rate scheduler is used. *Default=0.0005*
- `--batchsize`: Training batch size. *Default=64*
- `--optimizer`: Select optimizer to use. Current available optimizers are `adam`, `sgd`, `rmsprop` and `adadelta`. *Default=adam*
- `--window_size`: Window size that should be used for the model. *Default=10*
- `--save_path`: Location to save the model. Logs will be stored in `<save_path>/logs` and models will be saved in `<save_path>/ckpt`. *Required*
- `--load_model`: If this argument is present the model located at `<save_path>/ckpt` will be loaded. *Flag*
- `--skip_train`: If present the model will not be trained. This can be used if you only wish to load a previous model but do not want to train it. *Flag*
- `--data_path`: Provides the location data is stored. This expect data to be stored in csv files in the format provided by WebGazer. *Required*
- `--train_test_split`: The percentage of data that should be used for validation during training. *Default=0.2*

## Good to knows

1. We predict the 120-dimension eye features as extracted from the webgazer dataset. The eye features can then be used to train a new model which has a specified objective (e.g. x,y coordinates on a screen or parameters for a graphic model).
2. All models use the `Huber` loss function. This loss function is good for robust regression.
3. We do no further data processing on the eye features which can range from ~-400 to ~400, some of which are close to 0.
4. Because values can be positive or negative we use `tanh` for all activation functions and use no activation on the final layer.
5. Training statistics are saved using tensorboard.
6. All arguments provided when running the model are stored in the `run_settings.log` file which is found in the logs directory.
7. All data and preprocessing is taken from the [WebGazer dataset](https://webgazer.cs.brown.edu/data/).