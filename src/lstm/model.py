import tensorflow as tf

class LSTM(tf.keras.Model):
    def __init__(self, window_size, feature_size=120):
        super().__init__()

        self.window_size = window_size
        self.feature_size = feature_size

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(window_size, input_shape=(window_size, feature_size)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(feature_size*2, activation='tanh'),
            tf.keras.layers.Dense(feature_size)
        ])

    def call(self, input):
        return self.model(input)

