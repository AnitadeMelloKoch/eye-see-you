import tensorflow as tf

class CNN(tf.keras.Model):

    def __init__(self, window_size, feature_size=120):
        super().__init__()

        self.window_size = window_size
        self.feature_size = feature_size

        self.cnn_layers = []

        for x in range(window_size):
            self.cnn_layers.append(tf.keras.layers.Conv1D(
                filters=3,
                kernel_size=3,
                activation='relu'
            ))

        self.dense1 = tf.keras.layers.Dense(window_size*3, activation='relu')
        self.dense2 = tf.keras.layers.Dense(feature_size, activation='relu') # lets try relu for now

    def call(self, inputs):

        # shape of inputs should be (batchsize, window_size, feature_size)
        conv_outputs = []
        for x in range(self.window_size):
            conv_outputs.append(self.cnn_layers[x](inputs[:,x,:]))

        combined_conv_outputs = tf.concat(conv_outputs, axis=1)
        output = self.dense1(combined_conv_outputs)
        output = self.dense2(output)

        return output