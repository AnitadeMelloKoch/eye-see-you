import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, window_size, feature_size=120):
        super().__init__()

        self.window_size = window_size
        self.feature_size = feature_size

        self.cnn_layer1 = tf.keras.layers.Conv1D(
            filters=3,
            kernel_size=2,
            activation='tanh',
            strides=1,
            padding='same'
        )
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.cnn_layer2 = tf.keras.layers.Conv1D(
            filters=6,
            kernel_size=2,
            activation='tanh',
            strides=2,
            padding='same'
        )
        self.dropout_2 = tf.keras.layers.Dropout(0.2)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()


        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(feature_size*2, activation='tanh')
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.dropout_3 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(feature_size)

    def call(self, inputs):
        output = self.cnn_layer1(inputs)
        output = self.batchnorm1(output)
        output = self.dropout_1(output)
        output = self.cnn_layer2(output)
        output = self.batchnorm2(output)
        output = self.dropout_2(output)
        output = self.flatten(output)
        output = self.dense1(output)
        output = self.batchnorm3(output)
        output = self.dropout_3(output)
        output = self.dense2(output)
        
        return output


class Split_CNN(tf.keras.Model):

    def __init__(self, window_size, feature_size=120):
        super().__init__()

        self.window_size = window_size
        self.feature_size = feature_size

        self.cnn_layers = []

        for x in range(window_size):
            self.cnn_layers.append(tf.keras.layers.Conv1D(
                filters=3,
                kernel_size=3,
                activation='tanh',
                strides=1,
                padding='same'
            ))

        self.dropout = tf.keras.layers.Dropout(0.2)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(window_size*3, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(feature_size)

    def call(self, inputs):

        # shape of inputs should be (batchsize, window_size, feature_size)
        conv_outputs = []
        for x in range(self.window_size):
            conv_input = inputs[:,x,:]
            conv_input = tf.expand_dims(conv_input, axis=1)
            conv_outputs.append(self.cnn_layers[x](conv_input))

        combined_conv_outputs = tf.concat(conv_outputs, axis=2)
        combined_conv_outputs = tf.squeeze(combined_conv_outputs, axis=1)

        output = self.batchnorm(combined_conv_outputs)
        output = self.dropout(output)
        output = self.dense1(output)
        output = self.dense2(output)

        return output