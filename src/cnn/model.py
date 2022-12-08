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

class BlinkSplitCNN(tf.keras.Model):

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
        
        # eye feature extraction
        self.e_dropout1 = tf.keras.layers.Dropout(0.2)
        self.e_batchnorm1 = tf.keras.layers.BatchNormalization()
        self.e_dense1 = tf.keras.layers.Dense(window_size*3, activation='tanh')
        self.e_dropout2 = tf.keras.layers.Dropout(0.2)
        self.e_batchnorm2 = tf.keras.layers.BatchNormalization()
        self.e_dense2 = tf.keras.layers.Dense(feature_size*2)
        self.e_dense3 = tf.keras.layers.Dense(feature_size)

        # blink prediction
        self.b_dropout1 = tf.keras.layers.Dropout(0.2)
        self.b_batchnorm1 = tf.keras.layers.BatchNormalization()
        self.b_dense1 = tf.keras.layers.Dense(window_size, activation='tanh')
        self.b_dropout2 = tf.keras.layers.Dropout(0.2)
        self.b_batchnorm2 = tf.keras.layers.BatchNormalization()
        self.b_dense2 = tf.keras.layers.Dense(window_size//2, activation='tanh')
        self.b_dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):

        conv_outputs = []
        for x in range(self.window_size):
            conv_input = inputs[:,x,:]
            conv_input = tf.expand_dims(conv_input, axis=1)
            conv_outputs.append(self.cnn_layers[x](conv_input))

        combined_conv_outputs = tf.concat(conv_outputs, axis=2)
        combined_conv_outputs = tf.squeeze(combined_conv_outputs, axis=1)

        predicted_eye_feature = self.e_batchnorm1(combined_conv_outputs)
        predicted_eye_feature = self.e_dropout1(predicted_eye_feature)
        predicted_eye_feature = self.e_dense1(predicted_eye_feature)
        predicted_eye_feature = self.e_batchnorm2(predicted_eye_feature)
        predicted_eye_feature = self.e_dropout2(predicted_eye_feature)
        predicted_eye_feature = self.e_dense2(predicted_eye_feature)
        predicted_eye_feature = self.e_dense3(predicted_eye_feature)

        predicted_blink = self.b_batchnorm1(combined_conv_outputs)
        predicted_blink = self.b_dropout1(predicted_blink)
        predicted_blink = self.b_dense1(predicted_blink)
        predicted_blink = self.b_batchnorm2(predicted_blink)
        predicted_blink = self.b_dropout2(predicted_blink)
        predicted_blink = self.b_dense2(predicted_blink)
        predicted_blink = self.b_dense3(predicted_blink)

        return tf.concat([predicted_eye_feature, predicted_blink], axis=-1)

    def compile(self, optimizer, eye_loss, blink_loss, metric):
        super().compile()
        self.optimizer = optimizer
        self.eye_loss = eye_loss
        self.blink_loss = blink_loss
        self.metric = metric

    def train_step(self, data):
        x, y = data
        y_eye = y[:,:-1]
        y_blink = y[:,-1]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred_eye = y_pred[:,:-1]
            y_pred_blink = y_pred[:,-1]
            eye_loss = self.eye_loss(y_eye, y_pred_eye)
            blink_loss = self.blink_loss(y_blink, y_pred_blink)
            loss = eye_loss + blink_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        metric = self.metric(y_eye, y_pred_eye)

        return {"eye_feature_loss": eye_loss, "blink_loss": blink_loss, "mse": metric}
        
    def test_step(self, data):
        x, y = data
        y_eye = y[:,:-1]
        y_blink = y[:,-1]

        y_pred = self(x, training=True)
        y_pred_eye = y_pred[:,:-1]
        y_pred_blink = y_pred[:,-1]
        eye_loss = self.eye_loss(y_eye, y_pred_eye)
        blink_loss = self.blink_loss(y_blink, y_pred_blink)
        metric = self.metric(y_eye, y_pred_eye)

        return {"eye_feature_loss": eye_loss, "blink_loss": blink_loss, "mse": metric}
        



