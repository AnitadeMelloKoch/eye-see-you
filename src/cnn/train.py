import tensorflow as tf
import src.cnn.model

def get_windowed_data(data, window_size):
    # get data and labels based on window size
    # data should be shape = (num_samples, feature_size)
    num_samples = data.shape[0]//(window_size+1)
    samples = []
    labels = []
    for x in range(num_samples):
        offset = x*(window_size+1)
        samples.append(data[offset:offset+window_size,:])
        labels.append(data[offset+window_size+1])
    
    return samples, labels

def train(model, train_x, train_y, test_x, test_y, epochs):
    pass

def build_model(optimizer, metrics, loss):
    pass

def test(model, x, y):
    pass

