import csv
import glob
import numpy as np

fmPosKeys = ['fmPos_%04d' % i for i in range(0, 468)]
eyeFeaturesKeys = ['eyeFeatures_%04d' % i for i in range(0, 120)]
fieldnames = (['participant','frameImageFile','frameTimeEpoch','frameNum','mouseMoveX','mouseMoveY',
               'mouseClickX','mouseClickY','keyPressed','keyPressedX','keyPressedY',
               'tobiiLeftScreenGazeX','tobiiLeftScreenGazeY','tobiiRightScreenGazeX','tobiiRightScreenGazeY',
               'webGazerX','webGazerY','error','errorPix'])
fieldnames.extend( fmPosKeys )
fieldnames.extend( eyeFeaturesKeys )

def get_data(data_path):

    eye_features = []
    
    for name in glob.glob(data_path + '/*.csv'):
        # print(name)
        with open(name) as csvfile:
            data = csv.DictReader(csvfile, fieldnames=fieldnames,delimiter=',',quoting=csv.QUOTE_ALL)
            for row in data:
                features = []
                for key in eyeFeaturesKeys:
                    features.append(float(row[key])) 
                eye_features.append(features)
    
        # eye_features.append([-100]*120)
                        
    print('Eye-feature shape', np.array(eye_features).shape)
    return np.array(eye_features)

def split_test_train(samples, labels, split_percentage):
    num_samples = len(samples)
    num_test_samples = int(num_samples*split_percentage)
    train_samples = samples[:-num_test_samples,:]
    train_labels = labels[:-num_test_samples,:]
    test_samples = samples[-num_test_samples:,:]
    test_labels = labels[-num_test_samples:,:]

    return train_samples, train_labels, test_samples, test_labels

def window_data(data, window_size):
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

# data_path = '../WebGazer/www/data/FramesDataset'
# get_data(data_path)
