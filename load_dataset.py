import json
from PIL import Image
import numpy as np
import theano
import theano.tensor as T

def get_image_vector(path):
    data = np.array(Image.open(path))

    try:
        return np.reshape(data,(data.shape[0]*data.shape[1],))
    except:
        import pdb;pdb.set_trace()

def load_dataset(data_filename, valid_size=30, test_size=30):
    filenames = []
    points = []
    with open(data_filename) as data_file:
        for image in json.load(data_file):
            if len(image['annotations']) == 0:
                continue
            filenames.append(image['filename'])
            point = image['annotations'][0]
            points.append((point['x'],point['y']))        
    
    x = np.array([get_image_vector(filename) for filename in filenames])/255.0
                      
    y = np.array(points)/np.tile(np.array([400.0, 240.0]),(len(points),1))
    
    valid_set_x = theano.shared(x[:valid_size])
    test_set_x = theano.shared(x[valid_size:valid_size+test_size])
    train_set_x = theano.shared(x[valid_size+test_size:])

    valid_set_y = theano.shared(y[:valid_size])
    test_set_y = theano.shared(y[valid_size:valid_size+test_size])
    train_set_y = theano.shared(y[valid_size+test_size:])
    
    dataset = [(train_set_x, train_set_y),
               (valid_set_x, valid_set_y),
               (test_set_x, test_set_y)]
    
    return dataset

    

