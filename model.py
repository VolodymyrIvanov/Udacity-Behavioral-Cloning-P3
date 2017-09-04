'''
Created on 11.08.2017

@author: VIvanov
'''
import csv
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D, Dropout, Lambda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_path', 'data/', "Data Path")
flags.DEFINE_integer('epochs', 10, "Number of epochs")
flags.DEFINE_integer('batch', 12, "Batch size")

# Change file name from given row in CSV file to real location on GPU Linux machine
def correct_file_name(image_path, path):
    delim = path.rfind('/')
    if delim >= 0:
        file_name = path[delim + 1:]
    else:
        delim = path.rfind('\\')
        if delim >= 0:
            file_name = path[delim + 1:]
        else:
            file_name = path
    file_name = image_path + file_name
    return file_name

# Generator
def generator(samples, image_path, batch_size=32):
    num_samples = len(samples)
    # Angle correction for side images
    correction = 0.15
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                # Center image
                name = correct_file_name(image_path, batch_sample[0])
                image = np.asarray(Image.open(name))
                images.append(image)
                angle = center_angle
                angles.append(angle)
                # Left image
                name = correct_file_name(image_path, batch_sample[1])
                image = np.asarray(Image.open(name))
                images.append(image)
                angle = center_angle + correction
                angles.append(angle)
                # Right image
                name = correct_file_name(image_path, batch_sample[2])
                image = np.asarray(Image.open(name))
                images.append(image)
                angle = center_angle - correction
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def main(_):
    
    print("Data Path", FLAGS.data_path)
    print("Number of epochs", FLAGS.epochs)
    print("Batch size", FLAGS.batch)
    
    data_path = FLAGS.data_path 
    image_path = data_path + 'IMG/'
    log_file = 'driving_log.csv'

    # Loading data
    samples = []
    with open(data_path + log_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        for row in reader:
            samples.append(row)
 
    # Split data into train and validation subsets
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # Multiplying by 3 because of usage left/right images
    train_samples_size = 3 * len(train_samples)
    validation_samples_size = 3 * len(validation_samples)
    
    # Generate data sets
    train = generator(train_samples, image_path, FLAGS.batch)
    validation = generator(validation_samples, image_path, FLAGS.batch)

    # Model
    model = Sequential()
    # Preprocessing - normalization
    model.add(Lambda(lambda x: (x / 255.0) - .5, input_shape=(160, 320, 3)))
    # Cropping not usable area like sky, trees, etc.
    model.add(Cropping2D(cropping=((75, 10), (0,0)), input_shape=(160, 320, 3)))
    # NVIDIA CNN, recommended in lesson
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
    # Possible Dropout to avoid over fitting
    #model.add(Dropout(rate=0.75))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
    # Possible Dropout to avoid over fitting
    #model.add(Dropout(rate=0.5))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
    # Possible Dropout to avoid over fitting
    model.add(Dropout(rate=0.5))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    # Possible Dropout to avoid over fitting
    #model.add(Dropout(rate=0.75))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    # Possible Dropout to avoid over fitting
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Used default Adam Optimizer
    model.compile(loss = 'mse', optimizer = 'adam')
    # Evaluation of steps since the interface was changed in Keras 2.0.6
    steps = int(train_samples_size / FLAGS.batch)
    # Training...
    model.fit_generator(train, steps_per_epoch = steps, 
                    validation_data = validation, validation_steps = validation_samples_size,
                    epochs = FLAGS.epochs)
    #Storing the model in a file
    model.save(data_path + 'model.h5')
    #Plot model architecture
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
