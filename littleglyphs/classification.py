import numpy as np

import tensorflow as tf
import keras
import sklearn 
from sklearn.model_selection import train_test_split




# ------- Classification utilities -------

def split_data_for_learning(X, Y, random_seed, crossval_proportion = 0.2, test_proportion = 0.2):
    # Splits the dataset into training, cross-validation, and test sets.
    # E.g. with crossval_proportion = 0.2 and test_proportion = 0.2
    #   the function will split the dataset into
    #   60% training, 20% cross-val and 20% test set.    
    test_proportion_secondsplit = (test_proportion/(1-crossval_proportion))

    X_traintest, X_cv, Y_traintest, Y_cv = train_test_split(
        X, Y, 
        test_size=crossval_proportion, 
        random_state=random_seed
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_traintest, Y_traintest, 
        test_size=test_proportion_secondsplit, 
        random_state=random_seed
    )

    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)



def prep_data_for_CNN_model(A,imgsize):
    # Keras implementation of 2d convolutional layers requires that the image has channels, 
    # even if there's a single channel.
    return A.reshape(A.shape[0],1,imgsize,imgsize)



def make_CNN_model(imgsize, N_classes, complexity = 20):
    if not isinstance(imgsize,tuple):
        imgsize_for_conv = (imgsize,imgsize)
    else:
        imgsize_for_conv = imgsize
        
    model = keras.models.Sequential()
    # First layer: (imgsize x imgsize) inputs fed into a 2D convolutional layer:
    # * with 20 filters, kernel size of 5x5, stride of 1 across the image in both X and y directions
    # * using zero-padding of the sides of the image so that the output is __same__ size as input
    # * with linear rectifier activation: vanilla matrix convolution, after which we set any negative outputs to zero
    # * using bias for filter activations
    # * using regular "recommended" way to initialize weights and biases
    # * using no regularization for learning weights and bias 
    # * using no regularization of outputs
    # * using no constraints on kernel or bias
    # The output will be (20 x imgsize x imgsize) - 20 filters, 32x32 map of activations of each one    
    model.add(
        keras.layers.Conv2D(
            1*complexity, 5, strides=(1, 1),
            input_shape=(1,*imgsize_for_conv),
            padding='same', data_format="channels_first",
            activation='relu', 
            use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None
        )
    )
    
    # Second layer: max-pool the outputs of the first layer
    # The output will be 20 x imgsize/2 x imgsize/2 - 20 filters, 
    #    imgsize/2 x imgsize/2  map of biggest activations of each one
    # To prevent overfitting, we randomly drop out 25% of the nodes (their output becomes 0)
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2), 
            strides=None, padding='valid', 
            data_format="channels_first"
        )
    )
    model.add(keras.layers.Dropout(0.25))
    
    # Third layer: also a convolution layer but with more filters: we expect more higher-order features
    # The output will be 40 x imgsize/2 x imgsize/2
    # Use a smaller 3x3 kernel
    model.add(
        keras.layers.Conv2D(
            2*complexity, 3, strides=(1, 1), 
            padding='same', data_format="channels_first",
            activation='relu', 
            use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None
        )
    )    

    # Fourth layer: max-pool the outputs of the third layer
    # The output will be 40 x imgsize/4 x imgsize/4 - 40 filters, 
    #    imgsize/4 x imgsize/4  map of biggest activations of each one
    # To prevent overfitting, we randomly drop out 25% of the nodes (their output becomes 0)
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2), 
            strides=None, 
            padding='valid', 
            data_format="channels_first"
        )
    )    
    model.add(keras.layers.Dropout(0.25))
    
    # Unroll the 40 x imgsize/4 x imgsize/4 outputs to feed to the regular NN layer (like in perceptrons)
    model.add(keras.layers.Flatten())
    
    # Fifth layer: regular fully-connected NN layer (like in perceptrons)
    # To prevent overfitting, we randomly drop out 25% of the nodes (their output becomes 0)
    model.add(
        keras.layers.Dense(
            units=2*complexity*N_classes, 
            activation='relu', 
            input_dim=N_classes
        )
    )
    model.add(keras.layers.Dropout(0.25))
    
    # Sixth and final layer: output regular NN layer with the same number of outputs as categories
    # Use softmax output so that all categories have value from 0 to 1 and neatly sum up to 1
    # Output corresponds to "probabilities" in each class
    model.add(
        keras.layers.Dense(
            units=N_classes, 
            activation='softmax', 
            input_dim=N_classes
        )
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model



def prob_conf_ent_matrix(Y_test,Y_predicted,N_classes):
    # Calculate probabilistic confusion entropy matrix. 
    # The matrix is basically like a regular confusion matrix,
    #   but instead of treating the highest output of a classifier as the output class,
    #   it treats outputs of classifier as "surety" of the classifier in each class.
    # This allows to see the cases where the classifier works, 
    #   but is not very sure in its answers.
    #
    # Based on doi:10.3390/e15114969
    #
    # Each row corresponds to the "true" class.
    # Each column corresponds to the "surety" of the classifier
    #   in its output for the column.
    # So, for instance, an element [4,2] corresponds to 
    #   the degree of surety with which the classifier says "it belongs to class 2"
    #   when it sees an element that in reality belongs to class 4.
    # Similarly to the case of a regular confusion matrix,
    #   a good classifier will have high values of diagonal elements
    #   and low values of all other elements.
    
    Y_test_class = np.argmax(Y_test, axis=1)
    classfreqs = np.bincount(Y_test_class)
    prob_conf_ent_matrix = np.zeros((N_classes,N_classes))
    for i in range(0,Y_predicted.shape[0]):
        prob_conf_ent_matrix[Y_test_class[i]] += Y_predicted[i]
    prob_conf_ent_matrix = prob_conf_ent_matrix / classfreqs

    return prob_conf_ent_matrix

