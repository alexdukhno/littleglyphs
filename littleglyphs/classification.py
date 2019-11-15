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



def image_distance_euclidean(a,b):
    assert a.shape == b.shape, 'shapes of images are not equal'
    assert len(a.shape) == 2, 'images have to be 2d grayscale'
    imgsize_x = a.shape[0]
    imgsize_y = a.shape[1]
    return ( np.linalg.norm(a-b) / (imgsize_x*imgsize_y) )



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



def seen_based_probabilities(seen):
    # Utility for selecting items from an array based on how many times they were seen.
    # Calculates a normalized probability array weighted towards less-seen elements.
    # Input: a 1D numpy array of how many times the elements have been seen.
    p = np.sum(seen+1) / (seen+1)
    p = p / np.sum(p)
    return p



def select_with_seen_based_probabilities(elems,seen,quantity=1,yield_indices=True):
    # Utility for selecting items from an array based on how many times they were seen.
    # Randomly selects elements from 'elems', weighted towards less seen elements.
    # Returns a list of elements, or a numpy array of their indices if 'yield_indices' is true.
    indices = np.random.choice(np.arange(len(elems)),size=quantity,p=seen_based_probabilities(seen))
    
    if quantity == 1:
        indices = [indices]
    if not yield_indices:
        res = []
    for index in indices:
        seen[index] += 1
        if not yield_indices:
            res.append(elems[index])
    if yield_indices:
        res = indices
    return res



def find_lowest_cross_similarity(costmatrix,N_elems_to_select,penalize_neighbors=0):
    '''
    This function finds an answer to the question:
    "In a matrix of cross-similarity of classes, 
      find the set of N classes with the lowest total of their cross-similarities".
    
    This function finds a solution to this problem by treating it as a 
      quadratic program for the maximum edge weight clique problem:
    see https://stats.stackexchange.com/questions/110426/least-correlated-subset-of-random-variables-from-a-correlation-matrix
    
    It usually finds a good but not necessarily global minimum.
    
    penalize_neighbors parameter is a multiplier that controls 
      the size of consecutive mutually exclusive subsets of classes.
    E.g. for choosing 2 classes out of 9 with penalize_neighbors=3,
      [0,5], [0,8], [4,9], [2,7] could be valid answers and
      [0,1], [4,5], [6,7], [6,8] will be invalid, because
      elements 0,1 and 2 will exclude each other (and so will 3,4,5 and 6,7,8).
    '''
    N_elems = costmatrix.shape[0]
    costmatrix_w = np.copy(costmatrix)
    
    if (penalize_neighbors>1) and (N_elems % penalize_neighbors != 0):
        raise 'Neighbor penalization requires matrix to be divisible by "penalize_neighbors" parameter'    
    if penalize_neighbors>1:
        N_neighbors = penalize_neighbors
        for i in range(0,N_elems_to_select): 
            costmatrix_w[i*N_neighbors:(i+1)*N_neighbors,i*N_neighbors:(i+1)*N_neighbors] += N_elems
    
    def l1_sum_con(vec):
        return np.sum(vec) - N_elems_to_select
    def quadratic_costfunc(vec):
        return np.dot( np.dot(np.transpose(vec), costmatrix_w), vec)
        
    cons = [{'type':'eq', 'fun': l1_sum_con}]
    
    # v0 = np.random.random(N_elems)
    v0 = np.ones(N_elems)*0.5
    
    bounds = scipy.optimize.Bounds(*np.transpose(np.repeat([[0,1]],N_elems,axis=0)))
    res = scipy.optimize.minimize(quadratic_costfunc, v0, constraints=cons, bounds=bounds)
    #print(res.message)
    #print(res.x)
    #plt.plot(res.x)
    best_elem_indices = np.argpartition(res.x, -N_elems_to_select)[-N_elems_to_select:]
    return best_elem_indices



def find_lowest_cross_similarity_greedy(costmatrix,N_elems_to_select,penalize_neighbors=0):
    '''
    This function finds an answer to the question:
    "In a matrix of cross-similarity of classes, 
      find the set of N classes with the lowest total of their cross-similarities".
    
    This function finds a solution to this problem via a greedy heuristic:
    
    1) Find an element with the lowest cross-similarity sum. Use it as the "best" element.
    2) Iterate through all elements and find which element has lowest cross-similarity to the "best" element.
    3) Add this element to the list of "best" elements.
    4) Iterate through all elements and find which element has lowest cross-similarity sum to "best" elements; add it to the list.
    5) Repeat until we have the desired amount of "best" elements.
    It usually finds a good but not necessarily global minimum.
    It scales well for large matrices (complexity of approx. O(N*N_elems_to_select))
    
    penalize_neighbors parameter is a multiplier that controls 
      the size of consecutive mutually exclusive subsets of classes.
    E.g. for choosing 2 classes out of 9 with penalize_neighbors=3,
      [0,5], [0,8], [4,9], [2,7] could be valid answers and
      [0,1], [4,5], [6,7], [6,8] will be invalid, because
      elements 0,1 and 2 will exclude each other (and so will 3,4,5 and 6,7,8).
    '''
    
    N_elems = costmatrix.shape[0]
    costmatrix_w = np.copy(costmatrix)
    elem_cost_global = (np.sum(costmatrix_w, axis=0)-1)
    if (penalize_neighbors>1) and (N_elems % penalize_neighbors != 0):
        raise 'Neighbor penalization requires matrix to be divisible by "penalize_neighbors" parameter'    
    if penalize_neighbors>1:
        N_neighbors = penalize_neighbors
        for i in range(0,N_elems_to_select): 
            costmatrix_w[i*N_neighbors:(i+1)*N_neighbors,i*N_neighbors:(i+1)*N_neighbors] = 0
    
    # First element guess is "the one that overall has the lowest cross-corr sum"
    best_elem_indices = np.zeros(1, dtype=np.int)
    best_elem_indices[0] = np.argmin(elem_cost_global)

    is_elem_available = np.full((N_elems,),True)
    is_elem_available[best_elem_indices[0]] = False
    if penalize_neighbors>1:
        # Also remove the neighbors from heuristic
        min_index = best_elem_indices[0]//N_neighbors*N_neighbors
        max_index = best_elem_indices[0]//N_neighbors*N_neighbors + N_neighbors
        is_elem_available[min_index:max_index] = False
    
    for i in range(0,N_elems_to_select-1):
        test_elem_indices = is_elem_available.nonzero()[0]        
        elem_cost_local = np.zeros(N_elems)
        for j in np.nditer(test_elem_indices):
            elem_cost_local[j] = np.sum(costmatrix_w[j,best_elem_indices])
        elem_cost_local[best_elem_indices] = np.inf
        elem_cost_local[np.where(is_elem_available == False)[0]] = np.inf
        best_candidate_index = np.argmin(elem_cost_local)

        is_elem_available[best_candidate_index] = False
        if penalize_neighbors>1:
            # Also remove the neighbors from heuristic
            min_index = best_candidate_index//N_neighbors*N_neighbors
            max_index = best_candidate_index//N_neighbors*N_neighbors + N_neighbors
            is_elem_available[min_index:max_index] = False
        
        best_elem_indices = np.append(best_elem_indices,best_candidate_index)
    
    return best_elem_indices



# ------- Classification model: naive distance metric -------

class DistanceMetricClassifier:
    N_categories = None
    distance_function = None
    cat_distance_matrix = None    
    imgsize = None
    
    def __init__(self, N_categories, imgsize, distance_function):
        self.N_categories = N_categories
        self.distance_function = distance_function
        self.imgsize = imgsize
        self.avg_cat_representations = np.zeros(
            (self.N_categories,imgsize,imgsize)
        )
        self.cat_distance_matrix = np.zeros(
            (self.N_categories, self.N_categories)
        )    
    
    def train(self, X_train, Y_train):        
        N_examples = X_train.shape[0]
        avg_category_X = np.zeros(
            (self.N_categories,self.imgsize,self.imgsize)
        )
        Y_categories = np.argmax(Y_train, axis=1)        
        Y_category_counts = np.sum(Y_train, axis=0)
        
        # Build an average representation for each category
        for i in range(0,N_examples):
            avg_category_X[Y_categories[i]] += X_train[i]       
        for i in range(0,self.N_categories):
            avg_category_X[i] = avg_category_X[i] / Y_category_counts[i]
                            
        # Find distances between average representations for each pair of categories
        for i in range(0,self.N_categories):
            for j in range(0,self.N_categories):
                self.cat_distance_matrix[i,j] = self.distance_function(avg_category_X[i], avg_category_X[j])        
        self.avg_cat_representations = avg_category_X
                    
    def predict(self, X_test):
        # Output is softmax of inverse distances from the example to the avg representations for each category
        N_examples = X_test.shape[0]
        candidate_distances = np.zeros(self.N_categories)
        Y_predicted = np.zeros((N_examples, self.N_categories))
        for i in range(0,N_examples):
            for c in range(0,self.N_categories):
                candidate_distances[c] = self.distance_function(X_test[i], self.avg_cat_representations[c])            
            candidate_distances = np.power(candidate_distances, -1)
            candidate_distances = np.exp(candidate_distances) / np.sum(np.exp(candidate_distances), axis=0)
            Y_predicted[i] = candidate_distances
            
        return Y_predicted
    
    def evaluate(self, X_test, Y_test):
        N_examples = X_test.shape[0]
        Y_predicted = self.predict(X_test)
        Y_test_categories = np.argmax(Y_test,axis=1)
        Y_predicted_categories = np.argmax(Y_predicted,axis=1)
        accuracy = np.sum(Y_test_categories==Y_predicted_categories) / N_examples
        return accuracy
        
        


# ------- Classification model: CNN -------


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



# ------- Classification model: Siamese CNN -------

def prep_data_for_SiameseCNN_model(A,imgsize):
    # Keras implementation of 2d convolutional layers requires that the image has channels, 
    # even if there's a single channel.
    return A.reshape(A.shape[0],2,1,imgsize,imgsize)



def select_random_data_pair_indices_for_SiameseCNN(
    data,      # Array containing data (has to support yielding category indices, like RasterArray)
    data_seen, # Array containing information about how many times each individual data element was already selected
    N_classes, # How many classes are there
    N_pairs_to_select = 600 # How many pairs in total to select (approx.)
):
    # Utility for selecting indices of random data pairs
    # Yields approx. 1:1 ratio of "identical" and "different" data pairs
    
    # Generate "random" pairs
    # Most random pairs will be different
    # Some random pairs can already be identical, so for 1:1 balance we will need a bit more "random" 
    # and a bit less "identical" pairs
    
    N_random_pairs_in_batch = int(
        (N_pairs_to_select / 2) * (1 + 1/(N_classes-1))
    )

    selection_indices_A = select_with_seen_based_probabilities(
        data,data_seen,
        quantity=N_random_pairs_in_batch*2,yield_indices=True
    )
    selection_indices_B = selection_indices_A[N_random_pairs_in_batch:]
    selection_indices_A = selection_indices_A[:N_random_pairs_in_batch]

    # Generate "identical" pairs for each class.     
    N_identical_pairs_in_batch = int(
        (N_pairs_to_select - N_random_pairs_in_batch)
    )
    for category in range(0,N_classes):
        curr_category_pair_quantity = N_identical_pairs_in_batch//N_classes
        curr_category_indices = data.category_indices[category]
        curr_category_indices = np.array(curr_category_indices)
        selected_curr_category_indices = select_with_seen_based_probabilities(
            data[curr_category_indices],data_seen[curr_category_indices],
            quantity=curr_category_pair_quantity*2,yield_indices=True
        )
        selection_indices_A_identical = curr_category_indices[selected_curr_category_indices]
        selection_indices_B_identical = selection_indices_A_identical[curr_category_pair_quantity:]
        selection_indices_A_identical = selection_indices_A_identical[:curr_category_pair_quantity]
        selection_indices_A = np.concatenate((selection_indices_A,selection_indices_A_identical),axis=0)
        selection_indices_B = np.concatenate((selection_indices_B,selection_indices_B_identical),axis=0)
    
    return [selection_indices_A,selection_indices_B]




def make_SiameseCNN_model(
    imgsize, N_classes, 
    complexity = 16, dense_complexity = 8,
    CNN_dropout_strength = 0.25,
    dense_dropout_strength = 0.01
):
    
    
    # Subroutine for making the CNN subnetwork
    def make_CNN_part(nameprefix = 'A',complexity = 16,dense_complexity = 8,dropout_strength = 0.25):
        CNNpart = keras.models.Sequential()
        CNNpart.add( keras.layers.Conv2D(
            1*complexity, 5, strides=(1, 1),
            padding='same', data_format="channels_first",
            activation='relu', 
            use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,
            name='CNN_conv1_'+nameprefix,
        ))

        CNNpart.add( keras.layers.MaxPooling2D(        
            pool_size=(2, 2), 
            strides=None, padding='valid', 
            data_format="channels_first",
            name='CNN_maxpool1_'+nameprefix,
        ))

        CNNpart.add( keras.layers.Dropout(
            CNN_dropout_strength,
            name='CNN_dropout1_'+nameprefix,
        ))

        CNNpart.add( keras.layers.Conv2D(
            2*complexity, 5, strides=(1, 1),
            padding='same', data_format="channels_first",
            activation='relu', 
            use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,
            name='CNN_conv2_'+nameprefix,
        ))

        CNNpart.add( keras.layers.MaxPooling2D(        
            pool_size=(2, 2), 
            strides=None, padding='valid', 
            data_format="channels_first",
            name='CNN_maxpool2_'+nameprefix,
        ))

        CNNpart.add( keras.layers.Dropout(
            CNN_dropout_strength,
            name='CNN_dropout2_'+nameprefix,
        )) 

        CNNpart.add( keras.layers.Flatten(
            name='CNN_finalflat_'+nameprefix,
        ))

        CNNpart.add( keras.layers.Dense(
            units=2*complexity*dense_complexity, 
            activation='relu', 
            kernel_initializer='glorot_uniform',
            name='CNN_dense_'+nameprefix,
        ))

        CNNpart.add( keras.layers.Dropout(
            CNN_dropout_strength,
            name='CNN_dense_dropout_'+nameprefix,
        )) 

        return CNNpart

    # Subroutine for making a Lambda layer which outputs absolute difference of the inputs.
    def absolute_tensor_difference(pair_of_tensors):
        x, y = pair_of_tensors
        return keras.backend.abs(x - y)
    
    
    # ---- Main routine ----
    
    if not isinstance(imgsize,tuple):
        imgsize_for_conv = (imgsize,imgsize)
    else:
        imgsize_for_conv = imgsize
    
    
    SCNN_inputA = keras.layers.Input(shape=(1,*imgsize_for_conv),name='inputA')
    SCNN_inputB = keras.layers.Input(shape=(1,*imgsize_for_conv),name='inputB')

    CNNpart = make_CNN_part(
        nameprefix = 'A',
        complexity = complexity,
        dense_complexity = dense_complexity,
        dropout_strength = CNN_dropout_strength
    )

    SCNN_conv_outputA = CNNpart(SCNN_inputA)
    SCNN_conv_outputB = CNNpart(SCNN_inputB)

    SCNN_abs_difference = keras.layers.Lambda( 
        absolute_tensor_difference,
        name='abs_difference',
    )([SCNN_conv_outputA,SCNN_conv_outputB])

    SCNN_finaldense = keras.layers.Dense(
        units=2*complexity*dense_complexity, 
        activation='sigmoid', 
        name='finaldense',
    )(SCNN_abs_difference)

    SCNN_finaldense_dropout = keras.layers.Dropout(
        dense_dropout_strength,
        name='finaldense_dropout'
    )(SCNN_finaldense) 

    SCNN_output = keras.layers.Dense(
        1, 
        activation='sigmoid',
        name='output'
    )(SCNN_finaldense_dropout)

    model = keras.models.Model(inputs=[SCNN_inputA,SCNN_inputB],outputs=[SCNN_output])

    # optimizer = keras.optimizers.RMSprop(0.01)
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model



class SiameseCNN_classifier():
    
    model = None
    support_raster_array = None
    similarity_matrix = None
    
    def __init__(self, model, support_raster_array): 
        self.model = model
        self.support_raster_array = support_raster_array

    def classify(self, test_image, subsample_size, subsample_attempts=1):        
        imgsize = self.support_raster_array.imgsize
        test_image_class_similarities = np.zeros(self.support_raster_array.N_different_categories)
        for i in range(0,subsample_attempts):
            # Make a representative subset of the support data array
            representative_rasters = self.support_raster_array.representative_subsample(subsample_size)
            # Make an array of image pairs: 
            #  first image is always the test image, 
            #  second image is each of the class-representative images
            test_image_array = np.repeat(test_image[np.newaxis,...], len(representative_rasters), axis=0)
            X_predict = np.array([test_image_array,representative_rasters.rasters])
            X_predict = X_predict.transpose(1,0,2,3)
            X_predict_conv = prep_data_for_SiameseCNN_model(X_predict,imgsize)

            # Feed the array to the model to obtain similarities to each of the class-representative images
            Y_predicted = self.model.predict(
                {'inputA': X_predict_conv[:,0], 'inputB': X_predict_conv[:,1]},
            )

            test_image_pair_similarities = np.array(
                [representative_rasters.categories,Y_predicted.flatten()]
            )
            test_image_pair_similarities = test_image_pair_similarities.transpose()
            #print(test_image_pair_probabilities)

            test_image_class_similarities += np.mean(
                Y_predicted.flatten().reshape(-1, subsample_size), 
                axis=1
            )
        
        return test_image_class_similarities / subsample_attempts
    
    
    def similarity_matrix(self, subsample_size, protect_against_oom=True, oom_protection_limit=100000000):
        
        imgsize = self.support_raster_array.imgsize        
        support_raster_indices_A = []
        support_raster_indices_B = []
        
        # At the moment the pairs are generated and stored in-memory, which can quickly get out of hand
        # with a large number of categories. Following is a hacky precaution to ensure IPython won't choke
        if protect_against_oom:
            memory_estimate = np.square(self.support_raster_array.N_different_categories)
            memory_estimate = memory_estimate * 2
            memory_estimate = memory_estimate * imgsize * imgsize
            if memory_estimate > oom_protection_limit:
                raise "Similarity matrix calculation liable to run out of memory; reduce subsample size"
        
        for category_A in range(0,self.support_raster_array.N_different_categories):
            for category_B in range(0,self.support_raster_array.N_different_categories):
                curr_category_A_indices = self.support_raster_array.category_indices[category_A]
                curr_category_A_indices = np.array(curr_category_A_indices)
                chosen_indices_A = np.random.choice(curr_category_A_indices, subsample_size, replace=False)
                curr_category_B_indices = self.support_raster_array.category_indices[category_B]
                curr_category_B_indices = np.array(curr_category_B_indices)
                chosen_indices_B = np.random.choice(curr_category_B_indices, subsample_size, replace=False)
                for index_A in chosen_indices_A:
                    for index_B in chosen_indices_B:
                        support_raster_indices_A.append(index_A)
                        support_raster_indices_B.append(index_B)

        support_raster_indices_A = np.array(support_raster_indices_A)
        support_raster_indices_B = np.array(support_raster_indices_B)
        
        X_predict = np.array(
            [self.support_raster_array[support_raster_indices_A],
             self.support_raster_array[support_raster_indices_B]]
        )
        print(X_predict.shape)
        X_predict = X_predict.transpose(1,0,2,3)
        X_predict_conv = prep_data_for_SiameseCNN_model(X_predict,imgsize)

        Y_predicted = self.model.predict(
            {'inputA': X_predict_conv[:,0], 'inputB': X_predict_conv[:,1]},
        )
        Y_predicted = Y_predicted.flatten()
        
        similarity_matrix = np.copy(Y_predicted)
        # NB: check if reshaping this way is correct
        similarity_matrix = similarity_matrix.reshape(
            self.support_raster_array.N_different_categories,
            self.support_raster_array.N_different_categories,
            subsample_size,subsample_size
        )
        similarity_matrix = similarity_matrix.mean((2,3))
        self.similarity_matrix = similarity_matrix
        return [similarity_matrix, Y_predicted]