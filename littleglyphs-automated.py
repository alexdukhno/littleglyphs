#!/usr/bin/env python
# coding: utf-8

# Import the project library for glyph generation, classification, plotting, etc.
import littleglyphs as lilg
import littleglyphs.plotting as lilgplt
import littleglyphs.classification as lilgcls
import littleglyphs.examples as lilgex

# Import prerequisite libraries.
import copy
import time

import numpy as np
import scipy
import skimage
import sklearn 
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

random_seed = 456

np.random.seed(random_seed)

from datetime import datetime
import os
output_path_prefix = './output/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
if not os.path.exists(output_path_prefix):
    os.makedirs(output_path_prefix)
output_path_prefix = output_path_prefix + '/'

# Generate a glyph _alphabet_ (an array of glyphs each belonging to a unique _category_). From it, making a set of glyphs with slightly different strokes (glyph _permutations_), produce images, then use a set of transformations (_distortions_) on those images to produce glyph _variations_.

N_glyphs_in_alphabet = 10

N_bezier_features = 0
N_line_features = 0
N_ellipse_features = 0
N_multipoint_line_features = 0
multipoint_line_feature_N_points = 3
N_multipoint_bezier_features = 1
multipoint_bezier_feature_N_points = 3

N_glyph_permutations = 20
permutation_strength = 0.05

imgsize = 16

N_glyph_raster_distortions = 30
rotat_distort_max = np.pi / 8
shear_distort_max = np.pi / 8
scale_distort_max = 0.25

blur_factor = 1

N_variations_per_glyph = N_glyph_permutations*N_glyph_raster_distortions


time_start = time.time()
print('Generating glyph alphabet and glyph variations... ', end='')

def makeRandomGlyph(category):
    glyph = lilg.Glyph(
        [lilg.FeatureBezierCurve() for count in range(0,N_bezier_features)]+
        [lilg.FeatureLineSegment() for count in range(0,N_line_features)]+
        [lilg.FeatureEllipse() for count in range(0,N_ellipse_features)]+
        [lilg.FeatureMultiPointLineSegment(multipoint_line_feature_N_points) for count in range(0,N_multipoint_line_features)]+
        [lilg.FeatureMultiPointBezierCurve(multipoint_bezier_feature_N_points) for count in range(0,N_multipoint_bezier_features)]
    )
    glyph.set_category(category)
    glyph.randomize_all_features()
    return glyph

glyphs = []
glyph_categories = list(range(0,N_glyphs_in_alphabet))

#for category in glyph_categories:
#    glyph = makeRandomGlyph(category)
#    glyphs.append(glyph)
    
starter_glyph = makeRandomGlyph(0)    
for category in glyph_categories:
    glyph = starter_glyph.permuted(permutation_strength)
    glyph.set_category(category)
    glyphs.append(glyph)

glyph_alphabet = lilg.GlyphList(glyphs)

#glyph_alphabet = lilg.examples.MNISTlike_glyph_alphabet()
#N_glyphs_in_alphabet = len(glyph_alphabet)

glyph_permuted_alphabet = glyph_alphabet.permuted(permutation_strength, N_glyph_permutations)

glyph_rasters = glyph_permuted_alphabet.render(
    (imgsize,imgsize), 
    blur_factor=blur_factor,randomize_blur=True,random_blur_extent=2
)
distorter = lilg.SequentialDistorter(
    [
        lilg.DistortionRandomAffine(
            rotat_distort_max = rotat_distort_max, 
            shear_distort_max = shear_distort_max,
            scale_distort_max = scale_distort_max
        )
    ]
)
glyph_rasters = glyph_rasters.distorted(distorter, N_glyph_raster_distortions)


time_end = time.time()
print('done in '+'{0:.3f}'.format(time_end-time_start)+' sec '+
     '('+'{0:.3f}'.format((time_end-time_start)/N_glyphs_in_alphabet)+' sec per glyph).')


# Visualise the glyphs and show some examples of glyph rasters.

print('Ground truth glyphs:')
fig, axs = lilgplt.visualize_glyph_list(
    glyph_alphabet,
    N_glyphs_to_show = N_glyphs_in_alphabet, 
    imgsize=128, 
    blur_factor=0.5*16,
    figsize=(12,6)
)
fig.savefig(output_path_prefix+'ground_truth.png',bbox_inches='tight')


# --- Start the loop for improving the glyph alphabet ---
glyph_alphabet_improval_iter = 0
glyph_alphabet_improval_maxiter = 100

while glyph_alphabet_improval_iter < glyph_alphabet_improval_maxiter:

    print()
    print()
    print('-------- Alphabet improval, iteration '+str(glyph_alphabet_improval_iter)+' --------')
    print()
    
    # ### Categorize the data using a CNN

    # Make a one-hot encoded category correspondence array for the glyph variations.

    category_classes, inverse_category_class_indices = np.unique(glyph_rasters.categories, return_inverse=True)
    N_classes = len(category_classes)

    X = glyph_rasters.rasters
    Y = keras.utils.to_categorical(glyph_rasters.categories, num_classes=N_classes)

    # We have our data ready to be fed to the categorizer. Split it into training, cross-validation, and test sets.

    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = lilgcls.split_data_for_learning(
        X, Y, 
        crossval_proportion = 0.2, 
        test_proportion = 0.2, 
        random_seed=random_seed
    )

    print("X_train matrix shape: "+str(X_train.shape)+"; Y_train matrix shape: "+str(Y_train.shape))
    print("X_test  matrix shape: "+str(X_test.shape )+"; Y_test  matrix shape: "+str(Y_test.shape ))
    print("X_cv    matrix shape: "+str(X_cv.shape   )+"; Y_cv    matrix shape: "+str(Y_cv.shape   ))


    # Prepare a CNN model.

    keras.backend.clear_session()
    model = lilgcls.make_CNN_model(imgsize, N_classes, complexity = 4)


    # We have our model. Transform the data into a format that can be fed to it, and start training.

    X_train_conv = lilgcls.prep_data_for_CNN_model(X_train, imgsize)
    X_cv_conv = lilgcls.prep_data_for_CNN_model(X_cv, imgsize)

    N_epochs = 10

    #from keras.callbacks import ModelCheckpoint
    #checkpoint_filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    #checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    h = model.fit(
        X_train_conv, Y_train, 
        epochs=N_epochs, batch_size=N_classes*10, 
        verbose=2,
        #callbacks=callbacks_list, 
        validation_data=(X_cv_conv,Y_cv)
    )

    #from IPython.display import Audio
    #Audio('./bell.ogg',autoplay=True)


    # Show the evolution of training/cross-validation losses and accuracies.

    accurs = h.history['acc']
    val_accurs = h.history['val_acc']
    losses = h.history['loss']
    val_losses = h.history['val_loss']
    epoch_numbers = h.epoch

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(epoch_numbers, losses, label='training loss per epoch')
    plt.plot(epoch_numbers, val_losses, label='cross-val. loss per epoch')
    plt.plot(epoch_numbers, accurs, label='training accuracy per epoch')
    plt.plot(epoch_numbers, val_accurs, label='cross-val. accuracy per epoch')
    ax.set_ylim([0,None])
    ax.legend()
    
    fig.savefig(
        output_path_prefix+'classifier_iter'+str(glyph_alphabet_improval_iter)+'.png',
        bbox_inches='tight'
    )

    X_test_conv = lilgcls.prep_data_for_CNN_model(X_test,imgsize)
    loss_and_metrics = model.evaluate(X_test_conv, Y_test, batch_size=128)
    print('Loss on test set: ~'+'{0:.2f}'.format(loss_and_metrics[0]))
    print('Accuracy on test set: ~'+'{0:.0f}'.format(loss_and_metrics[1]*100)+'%')


    # Evaluate the model's predictions for the test data.

    Y_predicted = model.predict(X_test_conv, batch_size=128)
    Y_predicted_class = np.argmax(Y_predicted, axis=1)
    Y_predicted_probability = np.max(Y_predicted, axis=1)
    Y_test_class = np.argmax(Y_test, axis=1)


    # ### Change the ambiguous glyphs

    # Calculate probabilistic confusion entropy matrix. 
    # 
    # The matrix is basically like a regular confusion matrix, but instead of discretely treating the highest output of the classifier as the output class, it smoothly treats all outputs of the classifier as "surety" of the classifier in each class. This uncovers information about the cases where the classifier works, but is not very sure in its answers.
    # 
    # Each row corresponds to the "true" class. Each column corresponds to the "surety" of the classifier in its output for the column. So, for instance, an element [4,2] corresponds to the average degree of surety with which the classifier says "it belongs to class 2" when it sees an element that in reality belongs to class 4.
    # Similarly to the case of a regular confusion matrix, a good classifier will have high values of diagonal elements and low values of all other elements.
    # 
    # ( Based on doi:10.3390/e15114969 )
    # 

    prob_conf_ent_matrix = lilgcls.prob_conf_ent_matrix(Y_test,Y_predicted,N_classes)

    worst_class_index = np.diagonal(prob_conf_ent_matrix).argmin()
    best_class_index =  np.diagonal(prob_conf_ent_matrix).argmax()

    print('Probabilistic confusion entropy matrix:')
    #print(np.around(prob_conf_ent_matrix,decimals=3))
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    plt.imshow(prob_conf_ent_matrix, vmin=0, vmax=1, cmap='hot')
    plt.colorbar()
    fig.savefig(
        output_path_prefix+'probmatrix_iter'+str(glyph_alphabet_improval_iter)+'.png',
        bbox_inches='tight'
    )
    print('Class with worst performance: '+str(worst_class_index))
    print('Class with best performance:  '+str(best_class_index))


    # Show the worst confusion case.




    prob_conf_ent_matrix_nodiag = prob_conf_ent_matrix.copy()
    np.fill_diagonal(prob_conf_ent_matrix_nodiag,0)

    worst_confusion_index = np.unravel_index(
        np.argmax(prob_conf_ent_matrix_nodiag, axis=None), 
        prob_conf_ent_matrix_nodiag.shape
    )
    print('Most confused pair: class '
          +str(worst_confusion_index[0])
          +' is being mistaken for class '
          +str(worst_confusion_index[1])
          +' with probability '
          +str(np.around(prob_conf_ent_matrix_nodiag[worst_confusion_index],decimals=3))
         )


    print('Actual distribution of classes for test data:')
    print(np.bincount(Y_test_class))
    print('Distribution of classes for test data as predicted by the classifier:')
    print(np.bincount(Y_predicted_class))

    # Randomly generate glyphs until spotting a glyph that has a lot of _ambiguity for the current classifier_. An _ambiguous_ glyph is defined as one with which the classifier has trouble assigning it to any of the classes.

    best_ambiguity = N_classes # an impossibly high value for ambiguity distance
    ideal_ambiguity_probability = np.ones(N_classes) * (1/N_classes)
    worst_glyph_ambiguity = np.sum((prob_conf_ent_matrix[worst_class_index] - ideal_ambiguity_probability)**2)

    new_candidate_glyph_ambiguity_threshold = worst_glyph_ambiguity / 10
    new_candidate_glyph_ambiguity_maxiter = 20

    print(
        'Worst glyph\'s distance from ideal ambiguity: '+
        str(np.around(worst_glyph_ambiguity, decimals=3))
    )

    print(
        'Generating new glyph to replace the glyph with worst performance (target ambiguity distance: '+
        str(np.around(new_candidate_glyph_ambiguity_threshold,decimals=3))+
        '):'
    )

    i = 0
    while (
        (i < new_candidate_glyph_ambiguity_maxiter) and 
        (best_ambiguity > new_candidate_glyph_ambiguity_threshold)
    ):
        new_candidate_glyph = makeRandomGlyph(worst_class_index)

        new_glyph_perm_list = lilg.GlyphList([new_candidate_glyph])
        new_glyph_perm_list = new_glyph_perm_list.permuted(permutation_strength, N_glyph_permutations)

        new_glyph_rasters = new_glyph_perm_list.render(
            (imgsize,imgsize), 
            blur_factor=blur_factor,randomize_blur=True,random_blur_extent=2
        )

        new_glyph_rasters = new_glyph_rasters.distorted(distorter, N_glyph_raster_distortions)

        X_new = new_glyph_rasters.rasters
        X_new_conv = lilgcls.prep_data_for_CNN_model(X_new,imgsize)
        Y_new_predicted = model.predict(X_new_conv, batch_size=128)

        Y_new_mean_class_probability = Y_new_predicted.mean(axis=0)
        #print('Candidate for new glyph - average classification:')
        #print(np.around(Y_new_mean_class_probability, decimals=3))
        #print('Candidate for new glyph - square distance from ideal ambiguity:')    
        distance_from_ideal_ambiguity = (Y_new_mean_class_probability - ideal_ambiguity_probability)**2
        #print(np.around(distance_from_ideal_ambiguity, decimals=3))
        total_distance_from_ideal_ambiguity = np.sum(distance_from_ideal_ambiguity)
        #print('Total distance from ideal ambiguity: '+str(np.around(total_distance_from_ideal_ambiguity, decimals=3)))
        #print()

        if best_ambiguity>total_distance_from_ideal_ambiguity:
            best_ambiguity = total_distance_from_ideal_ambiguity
            best_new_candidate_glyph = copy.deepcopy(new_candidate_glyph)
            best_new_glyph_rasters = copy.deepcopy(new_glyph_rasters)

        print(
            '\rIteration ' + str(i) + ': '+
            'current distance from ideal ambiguity: '+
            str(np.around(total_distance_from_ideal_ambiguity, decimals=3))+
            '; best: '+
            str(np.around(best_ambiguity, decimals=3)),
            end=''
        )

        i = i + 1


    new_candidate_glyph = best_new_candidate_glyph
    new_glyph_rasters = best_new_glyph_rasters
    print()
    print(
        'Best candidate distance from ideal ambiguity: '+str(np.around(best_ambiguity, decimals=3))+
        ', found in '+str(i)+' iterations (max '+str(new_candidate_glyph_ambiguity_maxiter)+' iterations).'
    )        

    # Incorporate the new glyph in place of the old one. Generate a new set of rasters.
    glyph_alphabet.remove_glyph_category(worst_class_index)
    glyph_alphabet.add_glyph(new_candidate_glyph)
    glyph_alphabet.sort_by_category()

    fig, axs = lilgplt.visualize_glyph_list(
        glyph_alphabet,
        N_glyphs_to_show = N_glyphs_in_alphabet, 
        imgsize=128, 
        blur_factor=0.5*16,
        figsize=(12,6)
    )
    fig.savefig(
        output_path_prefix+'alphabet_iter'+str(glyph_alphabet_improval_iter)+'.png',
        bbox_inches='tight'
    )
    
    time_start = time.time()
    print('Generating new glyph variations... ', end='')

    glyph_permuted_alphabet = glyph_alphabet.permuted(permutation_strength, N_glyph_permutations)

    glyph_rasters = glyph_permuted_alphabet.render(
        (imgsize,imgsize), 
        blur_factor=blur_factor,randomize_blur=True,random_blur_extent=2
    )
    glyph_rasters = glyph_rasters.distorted(distorter, N_glyph_raster_distortions)

    time_end = time.time()
    print('done in '+'{0:.3f}'.format(time_end-time_start)+' sec '+
         '('+'{0:.3f}'.format((time_end-time_start)/N_glyphs_in_alphabet)+' sec per glyph).')

    
    
    plt.close('all')
    glyph_alphabet_improval_iter += 1
    
