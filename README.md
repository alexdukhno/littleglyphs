# littleglyphs
A Python library containing a set of useful classes and routines for investigating the performance of image classifiers 
on procedurally generated letter-like images. In a lot of aspects it is inspired by the Omniglot dataset (see [Lake et al. 2011][1] for details) and a plethora of work dedicated to it (for a relatively recent review see [Lake et al. 2019][2]).

Built around:
* _numpy_ and _scipy_ for calculations, 
* _scikit-image_ for rendering, 
* _Keras_ (with Tensorflow backend) for classification with neural networks,
* _scikit-learn_ for classification with other methods,
* _matplotlib_ for visualization.

## Features
* Glyph generation: generate randomized alphabets of glyphs (vector symbol representations) with a set of predetermined features: lines, bezier curves, circles, etc.

![example of glyph images](https://github.com/alexdukhno/littleglyphs/blob/master/images/glyphs_example.png)

* Raster generation and data augmentation: from the glyph as a vector representation, produce raster images to be used in recognition tasks. Distort the glyphs and raster images to introduce more variety into the data.

![example of glyph images](https://github.com/alexdukhno/littleglyphs/blob/master/images/rasters_example.png)

* Basic classification methods: classification via CNN, Siamese networks with a CNN as the feature extractor, etc.
* Visualization: one-line utilities for quick examination of glyphs, glyph alphabets, raster arrays, etc.

## Quick start
For a system equipped with a conda environment:
```
git clone https://github.com/alexdukhno/littleglyphs
conda env create -f littleglyphs/conda_env.yml
conda activate littleglyphs-cpu
```

## What are the applications?
An obvious utility would be to construct more [wear-resistant](https://i.imgur.com/lcrC9VB.png), more easily machine-readable 
and/or human-readable fonts.

This approach can be generalized to gauge performance of image classifiers against each other and uncover their 
issues and points of failure using simple examples. For instance, there is an issue with convolutional neural networks for image classification
[being translationally-invariant](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b), resulting in high perceived similarity for images with some of the parts swapped around.

The library can also be useful as a toolkit for introducing students to machine learning.

## Why?
This originated as a hobby project to tinker with machine learning.

The original idea for this library in particular came from asking a simple question: 

_"What would a very unambiguous alphabet look like?"_

or, more precisely,

_"How to construct a set of glyphs that look as different from each other as possible (preferably containing not too many elements)?"_

For example, in Latin script uppercase I and lowercase l usually look almost identical, 
sans-serif Hebrew has very similar-looking Het ח and Tav ת,
while Katakana, one of two Japanese syllabaries, contains very similar N ン and So ソ or Shi シ and Tsu ツ.
Such ambiguity can often result in confusion for both human and machine readers (especially untrained ones).

To answer the question, there would be a need for: 
* a routine to construct glyphs (_de novo_ or based on an existing image);
* a metric of similarity between glyphs - e.g., a performance metric of a classifier trained to distinguish images as pertaining
to one glyph or another;
* an algorithm to find a set of glyphs with minimized similarity.

_littleglyphs_ provides a set of tools for attempting to answer such kind of a question.

[1]: https://scholar.google.com/scholar?cluster=4819082874281835428&hl=en&as_sdt=0,5
[2]: https://arxiv.org/abs/1902.03477
