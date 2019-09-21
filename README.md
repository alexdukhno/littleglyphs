# littleglyphs
A Python library containing a set of useful classes and routines for investigating the performance of image classifiers 
on procedurally generated letter-like images. 

Built around:
* _numpy_ for calculations, 
* _scikit-image_ for rendering, 
* _Keras_ (with Tensorflow backend) for classification with neural networks,
* _scikit-learn_ for classification with a bunch of other methods.

## Why?
The original idea for this library came from asking a simple question: 

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

## What are the applications?
An obvious utility would be to construct a [wear-resistant](https://i.imgur.com/lcrC9VB.png), easily machine-readable 
and human-readable font.

This approach can be generalized to gauge performance of image classifiers against each other and uncover their 
issues and points of failure using simple examples. For instance, there is an issue with convolutional neural networks for image classification
[being translationally-invariant](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b), resulting in high perceived similarity for images with some of the parts swapped around.

Also, figuring stuff out is fun (and like with any hobby project, it yields experience along the way).

