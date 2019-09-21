## Todo

* integrate some of the random generators into a separate submodule
* currently, for an alphabet with identical symbols and with maximization of ambiguity, the ideal candidate would be the same identical symbol
  * instead of random candidates, try candidates with mutation of a couple parameters at a time?
  * instead of maximization of ambiguity, use multi-hot encoding with outputs for "definitely not a class 1"? would it be just an inversion of the output, or would it yield additional interesting info?
  * or use a parallel network for deciding what the output is _not_?
  * train for a bigger alphabet and then cherry-pick the elements with lowest cross-confusion?
  * figure out a global metric for gauging overall alphabet unambiguity
    * based on non-diagonal elements for probability matrix?
    * based on classifier performance (test accuracy/loss)?
* think about implementing a metric for "have we seen something similar to this at any point"
  * if we use a separate network for this, does this basically make something along the lines of a GAN?
  * such a metric would go lower and lower for a very long time, as we will have seen more and more of the glyph space - limit by implementing forgetting (e.g. sliding average)? 
* implement a stop condition into the automated script (other than max allowed iterations)
  * based on one of the aforementioned metrics?
* think about training time:
  * force a small number of epochs? this basically introduces pressure to find stuff that is _easily learnable_ for a given classifier architecture, which might be the right track
* implement other classifiers
  * naive pixel-wise cross-correlation of average image for each category
* add a nice rendered Jupyter notebook with all outputs shown
