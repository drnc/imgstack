========
imgstack
========

Copyright 2017, Guillaume Duranceau

https://github.com/drnc/imgstack

*imgstack* is a **TIFF file stacker**,
delivered as a **Python** script.
It stacks aligned images of the same scene,
producing a **sigma clipped average image**.

Sigma clipped average
=====================

Sigma clipping consists in the following operations,
for each color component of each image pixel (called point below):

1. Compute the average value of a same point in all images

2. Compute the `standard deviation`_ of a same point in all images

3. Exclude points whose values if off from the average by a value
   greater then a factor times the standard deviation
   (a small factor excludes more points than a bigger one).

This process can be done iteratively.
At the end, the average value of the same **remaining** points
in all images is computed to obtain the resulting image.

The benefit of sigma clipping compared to simply averaging images
is that **it removes abnormal values from the end result**.
When stacking astrophotographies for example,
sigma clipped average removes light trails from airplanes or meteors.

Features and options
====================

*imgstack* performs all computation with 64 bits floating point values.
It writes the resulting TIFF image file
with the same type than the input images
(if input images are 16 bits TIFF files,
output image is a 16 bits TIFF file).

*imgstack* can optionally compress the output TIFF file.

*imgstack* supports setting the sigma factor value.
Reasonable values range from 1.0 to 3.0.
Default is 2.5.
Assuming a `normal distribution`_ of data,
the following sigma values excludes approximately:

* 32% of points for sigma=1.0
* 13% of points for sigma=1.5
* 5% of points for sigma=2.0
* 1% of points for sigma=2.5
* 0.3% of points for sigma=3.0

The algorithm can be applied iteratively.
*imgstack* allows to set the number of sigma clipping iterations to run.
Note that *imgstack* will stop by itself
if the number of iterations specified is large and
if it detects that no points were excluded
in the last sigma clipping pass.

Stacking images is a CPU intensive and memory consuming operation,
especially when processing many large images.
To **limit memory consumption**,
*imgstack* **stacks images by group of limited number of rows**
(100 by default,
but this can lowered
to accomodate a user environment with low memory constraints).

*imgstack* can deal with transparency layer on images.
If an image includes an alpha transparency channel,
only points with maximum opacity
are considered during the stacking operation.

If present in the first input image,
*imgstack* copies the embedded color profile
in the output image.

How to run
==========

imgstack.py_ is a Python 3 script,
depending on numpy_ and tifffile_ packages.

To run it and display the usage and options::

    python imgstack.py -h

.. _standard deviation: https://en.wikipedia.org/wiki/Standard_deviation
.. _normal distribution: https://en.wikipedia.org/wiki/Normal_distribution
.. _numpy: http://www.numpy.org/
.. _tifffile: http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
.. _imgstack.py: https://github.com/drnc/imgstack/blob/master/imgstack/imgstack.py
