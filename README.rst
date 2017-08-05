========
imgstack
========

*imgstack* is a TIFF file stacker.
It stacks aligned images of the same scene,
producing a sigma clipped average image.

Sigma clipping average
======================

Sigma clipping consists in the following operations,
for each image pixel and each color channel:

1. Compute the average value of the pixel color channel in all images

2. Compute the `standard deviation`_
   of the pixel color channel in all images

3. Exclude pixel values in image for which the difference
   between their value and the average is greater than
   a factor times the standard deviation
   (a small factor will exclude more pixels than a bigger one).

4. Compute the average value of the pixel color channel for all
   pixel values which have not been excluded in previous step.

The benefit of sigma clipping compared to averaging the images
is that it removes from the end results abnormal values.
When stacking astrophotographies for example,
sigma clipping average removes light trails from airplane or meteors.

Features and options
====================

*imgstack* supports setting the sigma factor value.
Reasonable values range from 1.0 to 3.0.
Default is 2.5.
Assuming a `normal distriction`_ of data,
the following sigma values would exclude approximately:
* 32% of points for sigma=1.0
* 5% of points for sigma=2.0
* 0.3% of points for sigma=3.0

The algorithm can be applied several times.
*imgstack* allows to set the number of sigma clipping loops to run.
Note that *imgstack* will stop by itself
if the number of loops specified is large and
if it detects that no points were excluded
in the last sigma clipping pass.

*imgstack* writes a result TIFF image file
with the same type than the input images
(if input images are 16 bits TIFF, output image is 16 bits TIFF)
It can also optionnaly compress the resulting TIFF file.

Stacking images is a CPU intensive and memory consuming operation,
especially when processing many large images.
*imgstack* doesn't load the full content of images in memory.
It loads partial images only in memory
(a limit of 1 GiB is set by default),
and stack them by group of limited number of rows (100 by default).
Once the partial content of the images have been fully processed,
it loads the following parts of the images,
and continues the stacking process, until full completion.
Those default values (1 GiB of memory and stacking by 100 rows)
can be changed to accomodate the user environment
(low memory constraints, many images to stack...)

How to run
==========

imgstack.py is a Python 3 script,
depending on numpy_ and tifffile_ packages.

To run it and display the usage and options::

    python imgstack.py -h

.. _standard deviation: https://en.wikipedia.org/wiki/Standard_deviation
.. _normal distribution: https://en.wikipedia.org/wiki/Normal_distribution
.. _numpy: http://www.numpy.org/
.. _tifffile: http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
