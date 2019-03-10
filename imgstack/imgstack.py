#!/usr/bin/env python
# imgstack.py

# Copyright 2017 Guillaume Duranceau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stack multiple TIFF files, producing a sigma clipped average image

For command line usage, run:
    python imgstack.py --help

:Author:
    Guillaume Duranceau
"""

import argparse
import logging
import numpy
import tifffile
import os.path

logger = logging.getLogger('imgstack')

class TiffReader:
    """Read image data from a TIFF file."""

    def __init__(self, filename):
        """Create the file reader object.

        Parameters
        ----------
        filename : str
            Path of the file
        """
        self._filename = filename
        self._tags = []

    def name(self):
        """Return the reader's file name"""
        return self._filename

    def is_valid(self):
        """Check that this is a valid and readable file"""
        if not os.path.isfile(self._filename):
            logger.error("invalid file [{}]".format(self._filename))
            return False

        if not os.access(self._filename, os.R_OK):
            logger.error(
                "cannot read file [{}] (no permission)".format(self._filename))
            return False

        return True

    def load(self):
        """Load file image from the TIFF file as numpy array.

        Returns
        -------
        data : numpy.ndarray
            Image data content
        """
        logger.info("loading image [{}]".format(self._filename))
        try:
            with tifffile.TiffFile(self._filename) as tif:
                data = tif.asarray(out='memmap')
                self._extract_icc(tif.pages[0].tags)
        except (OSError, IOError, ValueError) as err:
            logger.error(
                "couldn't load image [{}]: {}".format(self._filename, err))
            return None
        return data

    def tags(self):
        """Return ICC profile tag extracted from the input image."""
        return self._tags

    def _extract_icc(self, tags):
        if 'InterColorProfile' in tags:
            tag = tags['InterColorProfile']
            self._tags.append((tag.code, 'B', tag.count, tag.value, True))

class TiffWriter:
    """Write image data in a TIFF file."""

    def __init__(self, filename):
        """Create the file writer object.

        Parameters
        ----------
        filename : str
            Path of the file
        """
        self._filename = filename

    def is_valid(self):
        """Check that the file doesn't already exist and can be created"""
        if os.path.exists(self._filename):
            logger.error("file [{}] already exists".format(self._filename))
            return False

        try:
            out = open(self._filename, 'a')
        except Exception as err:
            logger.error(
                "cannot open file [{}]: {}".format(self._filename, err))
            return False
        out.close()
        os.remove(self._filename)

        return True

    def write(self, data, datatype, compress=0, tags=[]):
        """Write image data in the TIFF file.

        Parameters
        ----------
        data : numpy.ndarray
            Image data,
        datatype : numpy.dtype
            TIFF data type
        compress : int
            Zlib compression level for TIFF data written.
        tags : sequence of tuple
            Extra tags to write in image, as defined in tifffile.TiffWriter.save
        """
        logger.info("writing image [{}]".format(self._filename))
        data_conversion = data.astype(datatype)
        try:
            tifffile.imwrite(
                self._filename, data_conversion, compress=compress,
                extratags=tags)
        except ValueError as err:
            logger.error(
                "couldn't write image [{}]: {}".format(self._filename, err))
            return False
        return True

class ImageStacker:
    """Stack images with similar size, producing a sigma clipped average image.
    """

    def __init__(self, data, loop, sigma, alpha=True):
        """Create the stacker object.

        Parameters
        ----------
        data : numpy.ndarray
            Images data in a single numpy array (1 entry per image).
        loop : int
            Number of sigma clipped average algorithm loops. Several loops will
            clip more points.
        sigma : float
            A point with a value such as its difference from the average value
            of the same points in all images is greater than sigma times the
            standard deviation for those points will be clipped.
        alpha : bool
            True to consider the 4th channel of a 2D image data as an alpha
            (transparency) channel. Transparent pixels are not taken into
            account when computing the resulting stacked image.
        """
        self._loop = loop
        self._sigma = sigma
        self._average = None # input data average
        self._deviation = None # input data standard deviation, times sigma
        self._loops_run = 0 # number of loops actually run
        self._clipped = 0 # number of points clipped
        self._masked = 0 # number of points masked initially

        self._data = numpy.ma.masked_array(
            numpy.stack(data, axis=0), numpy.ma.nomask)

        # alpha transparency layer is used to generate a mask on the data:
        # pixel with maximum opacity are not masked, all others are.
        if alpha and self._data.ndim == 4 and self._data.shape[-1] >= 4:
            self._data.mask = numpy.repeat(numpy.where(
                self._data[...,3] >= ImageStacker.max_opacity(self._data.dtype),
                False, True), 4)
            self._masked = numpy.ma.count_masked(self._data)

    def run(self):
        """Compute the stacked image.

        Returns
        -------
        image : numpy.ndarray
            Stacked image data. The returned image has the same shape than the
            images loaded. dtype=float64
        """

        if not self._average:
            self._compute_average()

        # less than 2 images provided: simply return the average
        if len(self._data) < 3:
            return self._average

        for n in range(0, self._loop):
            self._loops_run += 1
            self._compute_standard_deviation()
            previous_clipped = self._clipped # current number of points clipped
            self._compute_clipped_data()
            self._compute_average()
            if self._clipped == previous_clipped:
                # no point was clipped in this loop. no need for another one
                break

        # replace None values in the average by 0. None values are possible if
        # some points are transparent in all images
        return numpy.where(self._average == None, 0, self._average)

    def clipped_points(self):
        """Return the number of points clipped"""
        return self._clipped - self._masked

    def loops_run(self):
        """Return the number of loops run.

        This can be lower than the providing loop number in the constructor if
        the stacker realizes that the last loop didn't clip any data.
        """
        return self._loops_run

    def _compute_average(self):
        self._average = numpy.ma.mean(self._data, axis=0)

    def _compute_standard_deviation(self):
        self._deviation = \
            self._sigma * numpy.ma.std(self._data, axis=0)

    def _compute_clipped_data(self):
        # for each point, compare the difference to the average from the
        # maximum deviation allowed.
        clip = abs(self._data - self._average) - self._deviation
        # generate a mask of data to clip
        new_mask = numpy.ma.masked_greater(clip, 0, copy=False).mask
        # apply the mask on input data
        self._data.mask = numpy.ma.mask_or(self._data.mask, new_mask)
        self._clipped = numpy.ma.count_masked(self._data)

    @staticmethod
    def max_opacity(dtype):
        if numpy.issubdtype(dtype, numpy.integer):
            return numpy.iinfo(dtype).max
        elif numpy.issubdtype(dtype, numpy.float):
            return 1.0
        else:
            return 0

class TiffStacker:
    """Stack TIFF images with similar size, producing a sigma clipped average
    image.
    """

    def __init__(self, loop, sigma, rows, alpha=True):
        """Create the stacker object.

        Parameters
        ----------
        loop : int
            Number of sigma clipped average algorithm loops. Several loops will
            clip more points. loop = 0 simply computes the average.
        sigma : float
            See ImageStacker.__init__ sigma parameter.
        rows : int
            Number of rows stacked simultaneously.
        alpha : bool
            True to consider the 4th channel of a 2D image data as an alpha
            (transparency) channel.
        """
        self._loop = loop
        self._sigma = sigma
        self._rows = rows
        self._alpha = alpha
        self._outdata = []

    def run(self, inputfiles, outfile, compress=0):
        """Compute the stacked image from TIFF files and write it.

        Parameters
        ----------
        inputfiles : sequence
            TIFF file readers. This must contain at least 2 readers.
        outfile : TiffWriter
            TIFF file where the stacked image is written.
        compress : int
            Zlib compression level for TIFF data written.

        Returns
        -------
        success : bool
            True if the run was successful and resulting image written.
            False in case an error occurred.
        """
        if len(inputfiles) < 2:
            return False

        shape = None
        dtype = None

        # load images one by one. interrupt in case of error
        images = []
        for inputfile in inputfiles:
            image = inputfile.load()
            if image is None:
                return False
            if not shape:
                shape = image.shape
                dtype = image.dtype
                tags = inputfile.tags()
            elif shape != image.shape:
                logger.error(
                    "image [{}] has a different shape than the previous ones"\
                    .format(inputfile.name()))
                return False
            elif dtype != image.dtype:
                logger.error(
                    "image [{}] has a different type than the previous ones"\
                    .format(inputfile.name()))
                return False
            images.append(image)

        # stats on stacking processing
        clipped = 0
        loops_run = 0

        # while all rows have not been processed, stack them
        outdata = []
        rbegin = 0
        while rbegin < shape[0]:
            rend = min(rbegin + self._rows, shape[0])

            # load needed rows of all images
            inputdata = []
            for image in images:
                data = image[rbegin:rend]
                inputdata.append(data)

            stacker = ImageStacker(
                inputdata, self._loop, self._sigma, self._alpha)

            logger.info(
                "stacking images. rows {} to {}...".format(rbegin, rend - 1))

            outdata.append(stacker.run())

            # update stacking stats
            clipped += stacker.clipped_points()
            loops_run = max(loops_run, stacker.loops_run())

            rbegin = rend

        logger.info("{} loop(s) run. {}/{} points clipped".format(
            loops_run, clipped, numpy.product(shape) * len(inputfiles)))

        return outfile.write(numpy.concatenate(outdata), dtype, compress, tags)

def parse_args():
    help_description =\
        "Stack multiple TIFF files, producing a sigma clipped average image"

    help_loop =\
        "number of iterations. 0 simply computes the average image (default=1)"
    help_sigma =\
        "maximum deviation allowed. "\
        "a pixel value off from the average by this value times the standard "\
        "deviation is excluded. "\
        "values 1.0, 2.0, 3.0 respectively exclude approximately "\
        "32 percent, 5 percent, and 0.3 percent of points for a normal "\
        "distribution of data (default=2.5)"
    help_compress =\
        "zlib compression level for output file (default=0)"
    help_rows =\
        "number of rows stacked simultaneously. high values require more "\
        "memory (default=100)"
    help_no_alpha =\
        "doesn't consider the 4th channel of 2D images as an alpha "\
        "transparency layer"

    parser = argparse.ArgumentParser(description=help_description)
    parser.add_argument('-i', '--input', help='input files', action='store', required=True, nargs='+')
    parser.add_argument('-o', '--output', help='output file', action='store', required=True)
    parser.add_argument('-l', '--loop', help=help_loop, action='store', type=int, default=1)
    parser.add_argument('-s', '--sigma', help=help_sigma, action='store', type=float, default=2.5)
    parser.add_argument('-c', '--compress', help=help_compress, action='store', type=int, default=0)
    parser.add_argument('-r', '--rows', help=help_rows, action='store', type=int, default=100)
    parser.add_argument('-a', '--no-alpha', help=help_no_alpha, action='store_true')
    parser.add_argument('-q', '--quiet', help='mute message output on stdout', action='store_true')
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    args.inputfiles = [TiffReader(f) for f in args.input]
    args.outputfile = TiffWriter(args.output)

    if len(args.inputfiles) < 2:
        logger.error("only 1 input file provided")
        return None

    for f in args.inputfiles + [args.outputfile]:
        if not f.is_valid():
            return None

    return args

def main():
    args = parse_args()
    if not args:
        return 1

    stacker = TiffStacker(args.loop, args.sigma, args.rows, not args.no_alpha)
    if stacker.run(args.inputfiles, args.outputfile, args.compress):
        return 0
    return 1

if __name__ == '__main__':
    main()
