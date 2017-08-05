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

For command line usage run `python imgstack.py --help

:Author:
    Guillaume Duranceau
"""

import argparse
import logging
import numpy
import tifffile
import os.path

logger = logging.getLogger('imgstack')

class TiffFile:
    """Load and cache TIFF file data.
    
    This object caches the image data for fast access, and manages partial load
    of the image.

    Image data is presented as a numpy array.

    Example
    -------
        image = TiffFile('file')
        image.shape  # shape of the image (as in numpy array)
        image.nbytes # in-memory size of the image (as in numpy array)
        image.dtype  # numpy type of image points
        image.cached # number of rows cached

        image[1]    # get the second row of the image
        image[:]    # get all image data
        image[0:2]  # get the first 2 rows
        image[3:-1] # get row 3 (numbered from 0) to the second last one
    """

    def __init__(self, filename, cache_size):
        """Create the file object and (partially) load the image.
        
        Parameters
        ----------
        filename : str
            TIFF file name.
        cache_size : int
            Amount of memory (in bytes) available to cache image data.
        """
        self.shape = None
        self.nbytes = None
        self.dtype = None
        self.cached = 0
        self._filename = filename

        data = self._load(True)
        if self.shape == None:
            return

        # the full image can be cached
        if cache_size >= data.nbytes:
            self._cache = data
            self.cached = len(self)
            self._cache_rbegin = 0
            self._cache_rlast = len(self)

        # let's cache the first rows
        else:
            self.cached = int(len(self) * cache_size / data.nbytes)
            if self.cached > 0:
                self._cache = data[:self.cached]
            self._cache_rbegin = 0
            self._cache_rlast = self.cached

    def __len__(self):
        """Get the number of rows in the image."""
        if self.shape:
            return self.shape[0]
        return 0

    def __getitem__(self, key):
        """Get image data for a single row or a slice of rows.

        For a 2 dimension image (standard case), keeps only RGB channels.

        Parameters
        ----------
        key : int or slice
            Index or slice of indices of the row(s) to retrieve.

        Returns
        -------
        data : numpy.ndarray
            Image data.
        """
        if self.shape == None:
            return numpy.array([], numpy.uint16)

        if isinstance(key, int):

            if key < 0:
                key += len(self)
            if (key < 0) or (key >= len(self)):
                raise IndexError("list index out of range")

            # get data from cache if possible
            if key >= self._cache_rbegin and key < self._cache_rlast:
                return self._cache[key - self._cache_rbegin]

            # clear the cache and load data from file
            self._cache = None
            data = self._load(False)

            # image couldn't be loaded
            if self.shape == None:
                return numpy.array([], numpy.uint16)

            # cache from row key
            self._cache_rbegin = key
            self._cache_rlast = min(key + self.cached, len(self))
            self._cache = data[self._cache_rbegin:self._cache_rlast]

            return data[key]

        elif isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]

        else:
            raise TypeError('TiffFile indices must be integers or slices')

    # load full image
    def _load(self, first_time):
        if first_time:
            logger.info("loading image [{}]".format(self._filename));
        else:
            logger.info("reloading image [{}]".format(self._filename));

        try:
            data = tifffile.imread(self._filename)

        except (OSError, IOError, ValueError) as err:
            logger.error(
                "couldn't load image [{}]: {}".format(self._filename, err))
            self.shape = None
            self.nbytes = None
            self.dtype = None
            return None

        # 2 dimension image => keep RGB channels only
        if data.ndim == 3 and data.shape[-1] >= 4:
            if first_time:
                logger.info(
                    "image [{}] has more than 3 chanels (maybe RGBA). "\
                    "keeping only the first 3 channels (RGB)".format(
                    self._filename, data.shape[-1]))
            data = numpy.delete(data, numpy.s_[3:], 2)

        self.shape = data.shape
        self.nbytes = data.nbytes
        self.dtype = data.dtype

        return data

class ImageStacker:
    """Stack images with similar size, producing a sigma clipped average image.
    """

    def __init__(self, data, loop, sigma):
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
        """
        self._loop = loop
        self._sigma = sigma
        self._average = None # input data average
        self._deviation = None # input data standard deviation, times sigma
        self._loops_run = 0 # number of loops actually run
        self._clipped = 0 # number of points clipped

        self._data = numpy.ma.masked_array(
            numpy.stack(data, axis=0), numpy.ma.nomask)

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

        return self._average

    def clipped_points(self):
        """Return the number of points clipped"""
        return self._clipped

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

class TiffStacker:
    """Stack TIFF images with similar size, producing a sigma clipped average
    image.
    """

    def __init__(self, loop, sigma, cache_size, rows):
        """Create the stacker object.

        Parameters
        ----------
        loop : int
            Number of sigma clipped average algorithm loops. Several loops will
            clip more points. loop = 0 simply computes the average.
        sigma : float
            See ImageStacker.__init__ sigma parameter.
        cache_size : int
            Amount of memory (in bytes) used for caching image data.
        rows : int
            Number of rows stacked simultaneously.
        """
        self._loop = loop
        self._sigma = sigma
        self._cache_size = cache_size
        self._rows = rows
        self._outdata = []

    def run(self, inputfiles, outfile, compress = 0):
        """Compute the stacked image from TIFF files and write it.

        Parameters
        ----------
        inputfiles : sequence
            TIFF file names. This must contain at least 2 files.
        outfile : str
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
            return False;

        cache_per_image = int(self._cache_size / len(inputfiles))
        shape = None
        dtype = None

        # load images one by one. interrupt in case of error
        images = []
        for inputfile in inputfiles:
            image = TiffFile(inputfile, cache_per_image)
            if not image.shape:
                return False
            if not shape:
                shape = image.shape
                dtype = image.dtype
            elif shape != image.shape:
                logger.error(
                    "image [{}] has a different shape than the previous ones"\
                    .format(inputfile))
                return False
            elif dtype != image.dtype:
                logger.error(
                    "image [{}] has a different type than the previous ones"\
                    .format(inputfile))
                return False
            images.append(image)

        # stats on stacking processing
        clipped = 0
        loops_run = 0

        # while all rows have not been processed, stack them
        rbegin = 0
        while rbegin < shape[0]:
            rend = min(rbegin + self._rows, shape[0])

            # load needed rows of all images
            inputdata = []
            for image in images:
                data = image[rbegin:rend]
                if not data:
                    return False
                inputdata.append(data)

            stacker = ImageStacker(inputdata, self._loop, self._sigma)

            logger.info(
                "stacking images. rows {} to {}...".format(rbegin, rend - 1))

            self._outdata.append(stacker.run())

            # update stacking stats
            clipped += stacker.clipped_points()
            loops_run = max(loops_run, stacker.loops_run())

            rbegin = rend

        logger.info("{} loop(s) run. {}/{} points clipped".format(
            loops_run, clipped, numpy.product(shape) * len(inputfiles)))

        return TiffStacker._write_tiff(
            numpy.concatenate(self._outdata), outfile, compress, dtype)

    @staticmethod
    def _write_tiff(data, filename, compress, datatype):
        logger.info("writing image [{}]".format(filename));
        data_conversion = data.astype(datatype)
        try:
            tifffile.imsave(filename, data_conversion, compress=compress)
        except ValueError as err:
            logger.error(
                "couldn't write output image [{}]: {}".format(filename, err))
            return False
        return True

def check_args(args):
    if len(args.input) < 2:
        logger.error("only 1 input file provided")
        return False

    for f in args.input:
        if not os.path.isfile(f):
            logger.error("invalid file [{}]".format(f))
            return False

    if os.path.exists(args.output):
        logger.error("output file [{}] already exists".format(args.output))
        return False

    # try creating the output to report error immediately in case of
    # problem
    try:
        out = open(args.output, 'a')
    except Exception as err:
        logger.error(
            "couldn't open output file [{}]: {}".format(args.output, err))
        return False
    out.close()
    os.remove(args.output)

    return True

def main():
    help_description =\
        "Stack multiple TIFF files, producing a sigma clipped average image"
        
    help_loop =\
        "number of iterations. 0 simply computes the average image (default=1)"
    help_sigma =\
        "maximum deviation allowed. "\
        "values 1.0, 2.0, 3.0 respectively exclude approximately "\
        "32 percent, 5 percent, and 0.3 percent of points for a normal "\
        "distribution of data (default=2.5)"
    help_depth =\
        "TIFF data point depth. Supported values are 8, 16 and 32 bits "\
        "(32 bits is floating point). (default=16)"
    help_compress =\
        "zlib compression level for output file (default=0)"
    help_memory =\
        "amount of memory (in MiB) used for caching image data "\
        "(default=1024 MiB)"
    help_rows =\
        "number of rows stacked simultaneously. high values require more "\
        "memory (default=100)"

    parser = argparse.ArgumentParser(description=help_description)
    parser.add_argument('-i', '--input', help='input files', action='store', required=True, nargs='+')
    parser.add_argument('-o', '--output', help='output file', action='store', required=True)
    parser.add_argument('-l', '--loop', help=help_loop, action='store', type=int, default=1)
    parser.add_argument('-s', '--sigma', help=help_sigma, action='store', type=float, default=2.5)
    parser.add_argument('-c', '--compress', help=help_compress, action='store', type=int, default=0)
    parser.add_argument('-d', '--depth', help=help_depth, action='store', type=int, default=16)
    parser.add_argument('-m', '--memory', help=help_memory, action='store', type=int, default=1024)
    parser.add_argument('-r', '--rows', help=help_rows, action='store', type=int, default=100)
    parser.add_argument('-q', '--quiet', help='mute message output on stdout', action='store_true')
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    if not check_args(args):
        return 1

    stacker = TiffStacker(
        args.loop, args.sigma, args.memory * 1024**2, args.rows)
    if stacker.run(args.input, args.output, args.compress):
        return 0
    return 1

if __name__ == '__main__':
    main()
