import imgstack
import logging
import numpy
import unittest
import os

def datafile(filename):
    return os.path.join('test_data', filename)

def filereader(filename):
    return imgstack.TiffReader(datafile(filename))

def filewriter(filename):
    return imgstack.TiffWriter(datafile(filename))

class TestTiffReader(unittest.TestCase):

    def test_name(self):
        self.assertEqual(imgstack.TiffReader(datafile('name')).name(), datafile('name'))

    def test_invalid(self):
        self.assertFalse(imgstack.TiffReader(datafile('unexisting-file')).is_valid())

    def test_load(self):
        self.assertEqual(imgstack.TiffReader(datafile('1d_1.tif')).load().tolist(), [10, 100, 1000, 10000])
        self.assertEqual(imgstack.TiffReader(datafile('image_1.tif')).load().tolist(),
            [[[10, 100, 1000], [50, 500, 5000], [10000, 20000, 30000]], [[600, 500, 400], [20, 30, 40], [0, 3, 2]]])

class TestTiffWriter(unittest.TestCase):

    def test_invalid(self):
        self.assertFalse(imgstack.TiffWriter(datafile('unexisting-dir/out.tif')).is_valid())

    def test_write(self):
        data = imgstack.TiffReader(datafile('image_1.tif')).load()
        out_filename = datafile('out.tif')
        out = imgstack.TiffWriter(out_filename)
        out.write(data, numpy.uint16)
        out_read = imgstack.TiffReader(out_filename).load()
        self.assertEqual(out_read.tolist(), data.tolist())
        self.assertEqual(out_read.dtype, numpy.uint16)
        os.remove(out_filename)

    def test_write_compressed(self):
        data = imgstack.TiffReader(datafile('image_1.tif')).load()
        out_filename = datafile('out.tif')
        out = imgstack.TiffWriter(out_filename)
        out.write(data, numpy.uint16, 5)
        out_read = imgstack.TiffReader(out_filename).load()
        self.assertEqual(out_read.tolist(), data.tolist())
        self.assertEqual(out_read.dtype, numpy.uint16)
        os.remove(out_filename)

class TestImageStacker(unittest.TestCase):

    def test_1d_images(self):
        images = [filereader('1d_{}.tif'.format(i)).load() for i in range(1, 10)]

        # loop = 0 => average
        stacker = imgstack.ImageStacker(images, 0, 1)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [19, 4700, 1171, 9073])
        self.assertEqual(stacker.loops_run(), 0)
        self.assertEqual(stacker.clipped_points(), 0)

        # loop = 1, sigma=1.0
        stacker = imgstack.ImageStacker(images, 1, 1.0)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 99, 1015, 10144])
        self.assertEqual(stacker.loops_run(), 1)
        self.assertEqual(stacker.clipped_points(), 4)

        # loop = 2, sigma=1.0
        stacker = imgstack.ImageStacker(images, 2, 1.0)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 99, 1005, 9982])
        self.assertEqual(stacker.loops_run(), 2)
        self.assertEqual(stacker.clipped_points(), 12)

        # loop = 3, sigma=1.0
        stacker = imgstack.ImageStacker(images, 3, 1.0)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 99, 998, 9991])
        self.assertEqual(stacker.loops_run(), 3)
        self.assertEqual(stacker.clipped_points(), 22)

        # loop = 4, sigma=1.0
        stacker = imgstack.ImageStacker(images, 4, 1.0)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 100, 997, 10003])
        self.assertEqual(stacker.loops_run(), 4)
        self.assertEqual(stacker.clipped_points(), 26)

        # loop = 5, sigma=1.0
        stacker = imgstack.ImageStacker(images, 5, 1.0)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 100, 997, 10002])
        self.assertEqual(stacker.loops_run(), 5)
        self.assertEqual(stacker.clipped_points(), 28)

        # loop = 6, sigma=1.0
        stacker = imgstack.ImageStacker(images, 6, 1.0)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 100, 997, 10002])
        self.assertEqual(stacker.loops_run(), 6)
        self.assertEqual(stacker.clipped_points(), 28)

        # loop = 7, sigma=1.0
        stacker = imgstack.ImageStacker(images, 7, 1.0)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 100, 997, 10002])
        self.assertEqual(stacker.loops_run(), 6) # only 6 loops run
        self.assertEqual(stacker.clipped_points(), 28)

        # loop = 1, sigma=2.5
        stacker = imgstack.ImageStacker(images, 1, 2.5)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 99, 1015, 10144])
        self.assertEqual(stacker.loops_run(), 1)
        self.assertEqual(stacker.clipped_points(), 4)

        # loop = 2, sigma=2.5
        stacker = imgstack.ImageStacker(images, 2, 2.5)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 99, 1015, 9982])
        self.assertEqual(stacker.loops_run(), 2)
        self.assertEqual(stacker.clipped_points(), 5)

        # loop = 3, sigma=2.5
        stacker = imgstack.ImageStacker(images, 3, 2.5)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 99, 1015, 9982])
        self.assertEqual(stacker.loops_run(), 3)
        self.assertEqual(stacker.clipped_points(), 5)

        # loop = 4, sigma=2.5
        stacker = imgstack.ImageStacker(images, 4, 2.5)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(), [10, 99, 1015, 9982])
        self.assertEqual(stacker.loops_run(), 3) # only 3 loops run
        self.assertEqual(stacker.clipped_points(), 5)

    def test_2d_images(self):
        images = [filereader('image_{}.tif'.format(i)).load() for i in range(1, 5)]

        # loop = 0 => average
        stacker = imgstack.ImageStacker(images, 0, 1)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(),
            [[[10, 96, 1134], [52, 497, 4990], [10068, 19889, 29962]], [[601, 501, 377], [19, 31, 39], [1, 1, 1]]])
        self.assertEqual(stacker.loops_run(), 0)
        self.assertEqual(stacker.clipped_points(), 0)

        # loop = 1, sigma=1.0
        stacker = imgstack.ImageStacker(images, 1, 1)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(),
            [[[10, 99, 999], [49, 502, 5002], [10012, 20015, 29976]], [[600, 499, 399], [19, 31, 41], [1, 0, 1]]])
        self.assertEqual(stacker.loops_run(), 1)
        self.assertEqual(stacker.clipped_points(), 27)

        # loop = 2, sigma=1.0
        stacker = imgstack.ImageStacker(images, 2, 1)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(),
            [[[10, 100, 1000], [50, 502, 5003], [10012, 20014, 29976]], [[600, 500, 400], [19, 31, 41], [1, 0, 1]]])
        self.assertEqual(stacker.loops_run(), 2)
        self.assertEqual(stacker.clipped_points(), 42)

    def test_2d_with_alpha_images(self):
        images = [filereader('image_alpha_{}.tif'.format(i)).load() for i in range(1, 5)]

        # loop = 1, sigma=1.0
        stacker = imgstack.ImageStacker(images, 1, 1)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(),
            [[[10, 99, 999, 65535], [49, 500, 5001, 65535], [10012, 20015, 29952, 65535]], [[600, 500, 400, 65535], [16, 31, 41, 65535], [0, 0, 0, 0]]])
        self.assertEqual(stacker.loops_run(), 1)
        self.assertEqual(stacker.clipped_points(), 17)

    def test_4d_float_images(self):
        images = [filereader('4d_float_{}.tif'.format(i)).load() for i in range(1, 4)]

        # loop = 1, sigma=1.2, alpha = False
        stacker = imgstack.ImageStacker(images, 1, 1.2, False)
        self.assertEqual(stacker.run().astype(numpy.uint16).tolist(),
            [[[10, 102, 997, 8912], [49, 501, 4996, 62]], [[10002, 19992, 30020, 1040], [10, 19, 30, 499]]])
        self.assertEqual(stacker.loops_run(), 1)
        self.assertEqual(stacker.clipped_points(), 17)

class TestTiffStacker(unittest.TestCase):

    def test_errors(self):
        # no input file
        stacker = imgstack.TiffStacker(1, 1, 10)
        self.assertFalse(stacker.run([], filewriter('out.tif')))

        # only 1 file input file
        stacker = imgstack.TiffStacker(1, 1, 10)
        self.assertFalse(stacker.run([filereader('long_1d.tif')], filewriter('out.tif')))

        # input file doesn't exist
        stacker = imgstack.TiffStacker(1, 1, 10)
        self.assertFalse(stacker.run([filereader('long_1d.tif'), filereader('unexisting.tif')], filewriter('out.tif')))

        # stacking files with different shapes
        stacker = imgstack.TiffStacker(1, 1, 10)
        self.assertFalse(stacker.run([filereader('1d_1.tif'), filereader('image_1.tif')], filewriter('out.tif')))

        # stacking files with different types
        stacker = imgstack.TiffStacker(1, 1, 10)
        self.assertFalse(stacker.run([filereader('1d_1.tif'), filereader('uint8_1d.tif')], filewriter('out.tif')))

    def test_2_files(self):
        stacker = imgstack.TiffStacker(1, 1, 10)
        self.assertTrue(stacker.run([filereader('1d_1.tif'), filereader('1d_2.tif')], filewriter('out.tif')))

        self.assertEqual(filereader('out.tif').load().tolist(), [9, 97, 997, 9997])
        os.remove(datafile('out.tif'))

    def test_multiple_files(self):
        stacker = imgstack.TiffStacker(2, 1, 1)
        self.assertTrue(stacker.run([filereader('image_{}.tif'.format(i)) for i in range(1, 5)], filewriter('out.tif')))

        outfile = filereader('out.tif').load()
        self.assertEqual([outfile[i].tolist() for i in range (0, len(outfile))],
            [[[10, 100, 1000], [50, 502, 5003], [10012, 20014, 29976]], [[600, 500, 400], [19, 31, 41], [1, 0, 1]]])
        os.remove(datafile('out.tif'))

    def test_multiple_files_with_alpha(self):
        stacker = imgstack.TiffStacker(1, 1, 1)
        self.assertTrue(stacker.run([filereader('image_alpha_{}.tif'.format(i)) for i in range(1, 5)], filewriter('out.tif')))

        outfile = filereader('out.tif').load()
        self.assertEqual([outfile[i].tolist() for i in range (0, len(outfile))],
            [[[10, 99, 999, 65535], [49, 500, 5001, 65535], [10012, 20015, 29952, 65535]], [[600, 500, 400, 65535], [16, 31, 41, 65535], [0, 0, 0, 0]]])
        os.remove(datafile('out.tif'))

    def test_float32_compressed(self):
        stacker = imgstack.TiffStacker(2, 1, 1, False)
        self.assertTrue(stacker.run(
            [filereader('4d_float_{}.tif'.format(i)) for i in range(1, 4)], filewriter('out.tif'), 5))

        outfile = filereader('out.tif').load()
        self.assertEqual([outfile[i].tolist() for i in range (0, len(outfile))],
            [[[10.5, 102.5, 997.5, 8912], [49.5, 501, 4996, 62.0]], [[10002.0, 19992.5, 30020.5, 1040.5], [10.0, 19.0, 30.5, 499]]])
        os.remove(datafile('out.tif'))

if __name__ == '__main__':
    logging.basicConfig(level=100) # deactivate logging
    unittest.main()
