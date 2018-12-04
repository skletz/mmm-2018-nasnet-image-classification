#  *****************************************************************************
#       _    _      ()_()
#      | |  | |    |(o o)
#   ___| | _| | ooO--`o'--Ooo
#  / __| |/ / |/ _ \ __|_  /
#  \__ \   <| |  __/ |_ / /
#  |___/_|\_\_|\___|\__/___|
#  *****************************************************************************
#
#  Created by Sabrina Kletz on 3.12.18.
#  Copyright  2018 Sabrina Kletz. All rights reserved.
#  Parts of the source are based on build_image_data.py provided by ./research/inception
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.
  data_dir/00001/image0.jpeg
  data_dir/00001/image1.jpg
  ...
  data_dir/00002/weird-image.jpeg
  data_dir/00002/my-image.jpeg
  ...
where the sub-directory bundles a set of images.
This TensorFlow script converts the data into
a sharded data set consisting of TFRecord files
  output_dir/split-00000-of-01024
  output_dir/split-00001-of-01024
  ...
  output_dir/split-01023-of-01024

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'
  image/filename: string containing the basename of the image file
            e.g. '00002_0000049.JPEG' or '00002_0007249.JPEG'
"""

import os
import sys
import threading
from datetime import datetime

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', '/tmp/vbs2019/keyframes',
    'Path to the dataset directory.')

tf.app.flags.DEFINE_string(
    'output_dir', '/tmp/vbs2019/tfrecords',
    'Output data directory.')

tf.app.flags.DEFINE_integer('num_threads', 1,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('shards', 1,
                            'Number of shards in TFRecord files.')

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    name = os.path.basename(filename)
    parent_dir = os.path.split(os.path.dirname(filename))[1]

    filename = os.path.join(parent_dir, name)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


def _find_image_files(input_dir):
    """Build a list of all images files.

    Args:
      input_dir: string, path to the root directory of images.

        Assumes that the image data set resides in JPEG files located in
        the following directory structure.

          input_dir/00001/another-image.JPEG
          input_dir/00001/a-image.JPEG
          input_dir/00002/my-image.jpg

        where '00001' bundles a set of images.

    Returns:
      filenames: list of strings; each string is a relative path to an image file.
        path_to/00001/another-image.JPEG
        path_to/00001/a-image.JPEG
        path_to/00002/my-image.jpg
    """
    print("Find image files ....")

    filenames = []

    # Get all directories in input dir
    root = input_dir
    files = os.listdir(root)
    subdirectories = []
    for name in files:
        path = os.path.join(os.path.abspath(root), name)
        # Ignore files like .DS_Store
        if os.path.isdir(path):
            subdirectories.append(name)

    if not len(subdirectories) == 0:
        print ("Found %s subdirectorie(s)." % len(subdirectories))
    else:
        print ("No subdirectories found.")
        return filenames

    # Get all files within each subdirectory
    for dir in subdirectories:
        path = os.path.join(os.path.abspath(root), dir)
        files = os.listdir(path)

        # loop through all the files and folders
        for filename in files:
            object = os.path.join(os.path.abspath(path), filename)
            # check whether the current object is a image file or not
            if not os.path.isdir(object) and os.path.splitext(object)[1] == ".jpg":
                filenames.append(object)

    if not len(filenames) == 0:
        print ("Found %s file(s)." % len(filenames))
    else:
        print ("No files found.")

    return filenames


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, num_shards):
    num_threads = len(ranges)

    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
        filename = filenames[i]

        try:
            image_buffer, height, width = _process_image(filename, coder)
        except Exception as e:
            print(e)
            print('SKIPPED: Unexpected error while decoding %s.' % filename)
            continue

        example = _convert_to_example(filename, image_buffer, height, width)
        writer.write(example.SerializeToString())
        shard_counter += 1
        counter += 1

        if not counter % 1000:
            print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                  (datetime.now(), thread_index, counter, num_files_in_thread))
            sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))



    sys.stdout.flush()
    shard_counter = 0


def _process_keyframes(input_dir, output_dir):
    print("Process keyframes ....")

    name = "shard"
    filenames = _find_image_files(input_dir)

    # Split filenames according to the number of threads
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, FLAGS.shards)

        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def main(unused_argv):
    """Load and process keyframes"""
    print("Main ....")
    print("Command line arguments: ")
    print("--dataset_dir ", FLAGS.dataset_dir)
    print("--output_dir ", FLAGS.output_dir)

    if not os.path.exists(FLAGS.dataset_dir):
        print "Dataset directory does not exits!"
        exit(-1)

    if not os.path.exists(FLAGS.dataset_dir):
        print "Output directory does not exits!"
        exit(-1)

    _process_keyframes(FLAGS.dataset_dir, FLAGS.output_dir)


if __name__ == '__main__':
    print "build_data run ..."
    tf.app.run()
    print "build_data finished!"
