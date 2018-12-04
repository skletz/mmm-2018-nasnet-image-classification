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
# ==============================================================================
"""Script that classifies a tfrecord or image file. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
import time

from nets import nets_factory
from preprocessing import preprocessing_factory

# Get labels of imagenet data
from datasets import imagenet

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/checkpoints/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'input', '/tmp/tfrecord/ OR /tmp/images/image.jpg',
    'Directory containing TFRecord files or an image file to classify.')

# tf.app.flags.DEFINE_boolean('tfrecords', False, 'Input is a directory containing files formatted as TFRecord.')

tf.app.flags.DEFINE_boolean('tfrecord', False, 'Input is a file formatted as TFRecord.')

tf.app.flags.DEFINE_string(
    'split_name', 'shard',
    'If input is a directory, the file pattern of the TFRecord files.')

tf.app.flags.DEFINE_string(
    'output', '/tmp/classification/ OR /tmp/classification/image.txt',
    'Directory or an file to store classification')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'model_name', 'nasnet_large', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 331, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def _classify_image(data_path):
    tf.enable_eager_execution()

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        with tf.gfile.FastGFile(data_path, 'rb') as f:
            image_data = f.read()

        # Decode the RGB JPEG.
        image = tf.image.decode_jpeg(image_data, channels=3, try_recover_truncated=True, acceptable_fraction=0.3)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(1001),
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        # Networks accept images in batches.
        # The first dimension usually represents the batch size.
        # In our case the batch size is one.
        processed_images = tf.expand_dims(processed_image, 0)

        ####################
        # Define the model #
        ####################
        logits, end_points = network_fn(processed_images)

        probabilities = tf.nn.softmax(logits)

        ####################
        # Restore model variables #
        ####################
        variables_to_restore = slim.get_variables_to_restore()
        init_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.checkpoint_path), variables_to_restore)

        with tf.Session() as sess:
            tf.global_variables_initializer()

            init_fn(sess)
            np_image, network_input, probabilities = sess.run([image, processed_image, probabilities])

            probabilities = probabilities[0, 0:]

            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        names = imagenet.create_readable_names_for_imagenet_labels()

        for i in range(10):
            index = sorted_inds[i] - 1
            print('%0.5f => [%s]' % (probabilities[index + 1], names[index + 1]))


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    data_path = FLAGS.input

    eval_image_size = FLAGS.eval_image_size

    if FLAGS.tfrecord:
        fls = tf.python_io.tf_record_iterator(path=data_path)
    else:
        _classify_image(data_path)
        return

    checkpoint_path = FLAGS.checkpoint_path

    image_string = tf.placeholder(tf.string)

    image = tf.image.decode_jpeg(
        image_string, channels=3,
        try_recover_truncated=True,
        acceptable_fraction=0.3)

    # Select the model
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=1001,
        is_training=False)

    # Select the preprocessing function
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    # Input is a batch
    processed_images = tf.expand_dims(processed_image, 0)

    # Define the model
    logits, end_points = network_fn(processed_images)
    probabilities = tf.nn.softmax(logits)

    # Restore model variables
    variables_to_restore = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.checkpoint_path), variables_to_restore)

    # Run session
    sess = tf.Session()
    init_fn(sess)

    output_dir = FLAGS.output
    label_names = imagenet.create_readable_names_for_imagenet_labels()

    counter = 0
    start_time = time.time()
    for fl in fls:
        filename = None


        try:
            # Parse example proto
            example = tf.train.Example()
            example.ParseFromString(fl)

            filename = example.features.feature['image/filename'].bytes_list.value[0]
            output_filename = filename
            output_file = output_filename.replace('.jpg', '_nasnet.csv')

            output_file = os.path.join(output_dir, output_file)

            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            # Create one output file per image
            output_stream = open(output_file, 'w+')

            example_image = example.features.feature['image/encoded'].bytes_list.value[0]  # retrieve image string

            results = sess.run(probabilities, feed_dict={image_string: example_image})

        except Exception as e:
            tf.logging.warn('Cannot process image file %s' % fl)
            continue

        results = results[0, 0:]
        #print("Results=", results)

        sorted_indices = [i[0] for i in sorted(enumerate(-results), key=lambda x: x[1])]

        for i in range(10):
            index = sorted_indices[i] - 1
            print('%d;%0.5f;%s' % (index, results[index + 1], label_names[index + 1]), file=output_stream)

        output_stream.close()

        counter = counter + 1
        if not counter % 10:
            interim_time = time.time()
            elapsed_time = interim_time - start_time
            avg_time_per_image = elapsed_time / counter
            tf.logging.debug("Processed images: %d in %0.5f seconds (avg. seconds per image %0.5f(s))" % (counter, elapsed_time, avg_time_per_image))

            #print("Processed images: %d in %0.5f seconds (avg. seconds per image %0.5f(s))" % (counter, elapsed_time, avg_time_per_image))


    sess.close()

if __name__ == '__main__':
    tf.app.run()
