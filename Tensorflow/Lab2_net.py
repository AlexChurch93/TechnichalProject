############################################################
#                                                          #
#                       Based On Lab 2                     #
#                                                          #
############################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
import numpy as np
import cv2


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 1000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 8, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-3, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 320, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 240, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 1, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 1380, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

# 80-20 split
train_tfrecords_filename = 'tap_images_80-20_train.tfrecord'
validation_tfrecords_filename = 'tap_images_80-20_validation.tfrecord'

train_record_iterator = tf.python_io.tf_record_iterator(path=train_tfrecords_filename)
num_train_samples=0
for string_record in train_record_iterator:
    num_train_samples+=1

validation_record_iterator = tf.python_io.tf_record_iterator(path=validation_tfrecords_filename)
num_val_samples=0
for string_record in validation_record_iterator:
    num_val_samples+=1
validation_record_iterator = tf.python_io.tf_record_iterator(path=validation_tfrecords_filename)

print(num_train_samples)
print(num_val_samples)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
          'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
          'image/format': tf.FixedLenFeature([], tf.string, default_value='png'),
          'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
          'image/height': tf.FixedLenFeature([], tf.int64, default_value=240),
          'image/width': tf.FixedLenFeature([], tf.int64, default_value=320),
        })

    image = tf.image.decode_image(features['image/encoded'],3)
    img_format = tf.decode_raw(features['image/format'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)

    image = tf.image.rgb_to_grayscale(image)
    image = tf.reshape(image, [240,320,1])
    image = tf.image.per_image_standardization(image)

    images, labels = tf.train.shuffle_batch([image, label],
                                             batch_size=FLAGS.batch_size,
                                             capacity=30,
                                             num_threads=2,
                                             min_after_dequeue=10)
    return images, labels


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.
    Args:
        x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
          number of pixels in a standard CIFAR10 image.
    Returns:
        A tuple (y, img_summary)
        y: is a tensor of shape (N_examples, 10), with values
          equal to the logits of classifying the object images into one of 10 classes
          (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        img_summary: a string tensor containing sampled input images.
    """

    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
    img_summary = tf.summary.image('Input_images',x_image)

    # Small epsilon value for the BN transform
    epsilon = 1e-3

    # First convolutional layer - maps one RGB image to 32 feature maps.
    with tf.variable_scope("Conv_1"):
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope("Conv_2"):
        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope("FC_1"):
        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
        # is down to 8x8x64 feature maps -- maps this to 1024 features.
        W_fc1 = weight_variable([60*80*64, 200])
        b_fc1 = bias_variable([200])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 60*80*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.variable_scope("FC_2"):
        # Map the 1024 features to 10 classes
        W_fc2 = weight_variable([200, FLAGS.num_classes])
        b_fc2 = bias_variable([FLAGS.num_classes])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv, img_summary


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')


def main(_):
    tf.reset_default_graph()

    # create queue for pulling images
    filename_queue = tf.train.string_input_producer([train_tfrecords_filename])# add num_epochs to limit amount of time through data
    image,label = read_and_decode(filename_queue)

    with tf.variable_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # Build the graph for the deep net
    y_conv, img_summary = deepnn(x)

    with tf.variable_scope("x_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
    decay_steps = 1000  # decay the learning rate every 1000 steps
    decay_rate = 0.8  # the base of our exponential for the decay

    decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                               decay_steps, decay_rate, staircase=True)

    train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(cross_entropy, global_step=global_step)
    
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    acc_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", decayed_learning_rate)

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary, learning_rate_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    print('=============== Training ===============')
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        summary_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + "_validate", sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training
        for step in range(FLAGS.max_steps):
            
            # get batch of images
            batch_train_images, batch_train_labels = sess.run([image,label])

            # convert and reshape to match network
            batch_train_images = batch_train_images.astype(np.float32)
            batch_train_images = batch_train_images.reshape(FLAGS.batch_size,76800)

            #convert labels to sparse one hot 
            targets = np.array(batch_train_labels).reshape(-1)
            batch_train_labels = np.eye(FLAGS.num_classes)[targets]

            _, summary_str = sess.run([train_step, training_summary], feed_dict={x: batch_train_images, y_: batch_train_labels})
            
            if step % FLAGS.log_frequency == 0:
                summary_writer.add_summary(summary_str, step)

            if (step + 1) % FLAGS.log_frequency == 0:
                print(str(step+1) + ': Entropy = ' + str(cross_entropy.eval(feed_dict={x:batch_train_images,y_:batch_train_labels})) )

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + "_train", 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


        # Calculate Accuracy
        print('=============== Testing ===============')
        good = 0
        i=0
        for string_record in validation_record_iterator:
            i+=1
            
            # Parse the next example
            example = tf.train.Example()
            example.ParseFromString(string_record)

            val_label = int(example.features.feature['image/class/label']
                            .int64_list
                            .value[0])

            val_img_encoded = (example.features.feature['image/encoded']
                              .bytes_list
                              .value[0])

            img_1d = np.fromstring(val_img_encoded, dtype=np.uint8)
            val_img = cv2.imdecode(img_1d, cv2.IMREAD_GRAYSCALE)
            val_img = val_img.astype(np.float32)
            val_img = (val_img - np.mean(val_img)) / np.std(val_img)
            val_img = val_img.reshape(1,76800)

            val_label_4_print = val_label

            target = np.array(val_label).reshape(-1)
            val_lbl = np.eye(FLAGS.num_classes)[target]

            good += accuracy.eval(feed_dict={x:val_img,y_:val_lbl})
            print(i,': ', val_label_4_print,' = ', np.argmax(y_conv.eval(feed_dict={x:val_img,y_:val_lbl}),1)[0], 'Correct: ', good)
             

        print('Individual Test Accuracy: %g'%(good/num_val_samples))


if __name__ == '__main__':
    tf.app.run(main=main)
