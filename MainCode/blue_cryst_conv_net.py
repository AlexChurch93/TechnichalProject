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
import math

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 5000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 99,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 100,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 64, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 0.0008, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 320, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 240, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 1, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 2, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}_ms_{ms}'.format(bs=FLAGS.batch_size,
                                                                lr=FLAGS.learning_rate,
                                                                ms=FLAGS.max_steps))

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

# 80-20 split
train_tfrecords_filename = 'tfrecord_data/tap_images_80-20_train.tfrecord'
validation_tfrecords_filename = 'tfrecord_data/tap_images_80-20_validation.tfrecord'

train_record_iterator = tf.python_io.tf_record_iterator(path=train_tfrecords_filename)
num_train_samples=0
for string_record in train_record_iterator:
    num_train_samples+=1

validation_record_iterator = tf.python_io.tf_record_iterator(path=validation_tfrecords_filename)
num_val_samples=0
for string_record in validation_record_iterator:
    num_val_samples+=1
validation_record_iterator = tf.python_io.tf_record_iterator(path=validation_tfrecords_filename)

print('=============== Initialising ===============')
print('Number of Training Samples :',num_train_samples)
print('Number of Validation Samples :',num_val_samples)

# Read in the label file once and build a list of line offsets
lookup_filename = 'tfrecord_data/labels.txt'
lookup_file = open(lookup_filename, "rb", 0)
line_offset = []
offset = 0
for line in lookup_file:
    line_offset.append(offset)
    offset += len(line)
lookup_file.seek(0)

#Resizing Variables
resize_width = 80
resize_height = 80
resize_flag = 1

#Network Variables
num_fc1_nodes = 1024
num_fc2_nodes = 200
weight_mean = 0
weight_stdv = 0.01

def read_and_decode(filename_queue, train_flag):
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

    image = tf.image.decode_image(features['image/encoded'],1)
    img_format = tf.decode_raw(features['image/format'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)

    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [FLAGS.img_height,FLAGS.img_width,1])
    image = tf.image.per_image_standardization(image)

    if resize_flag:
        image = tf.image.resize_images(image,[resize_height,resize_width]) # resize
    else:
        image = tf.image.resize_images(image,[FLAGS.img_height,FLAGS.img_width]) # resize


    images, labels = tf.train.shuffle_batch([image, label],
                                             batch_size=FLAGS.batch_size,
                                             capacity=30,
                                             num_threads=2,
                                             min_after_dequeue=10,
                                             allow_smaller_final_batch=True)
    
    #tf.cond(train_flag,lambda: print('training'), lambda: print('validatiing'))
    
    return images, labels

## ====================== Building the Network ======================
def read_labels(n):
    lookup_file.seek(line_offset[n])
    string_label = lookup_file.readline().decode().rstrip() # read line, decode, remove \n
    string_label = string_label[string_label.index(':')+1:]
    string_label = string_label.split('_')
    int_label = list(map(int, string_label))
    
    # map from index to value
    mapped_int_label = [None,None] # initialise
    
    # range 1:46 to 46:-44
    a1 = 1.0
    a2 = 46.0
    b1 = 46.0
    b2 = -44.0

    mapped_int_label[0] = b1 + (int_label[0]-a1)*(b2-b1)/(a2-a1)

    # range 1:30 to 0:14.5
    a1 = 1.0
    a2 = 30.0
    b1 = 0
    b2 = 14.5

    mapped_int_label[1] = b1 + (int_label[1]-a1)*(b2-b1)/(a2-a1)
        
    return mapped_int_label
    #return int_label

def deepnn(x_image):

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

    with tf.variable_scope("Conv_3"):
        # Second convolutional layer -- maps 64 feature maps to 64.
        W_conv3 = weight_variable([5, 5, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        # Second pooling layer.
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.variable_scope("FC_1"):
        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
        # is down to 8x8x64 feature maps -- maps this to 1024 features.

        if resize_flag:
            W_fc1 = weight_variable([int(resize_width/8)*int(resize_height/8)*64, num_fc1_nodes]) # resize
            h_pool3_flat = tf.reshape(h_pool3, [-1, int(resize_width/8)*int(resize_height/8)*64]) # resize
        else:
            W_fc1 = weight_variable([int(FLAGS.img_width/8)*int(FLAGS.img_height/8)*64, num_fc1_nodes])
            h_pool3_flat = tf.reshape(h_pool3, [-1, int(FLAGS.img_width/8)*int(FLAGS.img_height/8)*64])
            
        b_fc1 = bias_variable([num_fc1_nodes])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    with tf.variable_scope("FC_2"):
        W_fc2 = weight_variable([num_fc1_nodes, num_fc2_nodes])
        b_fc2 = bias_variable([num_fc2_nodes])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.variable_scope("OUT"):
        W_out2 = weight_variable([num_fc2_nodes, FLAGS.num_classes])
        b_out2 = bias_variable([FLAGS.num_classes])

        y_conv = tf.matmul(h_fc2, W_out2) + b_out2

    return y_conv


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=weight_stdv)
    #initial = tf.random_uniform(shape, minval = -0.005, maxval = 0.005)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')
    

def main(_):
    tf.reset_default_graph()
        
    with tf.variable_scope("inputs"):
        
        if resize_flag:
            x = tf.placeholder(tf.float32, [None, resize_width * resize_height * FLAGS.img_channels]) #resize
            x_image = tf.reshape(x, [-1, resize_height, resize_width, FLAGS.img_channels]) #resize
        else:
            x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
            x_image = tf.reshape(x, [-1, FLAGS.img_height, FLAGS.img_width, FLAGS.img_channels])

        y_ = tf.placeholder(tf.float32)
        #y_ = tf.placeholder(tf.int32)
        train_flag = tf.placeholder(tf.bool)

    # create queues for pulling training and testing images from record
    train_filename_queue = tf.train.string_input_producer([train_tfrecords_filename])# add num_epochs to limit amount of time through data
    train_image,train_label = read_and_decode(train_filename_queue,train_flag)

    val_filename_queue = tf.train.string_input_producer([validation_tfrecords_filename])
    val_image,val_label = read_and_decode(val_filename_queue,train_flag)

    test_filename_queue = tf.train.string_input_producer([validation_tfrecords_filename], num_epochs=1)
    test_image, test_label = read_and_decode(val_filename_queue,train_flag)

    with tf.variable_scope("model"):
        # Build the graph for the deep net
        y_conv = deepnn(x_image)

        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        cost = tf.losses.mean_squared_error(y_,y_conv)
        val_cost = tf.losses.mean_squared_error(y_,y_conv)
        
##        correct_prediction = tf.reduce_min( tf.cast((tf.equal(tf.cast(tf.round(y_conv),tf.int32), y_)),tf.float32),1 )
##        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        
        # correct if abs(y_conv[0] - y_[0]) < 1
        # and     if abs(y_conv[1] - y_[1]) < 0.25
        abs_diff   = tf.abs(y_-y_conv)
        theta_diff = abs_diff[:,0]
        r_diff     = abs_diff[:,1]

        correct_theta = tf.less_equal(theta_diff, 1)
        correct_r = tf.less_equal(r_diff, 0.25)

        combined = tf.concat(tf.transpose([correct_theta, correct_r]), 0)
        correct_prediction = tf.reduce_min(tf.cast(combined, tf.float32),1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))


        global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
        decay_steps = 1000  # decay the learning rate every 1000 steps
        decay_rate = 0.8  # the base of our exponential for the decay

        decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                   decay_steps, decay_rate, staircase=True)

        train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(cost, global_step=global_step)
        #train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost, global_step=global_step)
    

    # summaries for TensorBoard visualisation
    loss_summary = tf.summary.scalar("Loss", cost)
    val_loss_summary = tf.summary.scalar("Validation Loss", val_cost)
    acc_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", decayed_learning_rate)
    
    train_img_summary = tf.summary.image('Train Images',x_image)
    val_img_summary = tf.summary.image('Validation Images', x_image)
    test_img_summary = tf.summary.image('Test Images', x_image)

    training_summary = tf.summary.merge([train_img_summary, loss_summary, learning_rate_summary])
    validation_summary = tf.summary.merge([val_img_summary, acc_summary, val_loss_summary])
    test_summary = tf.summary.merge([test_img_summary])


    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    # The op for initializing the variables, call after optimiser created
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    print('=============== Training ===============')
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validate", sess.graph)
        test_writer = tf.summary.FileWriter(run_log_dir + "_test", sess.graph)

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Training
        for step in range(FLAGS.max_steps):
            
            # get batch of images
            train_imgs, train_lbls_id = sess.run([train_image,train_label])

            # convert and reshape to match network
            train_imgs = train_imgs.astype(np.float32)
            
            if resize_flag:
                train_imgs = train_imgs.reshape(FLAGS.batch_size,resize_width*resize_height) # resize
            else:
                train_imgs = train_imgs.reshape(FLAGS.batch_size,FLAGS.img_width*FLAGS.img_height)

            #convert labels to angle and displacement
            train_lbls = np.zeros((FLAGS.batch_size,2))
            for i in range(FLAGS.batch_size):
                train_lbls[i,:] = read_labels(train_lbls_id[i])

            _, train_summary_str = sess.run([train_step, training_summary], feed_dict={x: train_imgs, y_: train_lbls, train_flag:1})
            
            if step % FLAGS.log_frequency == 0:
                train_writer.add_summary(train_summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % (FLAGS.log_frequency + 1) == 0:
                # get batch of images
                val_imgs,val_lbls_id = sess.run([val_image,val_label])
                
                # convert and reshape to match network
                val_imgs = val_imgs.astype(np.float32)
                
                if resize_flag:
                    val_imgs = val_imgs.reshape(val_imgs.shape[0],resize_width*resize_height) # resize
                else:
                    val_imgs = val_imgs.reshape(FLAGS.batch_size,FLAGS.img_width*FLAGS.img_height)
                
                #convert labels to angle and displacement
                val_lbls = np.zeros((val_imgs.shape[0],2))
                for i in range(val_imgs.shape[0]):
                    val_lbls[i,:] = read_labels(val_lbls_id[i])
                
                validation_accuracy,  validation_loss, val_summary_str = sess.run([accuracy, val_cost, validation_summary],
                                                            feed_dict={x: val_imgs, y_: val_lbls})
                
                loss = cost.eval(feed_dict={x:train_imgs,y_:train_lbls})
    
                print('Step : {} - Loss : {} - Accuracy on Validation Set : {}'.format(step, loss, validation_accuracy))
                validation_writer.add_summary(val_summary_str, step)

            
            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + "_train", 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)



        # Calculate Accuracy
        print('=============== Testing ===============')

        # loop through validation data in batches
        num_test_cycles = int(math.ceil(num_val_samples/FLAGS.batch_size))
        good = 0
        batch_count = 0
        for epoch in range(num_test_cycles):
            
            # get batch of images
            test_imgs,test_lbls_id = sess.run([test_image,test_label])
            
            # convert and reshape to match network
            test_imgs = test_imgs.astype(np.float32)
            
            if resize_flag:
                test_imgs = test_imgs.reshape(test_imgs.shape[0],resize_width*resize_height) # resize
            else:
                test_imgs = test_imgs.reshape(FLAGS.batch_size,FLAGS.img_width*FLAGS.img_height)
            
            #convert labels to angle and displacement
            test_lbls = np.zeros((test_imgs.shape[0],2))
            for i in range(test_imgs.shape[0]):
                test_lbls[i,:] = read_labels(test_lbls_id[i])

            test_accuracy, test_summary_str = sess.run([accuracy, test_summary], feed_dict={x: test_imgs, y_: test_lbls, train_flag:0})
            good += test_accuracy
            #good += accuracy.eval(feed_dict={x:test_imgs,y_:test_lbls,train_flag:0})

            # Get a single image and label for displaying results
            if resize_flag:
                print('Epoch', epoch+1, 'of', math.ceil(num_val_samples/FLAGS.batch_size),
                      ': Example Label = ', test_lbls[0],' = ',
                      y_conv.eval(feed_dict={x:test_imgs[0].reshape(1,resize_width*resize_height),y_:test_lbls[0]}), ' = Prediction',
                      ': Correct', correct_prediction.eval(feed_dict={x:test_imgs[0].reshape(1,resize_width*resize_height),y_:test_lbls[0]}))
            else:
                print('Epoch', epoch+1, 'of', math.ceil(num_val_samples/FLAGS.batch_size),
                      ': Example Label = ', test_lbls[0],' = ',
                      y_conv.eval(feed_dict={x:test_imgs[0].reshape(1,FLAGS.img_width*FLAGS.img_height),y_:test_lbls[0]}), ' = Prediction',
                      ': Correct', correct_prediction.eval(feed_dict={x:test_imgs[0].reshape(1,FLAGS.img_width*FLAGS.img_height),y_:test_lbls[0]}))


            test_writer.add_summary(test_summary_str, batch_count)
            batch_count += 1

            
        print('Batch Test Accuracy: %g'%(good/(num_test_cycles)))

        train_writer.close()
        validation_writer.close()
        test_writer.close()
        
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run(main=main)
