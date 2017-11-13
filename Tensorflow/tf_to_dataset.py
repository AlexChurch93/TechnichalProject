import tensorflow as tf

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)

dataset = dataset.map()     # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# Initialize `iterator` with training data.
training_filenames = ['tap_images_train.tfrecords']
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
