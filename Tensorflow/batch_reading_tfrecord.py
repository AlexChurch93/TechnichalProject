import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = 'tap_images_train.tfrecord'


with tf.Session() as sess:
    

    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/height': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/width': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=keys_to_features)


    image = tf.image.decode_image(features['image/encoded'],3)
    img_format = tf.decode_raw(features['image/format'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)

    image = tf.reshape(image, [240,320,3])

    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        print(batch_index)
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j+1)
            plt.imshow(img[j, ...])
            plt.title('cat' if lbl[j]==0 else 'dog')
        plt.show()
        
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()
