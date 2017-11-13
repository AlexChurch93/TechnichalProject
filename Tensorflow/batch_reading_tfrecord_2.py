import tensorflow as tf
import skimage.io as io
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt

tfrecords_filename = 'tap_images_train.tfrecord'

tfrecords_filename = 'tap_images_train.tfrecord'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

num_train_samples=0
for string_record in record_iterator:
    num_train_samples+=1

print(num_train_samples)
    
## 16560 total images
## 320*240 size
## 30% validation
## approx 11592 training images
## approc 4968 test images

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


    image = tf.reshape(image, [240,320,3])

    images, labels = tf.train.shuffle_batch([image, label],
                                             batch_size=64,
                                             capacity=30,
                                             num_threads=8,
                                             min_after_dequeue=10)
    return images, labels




filename_queue = tf.train.string_input_producer([tfrecords_filename])# add num_epochs to limit amount of time through data

# Even when reading in multiple threads, share the filename
# queue.
image,label = read_and_decode(filename_queue)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    #sess.run(init_op)
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    hm_batches = 1
    for batch_index in range(hm_batches):
        
        print('Batch: ' + str(batch_index))
        
        img,lbl = sess.run([image,label])
        img = img.astype(np.uint8)
        
        for j in range(6):
            plt.subplot(2, 3, j+1)
            plt.imshow(img[j, ...])
            plt.title(lbl[j])
            
        plt.show()

        nb_classes = 1380
        targets = np.array(lbl).reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        print(one_hot_targets)


    coord.request_stop()
    coord.join(threads)






