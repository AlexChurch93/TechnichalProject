import tensorflow as tf
import skimage.io as io
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import cv2

# Read in the file once and build a list of line offsets
lookup_filename = 'tfrecord_data/labels.txt'
lookup_file = open(lookup_filename, "rb", 0)
line_offset = []
offset = 0
for line in lookup_file:
    line_offset.append(offset)
    offset += len(line)
lookup_file.seek(0)


# 70-30 split
#train_tfrecords_filename = 'tfrecord_data/tap_images_train.tfrecord'
#validation_tfrecords_filename = 'tfrecord_data/tap_images_validation.tfrecord'

# 80-20 split
train_tfrecords_filename = 'tfrecord_data/tap_images_80-20_train.tfrecord'
validation_tfrecords_filename = 'tfrecord_data/tap_images_80-20_validation.tfrecord'

record_iterator = tf.python_io.tf_record_iterator(path=validation_tfrecords_filename)

num_train_samples=0
for string_record in record_iterator:
    num_train_samples+=1

hm_inbatch = 6
    
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

    image = tf.image.rgb_to_grayscale(image)
    image = tf.reshape(image, [240,320,1])
    image = tf.image.resize_images(image,[60,80])
    image = tf.image.per_image_standardization(image)

    images, labels = tf.train.shuffle_batch([image, label],
                                             batch_size=hm_inbatch,
                                             capacity=30,
                                             num_threads=8,
                                             min_after_dequeue=10)
    return images, labels


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

filename_queue = tf.train.string_input_producer([train_tfrecords_filename])# add num_epochs to limit amount of time through data

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
         
##        for j in range(6):
##            plt.subplot(2, 3, j+1)
##            plt.imshow(img[j, ...], cmap='gray')
##            plt.title(lbl[j])
##            
##        plt.show()

##        cv2.imshow('image',img[0])
##        cv2.waitKey(0)

        # class as one hot array
        nb_classes = 1380
        targets = np.array(lbl).reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        #print(one_hot_targets)

        # two classes, angle and displacement
        #int_label = read_labels(10)
        #print(int_label)

        new_labels = np.zeros((hm_inbatch,2))
        for i in range(hm_inbatch):
            int_label = read_labels(lbl[i])
            new_labels[i,:] = int_label

        print(new_labels)

            
            
        #print(lbl)
        

    coord.request_stop()
    coord.join(threads)






