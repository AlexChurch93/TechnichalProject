import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

tfrecords_filename = 'tap_images_train.tfrecord'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
sess = tf.Session()

i=0
for string_record in record_iterator:
    i+=1
    
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)


    # Get the features you stored (change to match your tfrecord writing code)
    height = int(example.features.feature['image/height']
                                 .int64_list
                                 .value[0])

    width = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])

    img_encoded = (example.features.feature['image/encoded']
                                  .bytes_list
                                  .value[0])

    img_format = (example.features.feature['image/format']
                                  .bytes_list
                                  .value[0])

    label = int(example.features.feature['image/class/label']
                                .int64_list
                                .value[0])

    #print(img_encoded)
    print(height)

    
    # Convert to a numpy array (change dtype to the datatype you stored)
    img_1d = np.fromstring(img_encoded, dtype=np.uint8) #convert tp nparray from byte string
    img_cv = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)     #convert to nparray in correct format with opencv
    img_pil = Image.fromarray(img_cv, 'RGB')            #convert to image from nparray
    img_tf = tf.image.decode_image(img_encoded,3)       #convert to tensor from byte string

    
    #img_pil.show()

    # check types
    print(type(img_encoded))
    print(img_1d.shape)
    print(img_cv.shape)
    print(sess.run(tf.size(img_tf)))

    
    if(i>5):
        break

sess.close()

