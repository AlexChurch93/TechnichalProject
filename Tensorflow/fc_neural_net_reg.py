import numpy as np
import tensorflow as tf
import cv2
from sys import stdout

## 16560 total images
## 320*240 size
## 30% validation
## approx 11592 training images
## approc 4968 validation images

# 70-30 split
#train_tfrecords_filename = 'tap_images_train.tfrecord'
#validation_tfrecords_filename = 'tap_images_validation.tfrecord'

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

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 1
hm_inbatch = 8

#height x width 320x240=76800
x = tf.placeholder(tf.float32,[None,76800])
y = tf.placeholder(tf.int32)

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
                                             batch_size=hm_inbatch,
                                             capacity=30,
                                             num_threads=2,
                                             min_after_dequeue=10)
    return images, labels




def neural_network_model(data):
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([76800,n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input_data*weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1);

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2);

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3);

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


filename_queue = tf.train.string_input_producer([train_tfrecords_filename])# add num_epochs to limit amount of time through data

# Even when reading in multiple threads, share the filename
# queue.
image,label = read_and_decode(filename_queue)

# The op for initializing the variables. (doesn't work?)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


def train_neural_network(x):

    prediction = neural_network_model(x)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    cost = tf.losses.mean_squared_error(y,prediction)


    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    
    # Passing global_step to minimize() will increment it at each step.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
    #optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

    #cycles = feed forward + backprop
    hm_epochs = 100

    with tf.Session() as sess:

         sess.run(tf.global_variables_initializer())
         try:
            sess.run(tf.assert_variables_initialized())
         except tf.errors.FailedPreconditionError:
            print(sess.run(tf.report_uninitialized_variables()))
            raise RuntimeError("Not all variables initialized!")
        
         coord = tf.train.Coordinator()
         threads = tf.train.start_queue_runners(coord=coord)

         for epoch in range(hm_epochs):
             epoch_loss = 0

             # get batch of images
             batch_imgs,batch_lbls = sess.run([image,label])

             # convert and reshape to match network
             batch_imgs = batch_imgs.astype(np.float32)
             batch_imgs = batch_imgs.reshape(hm_inbatch,76800)

             #convert labels 
             batch_lbls = batch_lbls.astype(np.int32).reshape(hm_inbatch,1)
            
             _,c = sess.run([optimizer,cost], feed_dict={x:batch_imgs,y:batch_lbls})
             epoch_loss += c

             print('Epoch',epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

         # Calculate the accuracy
         correct = tf.equal(tf.cast(tf.round(prediction),tf.int32), tf.round(y))
         accuracy_sum = tf.reduce_mean(tf.cast(correct,'float'))

         good = 0
         i=0
         max_iter = 100;
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

             val_label = np.array(val_label).astype(np.int32).reshape(1,1);
             
             good += accuracy_sum.eval(feed_dict={x:val_img,y:val_label})
             print(i,': ', val_label,' = ', (np.round(prediction.eval(feed_dict={x:val_img,y:val_label})).astype(np.int32)), 'Correct: ', good)
             #print(i)
             #print('\rTested  '+str(i)+' of ' + str(num_val_samples) + ' : ' + str(val_label) + ' = ' + str((np.round(prediction.eval(feed_dict={x:val_img,y:val_label})).astype(np.int32))), end='\r')

             
             if(i>=max_iter):
                 break
        

         #print('Test Accuracy: %g'%(good/num_val_samples))
         print('\nTest Accuracy: %g'%(good/max_iter))     


train_neural_network(x)

























