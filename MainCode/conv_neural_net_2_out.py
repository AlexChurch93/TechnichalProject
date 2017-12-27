import numpy as np
import tensorflow as tf
import math
np.set_printoptions(threshold=np.nan)

## 16560 total images
## 320*240 size
## 30% validation
## approx 11592 training images
## approc 4968 validation images

print('=============== Initialising ===============')
# 70-30 split
#train_tfrecords_filename = 'tfrecord_data/tap_images_train.tfrecord'
#validation_tfrecords_filename = 'tfrecord_data/tap_images_validation.tfrecord'

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

print(num_train_samples)
print(num_val_samples)


# Read in the label file once and build a list of line offsets
lookup_filename = 'tfrecord_data/labels.txt'
lookup_file = open(lookup_filename, "rb", 0)
line_offset = []
offset = 0
for line in lookup_file:
    line_offset.append(offset)
    offset += len(line)
lookup_file.seek(0)


n_classes = 2
hm_inbatch = 64
hm_epochs = 100 #cycles = feed forward + backprop

#height x width 320x240=76800
#x = tf.placeholder(tf.float32,[None,76800])
x = tf.placeholder(tf.float32,[None,4800]) # resize
y = tf.placeholder(tf.int32) 

## ====================== Decoding images from TFRecord ======================
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
    image = tf.image.resize_images(image,[60,80]) # resize

    images, labels = tf.train.shuffle_batch([image, label],
                                             batch_size=hm_inbatch,
                                             capacity=30,
                                             num_threads=2,
                                             min_after_dequeue=10,
                                             allow_smaller_final_batch=True)
    return images, labels


## ====================== Building the Network ======================
def read_labels(n):
    lookup_file.seek(line_offset[n])
    string_label = lookup_file.readline().decode().rstrip() # read line, decode, remove \n
    string_label = string_label[string_label.index(':')+1:]
    string_label = string_label.split('_')
    int_label = list(map(int, string_label))
    return int_label

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

num_fc1_nodes = 1024
num_fc2_nodes = 512
weight_mean = 0
weight_stdv = 0.1

weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32],weight_mean,weight_stdv)),
           'W_conv2':tf.Variable(tf.random_normal([5,5,32,64],weight_mean,weight_stdv)),
           #'W_fc1':tf.Variable(tf.random_normal([60*80*64,num_fc1_nodes])),
           'W_fc1':tf.Variable(tf.random_normal([15*20*64,num_fc1_nodes],weight_mean,weight_stdv)), # resize
           'W_fc2':tf.Variable(tf.random_normal([num_fc1_nodes,num_fc2_nodes],weight_mean,weight_stdv)),
           'out':tf.Variable(tf.random_normal([num_fc2_nodes,n_classes],weight_mean,weight_stdv))}

biases = {'b_conv1':tf.Variable(tf.random_normal([32],weight_mean,weight_stdv)),
          'b_conv2':tf.Variable(tf.random_normal([64],weight_mean,weight_stdv)),
          'b_fc1':tf.Variable(tf.random_normal([num_fc1_nodes],weight_mean,weight_stdv)),
          'b_fc2':tf.Variable(tf.random_normal([num_fc2_nodes],weight_mean,weight_stdv)),
          'out':tf.Variable(tf.random_normal([n_classes],weight_mean,weight_stdv))}


def convolutional_neural_network(x):
    
    #x = tf.reshape(x,shape=[-1,240,320,1])
    x = tf.reshape(x,shape=[-1,60,80,1]) # resize

    conv1 = tf.nn.relu( conv2d(x,weights['W_conv1'])+ biases['b_conv1'] )
    conv1 = maxpool2d(conv1)

    conv2 =  tf.nn.relu( conv2d(conv1,weights['W_conv2']) + biases['b_conv2'] )
    conv2 = maxpool2d(conv2)

    #fc1 = tf.reshape(conv2,[-1,60*80*64])
    fc1 = tf.reshape(conv2,[-1,15*20*64]) # resize
    fc1 = tf.nn.relu(tf.matmul(fc1,weights['W_fc1'])+biases['b_fc1'])

    fc2 = tf.nn.relu(tf.matmul(fc1,weights['W_fc2'])+biases['b_fc2'])

    output = tf.matmul(fc2,weights['out'])+biases['out']
                    
    return output


## ====================== Training the Network ======================
train_filename_queue = tf.train.string_input_producer([train_tfrecords_filename])# add num_epochs to limit amount of time through data
image,label = read_and_decode(train_filename_queue)

saver = tf.train.Saver()
tf_log = './logs/tf.log'

print('=============== Training ===============')

def train_neural_network(x):

    prediction = convolutional_neural_network(x)
    cost = tf.losses.mean_squared_error(y,prediction)
    
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    
    # Passing global_step to minimize() will increment it at each step.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
    #optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    # The op for initializing the variables. 
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
         file_writer = tf.summary.FileWriter('/logs', sess.graph)

         sess.run(init_op)

         # make sure variables are intialised
         try:
            sess.run(tf.assert_variables_initialized())
         except tf.errors.FailedPreconditionError:
            print(sess.run(tf.report_uninitialized_variables()))
            raise RuntimeError("Not all variables initialized!")
        
         coord = tf.train.Coordinator()
         threads = tf.train.start_queue_runners(coord=coord)

         # start training after previous ## 
         try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
         except:
            epoch = 1
         #epoch = 1
         
         while epoch <= hm_epochs:
             if epoch != 1:
                saver.restore(sess,"./logs/cnn-80-20.ckpt")
             epoch_loss = 1

             # get batch of images
             batch_imgs,batch_lbls = sess.run([image,label])

             # convert and reshape to match network
             batch_imgs = batch_imgs.astype(np.float32)
             #batch_imgs = batch_imgs.reshape(hm_inbatch,76800)
             batch_imgs = batch_imgs.reshape(hm_inbatch,4800) # resize

             #convert labels to angle and displacement
             new_labels = np.zeros((hm_inbatch,2))
             for i in range(hm_inbatch):
                 new_labels[i,:] = read_labels(batch_lbls[i])
             
             _,c = sess.run([optimizer,cost], feed_dict={x:batch_imgs,y:new_labels})
             epoch_loss += c

             saver.save(sess, "./logs/cnn-80-20.ckpt")
             print('Epoch', epoch, 'of',hm_epochs,': Loss =',epoch_loss)
             with open(tf_log,'a') as f:
                 f.write(str(epoch)+'\n') 
             epoch +=1

train_neural_network(x)

## ====================== Testing the network on individual samples ======================
print('=============== Testing ===============')

val_filename_queue = tf.train.string_input_producer([validation_tfrecords_filename], num_epochs=1)# add num_epochs to limit amount of time through data
val_image,val_label = read_and_decode(val_filename_queue)


## ====================== Testing the network on Batch samples ======================
def batch_test_neural_network():

    prediction = convolutional_neural_network(x)
    # The op for initializing the variables. 
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"./logs/cnn-80-20.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
        
        # Calculate the accuracy
        correct = tf.reduce_min( tf.cast((tf.equal(tf.cast(tf.round(prediction),tf.int32), y)),tf.float32),1 )
        accuracy_sum = tf.reduce_sum(tf.cast(correct,'float'))

        leftover = int(num_val_samples%hm_inbatch)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # loop through validation data in batches
        good = 0
        for epoch in range(math.ceil(num_val_samples/hm_inbatch)):

            # get batch of images
            val_imgs,val_lbls = sess.run([val_image,val_label])

            # convert and reshape to match network
            val_imgs = val_imgs.astype(np.float32)
            #val_imgs = val_imgs.reshape(val_imgs.shape[0],76800)
            val_imgs = val_imgs.reshape(val_imgs.shape[0],4800) # resize
            
            #convert labels to angle and displacement
            new_labels = np.zeros((val_imgs.shape[0],2))
            for i in range(val_imgs.shape[0]):
                new_labels[i,:] = read_labels(val_lbls[i])

            good += accuracy_sum.eval(feed_dict={x:val_imgs,y:new_labels})

            #print(val_imgs[0])

            # Get a single image and label for displaying
            print('Epoch', epoch+1, 'of', math.ceil(num_val_samples/hm_inbatch),
                  ': Example Label = ', new_labels[0],' = ',
                  #tf.cast(tf.round(prediction),tf.int32).eval(feed_dict={x:val_imgs[0].reshape(1,4800),y:new_labels[0]}), ' = Prediction')
                  prediction.eval(feed_dict={x:val_imgs[0].reshape(1,4800),y:new_labels[0]}), ' = Prediction',
                  ': Correct = ',correct.eval(feed_dict={x:val_imgs[0].reshape(1,4800),y:new_labels[0]}),
                  ': Accuracy = ',accuracy_sum.eval(feed_dict={x:val_imgs[0].reshape(1,4800),y:new_labels[0]}),
                  ': Good = ',good)

    print('Batch Test Accuracy: %g'%(good/(num_val_samples)))


batch_test_neural_network()
