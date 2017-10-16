'''
input > weights > hidden layer 1 > activation function >
        weights > hidden layer 2 > activation function > weights > output layer

compare output to label > cost function (cross entropy)

optimisation function > minimise cost (AdamOptimizer... SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn 

mnist = input_data.read_data_sets("/tmp/dara/", one_hot=True)

''' 10 classes 0-9

0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]...

'''

#cycles = feed forward + backprop
hm_epochs = 10
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128

#height x width (28x28 = 784)
x = tf.placeholder('float',[None,n_chunks,chunk_size])
y = tf.placeholder('float')

def reccurent_neural_network_model(x):
    
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x,[1,0,2]) #format for tensorflow rnn_cell
    x = tf.reshape(x,[-1,chunk_size])
    x = tf.split(x,n_chunks,0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size) 
    
    outputs, states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    
    
    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output
    

def train_neural_network(x):
    prediction = reccurent_neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    # learning_rate = 0.001 default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())

         for epoch in range(hm_epochs):
             epoch_loss = 0
             for _ in range(int(mnist.train.num_examples/batch_size)):
                 
                 epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                 epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                 
                 _,c = sess.run([optimizer,cost], feed_dict={x:epoch_x,y:epoch_y})
                 
                 epoch_loss += c

             print('Epoch',epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            
         correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
         accuracy = tf.reduce_mean(tf.cast(correct,'float'))
         print('Acuracy:', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels}))
                 

train_neural_network(x)

