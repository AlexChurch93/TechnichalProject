import tensorflow as tf

dummy_labels     = tf.constant([[1, 2], [4, 5], [7, 8]], tf.float32)
dummy_prediction = tf.constant([[1, 2], [4, 5], [5, 8]], tf.float32)

abs_diff   = tf.abs(dummy_labels-dummy_prediction)
theta_diff = abs_diff[:,0]
r_diff     = abs_diff[:,1]

correct_theta = tf.less_equal(theta_diff, 1)
correct_r = tf.less_equal(r_diff, 0.25)

combined = tf.concat(tf.transpose([correct_theta, correct_r]), 0)
correct = tf.reduce_min(tf.cast(combined, tf.float32),1)

accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        
with tf.Session() as sess:
    
    diff = sess.run(abs_diff)
    print(diff)

    thetadiff = sess.run(theta_diff)
    print(thetadiff)

    rdiff = sess.run(r_diff)
    print(rdiff)

    corrtheta = sess.run(correct_theta)
    print(corrtheta)

    corrr = sess.run(correct_r)
    print(corrr)

    comb = sess.run(combined)
    print(comb)

    corr = sess.run(correct)
    print(corr)

    acc = sess.run(accuracy)
    print(acc)

    
