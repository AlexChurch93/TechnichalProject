import tensorflow as tf
import numpy as np
import math


with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "C:\\Users\\alexc\\Documents\\University Course Documents\\Year 4\\Technical Project\\TestLogs\\logs\\exp_bs_64_lr_0.0008_ms_8000_train\\model.ckpt")
