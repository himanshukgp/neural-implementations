import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import numpy as np
import pickle
import os
import urllib.request
import argparse



def _get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images


def _load_data(filename):
    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _convert_images(raw_images)
    return images, cls

data_path = "data/CIFAR-10/"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--init', default=1, type=int)
parser.add_argument('--save-dir', default="./tensorflow/logdir3")
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
init = args.init
save_dir = args.save_dir


def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def _load_data(filename):
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    raw_float = np.array(raw_images, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])

    return images, cls


def load_training_data():
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    begin = 0

    for i in range(_num_files_train):
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)



images_train, cls_train, labels_train = load_training_data()



print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))



from sklearn.model_selection import train_test_split
trX, teX, trY, teY = train_test_split(images_train, labels_train, test_size=0.1, random_state=42)

print("Size of:")
print("- Training-set:\t\t{}".format(len(trX)))
print("- Test-set:\t\t{}".format(len(teX)))

img_size=32
num_channels=3
num_classes=10
test_size = 256


def model(X, w, w2, w3, w4, w5, w6, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 32, 32, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 16, 16, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 8, 8, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 8, 8, 256)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.dropout(l3a, p_keep_conv)
    
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4,                     # l4a shape=(?, 8, 8, 256)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1],              # l4 shape=(?, 4, 4, 256)
                        strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.reshape(l4, [-1, w5.get_shape().as_list()[0]])    # reshape to (?, 4096)
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5 = tf.nn.relu(tf.matmul(l4, w5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)
    
    l6 = tf.nn.relu(tf.matmul(l5, w6))
    l6 = tf.nn.dropout(l6, p_keep_hidden)

    pyx = tf.matmul(l6, w_o)
    return pyx



X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w = tf.get_variable("w", shape = [3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer())     # 3x3x1 conv, 64 outputs
w2 = tf.get_variable("w2", shape = [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())     # 3x3x32 conv, 128 outputs
w3 = tf.get_variable("w3", shape = [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())    # 3x3x32 conv, 256 outputs
w4 = tf.get_variable("w4", shape = [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())    # 3x3x32 conv, 256 outputs
w5 = tf.get_variable("w5", shape = [256 * 4 * 4, 1024], initializer=tf.contrib.layers.xavier_initializer()) # FC 128 * 4 * 4 inputs, 625 outputs
w6 = tf.get_variable("w6", shape = [1024, 1024], initializer=tf.contrib.layers.xavier_initializer())        # FC 1024 inputs, 625 outputs
w_o = tf.get_variable("w_o", shape = [1024, 10], initializer=tf.contrib.layers.xavier_initializer())         # FC 625 inputs, 10 outputs (labels)

if init == 2:
    w = tf.get_variable("w", shape = [3, 3, 3, 64], initializer=tf.keras.initializers.he_uniform())     # 3x3x1 conv, 64 outputs
    w2 = tf.get_variable("w2", shape = [3, 3, 64, 128], initializer=tf.keras.initializers.he_uniform())     # 3x3x32 conv, 128 outputs
    w3 = tf.get_variable("w3", shape = [3, 3, 128, 256], initializer=tf.keras.initializers.he_uniform())    # 3x3x32 conv, 256 outputs
    w4 = tf.get_variable("w4", shape = [3, 3, 256, 256], initializer=tf.keras.initializers.he_uniform())    # 3x3x32 conv, 256 outputs
    w5 = tf.get_variable("w5", shape = [256 * 4 * 4, 1024], initializer=tf.keras.initializers.he_uniform()) # FC 128 * 4 * 4 inputs, 625 outputs
    w6 = tf.get_variable("w6", shape = [1024, 1024], initializer=tf.keras.initializers.he_uniform())        # FC 1024 inputs, 625 outputs
    w_o = tf.get_variable("w_o", shape = [1024, 10], initializer=tf.keras.initializers.he_uniform())         # FC 625 inputs, 10 outputs (labels)


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w5, w6, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
tf.summary.scalar("cost", cost)

ckpt_dir = "./ckpt_dir7"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# This variable won't be stored, since it is declared after tf.train.Saver()
non_storable_variable = tf.Variable(777)



with tf.Session() as sess:

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(save_dir, sess.graph) # for 1.0

    # you need to initialize all variables
    tf.global_variables_initializer().run()

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    start = global_step.eval() # get last global_step
    print("Start from:", start)

    for i in range(start, 100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        global_step.assign(i).eval() # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))

