# python3 run.py > run.log
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np
import sys

import config
import lenet


def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, config.batchSize):
        batch_x, batch_y = x_data[offset:offset +
                                  config.batchSize], y_data[offset:offset+config.batchSize]
        accuracy = sess.run(accuracy_operation, feed_dict={
                            x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


if __name__ == '__main__':
    print("TensorFlow version", tf.__version__)
    print()

    mnist = input_data.read_data_sets(config.datasetDir, reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    x_test, y_test = mnist.test.images, mnist.test.labels

    print()
    print("Image shape: {}".format(x_train[0].shape))
    print()
    print("Training set length: {}".format(len(x_train)))
    print("Validation set length: {}".format(len(x_validation)))
    print("Test set length: {}".format(len(x_test)))

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_validation = np.pad(
        x_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    print()
    print("Image shape padded: {}".format(x_train[0].shape))

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 10)

    logits = lenet.lenet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Start training", file=sys.stderr)
        print("Start training")
        print()

        sess.run(tf.global_variables_initializer())
        num_examples = len(x_train)

        for i in range(config.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for offset in range(0, num_examples, config.batchSize):
                end = offset + config.batchSize
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={
                         x: batch_x, y: batch_y})

            validation_accuracy = evaluate(x_validation, y_validation)
            print(i, validation_accuracy)
            print(i, validation_accuracy, file=sys.stderr)

        saver.save(sess, config.model)
        print("Finish training")
        print("Finish training", file=sys.stderr)
        print()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(config.modelDir))
        test_accuracy = evaluate(x_test, y_test)
        print("Accuracy with {} test data = {:.3f}".format(
            len(x_test), test_accuracy))
        print("Accuracy with {} test data = {:.3f}".format(
            len(x_test), test_accuracy), file=sys.stderr)
