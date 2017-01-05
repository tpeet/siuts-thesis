import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import siuts
import glob

num_epochs = 4
batch_size = 128


def main():
    start = time.time()
    num_channels = 1
    num_labels = len(siuts.species_list)
    image_size = siuts.resized_segment_size
    num_files = len(glob.glob1(siuts.dataset_dir, "{}-training*".format(siuts.species_list[0])))

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(1337)
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels), name="train_dataset_placeholder")
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name="train_labels_placeholder")
        tf_test_dataset = tf.placeholder(tf.float32,
                                         shape=(siuts.test_batch_size, image_size, image_size, num_channels),
                                         name="test_dataset_placeholder")
        tf_one_prediction = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels),
                                           name="tf_one_prediction")

        def conv2d(name, data, kernel_shape, bias_shape, stride=1):
            with tf.variable_scope(name) as scope:
                weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(0.0, 0.05))
                biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(1.0))

                conv = tf.nn.conv2d(data, weights, [1, stride, stride, 1], padding='SAME', name="conv")
                pre_activation = tf.nn.bias_add(conv, biases)
                activation = tf.nn.elu(pre_activation, name="elu")
                return activation

        def fully_connected(name, data, weights_shape, bias_shape, dropout):
            with tf.variable_scope(name) as scope:
                weights = tf.get_variable("weights", weights_shape, initializer=tf.random_normal_initializer(0.0, 0.05))
                biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(1.0))
                activation = tf.nn.elu(tf.nn.bias_add(tf.matmul(data, weights), biases), name="elu")
                return tf.nn.dropout(activation, dropout, name="dropout")

        def model(data, input_dropout, fc_dropout):
            data = tf.nn.dropout(data, input_dropout)
            # Conv1
            print data
            conv = conv2d("conv1", data, [5, 5, 1, 32], [32], 2)
            pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
            print pool

            # Conv2
            conv = conv2d("conv2", pool, [5, 5, 32, 64], [64])
            pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            print pool

            # Conv3
            conv = conv2d("conv3", pool, [3, 3, 64, 128], [128])
            pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
            print pool

            # Fully connected 1
            shape = pool.get_shape().as_list()
            reshaped_layer = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

            fc = fully_connected("fc1", reshaped_layer, [shape[1] * shape[2] * shape[3], 256], [256], fc_dropout)

            # Fully connected 2
            fc = fully_connected("fc2", fc, [256, 128], [128], fc_dropout)

            # output layer
            return fully_connected("output", fc, [128, num_labels], [num_labels], 1)

        logits = model(tf_train_dataset, 1, 0.8)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits + 1e-50, tf_train_labels))
        optimizer = tf.train.MomentumOptimizer(0.0005, 0.95, use_locking=False, name='Momentum',
                                               use_nesterov=True).minimize(loss)

        tf.get_variable_scope().reuse_variables()
        train_prediction = tf.nn.softmax(model(tf_train_dataset, 1, 1), name="sm_train")
        test_prediction = tf.nn.softmax(model(tf_test_dataset, 1, 1), name="sm_test")
        one_prediction = tf.nn.softmax(model(tf_one_prediction, 1, 1), name="sm_one")

    checkpoint_path = siuts.checkpoints_dir + "model.ckpt"

    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(siuts.checkpoints_dir, session.graph)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)
        tf.train.write_graph(session.graph_def, siuts.checkpoints_dir, "graph.pb", False)  # proto

        train_dataset = np.empty
        train_labels = np.empty
        current_file = 0
        current_epoch = 1
        step = 0
        while True:
            if (step * batch_size) % (siuts.samples_in_file * num_labels - batch_size) == 0:
                if current_epoch > num_epochs:
                    break
                del train_dataset
                del train_labels
                sys.stdout.write("Loading datasets nr " + str(current_file))
                sys.stdout.flush()
                counter = 0

                train_dataset = np.empty
                train_labels = np.empty
                for specimen in siuts.species_list:
                    new_data = siuts.load(
                        "{0}{1}-training_{2}.pickle".format(siuts.dataset_dir, specimen, current_file))
                    new_labels = np.empty(new_data.shape[0])
                    new_labels.fill(siuts.species_list.index(specimen))
                    if counter == 0:
                        train_dataset = new_data
                        train_labels = new_labels
                    else:
                        train_dataset = np.vstack((train_dataset, new_data))
                        train_labels = np.concatenate((train_labels, new_labels))
                    counter += 1

                    sys.stdout.write(".")
                    sys.stdout.flush()

                print
                current_file += 1
                if current_file >= num_files - 1:
                    current_file = 0
                    current_epoch += 1
                train_dataset, _, train_labels, _ = train_test_split(train_dataset, siuts.reformat_labels(train_labels),
                                                                     test_size=0, random_state=1337)
            offset = (step * batch_size) % (num_labels * siuts.samples_in_file - batch_size)
            sys.stdout.write(".")
            sys.stdout.flush()

            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 25 == 0:
                batch_acc = siuts.accuracy(predictions, batch_labels)

                if step % 250 == 0:
                    saver.save(session, checkpoint_path, global_step=step)

                print '%d - Minibatch loss: %f | Minibatch accuracy: %.1f%%' % (step, l, batch_acc)
            step += 1

        saver.save(session, checkpoint_path, global_step=step)

    print "Training took " + str(time.time() - start) + " seconds"


if __name__ == "__main__":
    main()
