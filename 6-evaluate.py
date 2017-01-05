import operator
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
import siuts

siuts.create_dir(siuts.frozen_graphs_dir)

# How many results from saved checkpoints are used in calculating the mean accuracy metrics
nr_in_mean = 10

# Location to input graph
input_graph = siuts.checkpoints_dir + "graph.pb"

# TensorFlow saver file to load
input_saver = ""

# Whether the input files are in binary format
input_binary = True

# The name of the output nodes, comma separated
output_node_names = "sm_one,tf_one_prediction,sm_test,test_dataset_placeholder"

# The name of the master restore operator
restore_op_name = "save/restore_all"

# The name of the tensor holding the save path
filename_tensor_name = "save/Const:0"

# Whether to remove device specifications.
clear_devices = True

# comma separated list of initializer nodes to run before freezing
initializer_nodes = ""


def get_accuracies(graph_path, validation_data, validation_labels, recording_ids):
    acc_obj = siuts.Accuracy()
    with tf.Session() as persisted_sess:
        with gfile.FastGFile(graph_path, 'rb') as opened_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(opened_file.read())
            persisted_sess.graph.as_default()
            tf_test_dataset = tf.placeholder(tf.float32, shape=(siuts.test_batch_size, 64, 64, 1))
            test_predictions_op = tf.import_graph_def(graph_def,
                                                      input_map={"test_dataset_placeholder:0": tf_test_dataset},
                                                      return_elements=['sm_test:0'])

        testing_predictions = np.empty
        for i in range(validation_data.shape[0] / siuts.test_batch_size):
            start = i * siuts.test_batch_size
            end = (i + 1) * siuts.test_batch_size
            if i == 0:
                testing_predictions = test_predictions_op[0].eval(
                    feed_dict={tf_test_dataset: validation_data[start:end]})
            else:
                testing_predictions = np.concatenate((testing_predictions, test_predictions_op[0].eval(
                    feed_dict={tf_test_dataset: validation_data[start:end]})))

        validation_labels = validation_labels[:testing_predictions.shape[0]]

    predictions = np.argmax(testing_predictions, 1)
    labels = np.argmax(validation_labels, 1)
    recording_ids = recording_ids[:testing_predictions.shape[0]]

    acc_obj.seg_acc = siuts.accuracy(testing_predictions, validation_labels)
    acc_obj.seg_auc = roc_auc_score(validation_labels, testing_predictions, average="weighted")
    acc_obj.seg_f1 = f1_score(labels, predictions, average='weighted')
    acc_obj.seg_conf_matrix = confusion_matrix(labels, predictions)

    file_predictions = []
    file_labels = []
    for rec_id in recording_ids:
        rec_predictions = []
        test_label = -1
        for i in range(len(recording_ids)):
            if recording_ids[i] == rec_id:
                rec_predictions.append(np.array(testing_predictions[i]))
                test_label = validation_labels[i]
        if len(rec_predictions) > 0:
            file_predictions.append(np.array(rec_predictions))
            file_labels.append(test_label)

    file_predictions_mean = []
    for prediction in file_predictions:
        prediction = np.array(prediction)
        file_predictions_mean.append(np.asarray(np.mean(prediction, axis=0)))

    total = 0
    for i in range(len(file_predictions_mean)):
        if np.argmax(file_predictions_mean[i]) == np.argmax(file_labels[i]):
            total += 1

    acc_obj.file_acc = float(total) / len(file_predictions_mean)
    file_predictions_mean = np.array(file_predictions_mean)
    file_labels = np.array(file_labels)

    rec_predictions = np.array([np.argmax(pred) for pred in file_predictions_mean])
    rec_labels = np.argmax(file_labels, 1)

    acc_obj.file_auc = roc_auc_score(file_labels, file_predictions_mean, average="weighted")
    acc_obj.file_f1 = f1_score(rec_labels, rec_predictions, average='weighted')

    rec_conf_matrix = confusion_matrix(rec_labels, rec_predictions)
    acc_obj.file_conf_matrix = rec_conf_matrix

    file_predictions_top = []
    for i in range(len(file_predictions_mean)):
        top_3 = []
        pred = np.copy(file_predictions_mean[i])
        for j in range(3):
            index = np.argmax(pred)
            top_3.append(index)
            pred[index] = -1.0
        file_predictions_top.append(top_3)

    TPs = 0
    for i in range(len(file_predictions_mean)):
        if rec_labels[i] in file_predictions_top[i]:
            TPs += 1
    acc_obj.top3_acc = float(TPs) / len(file_predictions_mean)
    return acc_obj


def main():
    data = siuts.load(siuts.validation_data_filepath)
    labels = siuts.reformat_labels(siuts.load(siuts.validation_labels_filepath))
    rec_ids = siuts.load(siuts.validation_rec_ids_filepath)
    output_path = siuts.frozen_graphs_dir + "frozen_graph-{}.pb"
    accuracies_list = []
    for checkpoint in tf.train.get_checkpoint_state(siuts.checkpoints_dir).all_model_checkpoint_paths:
        step = checkpoint.split("-")[1]
        print "Evaluating for step {0}".format(step)
        freeze_graph.freeze_graph(input_graph, input_saver, input_binary, checkpoint, output_node_names,
                                  restore_op_name, filename_tensor_name, output_path.format(step), clear_devices,
                                  initializer_nodes)
        accuracies = get_accuracies(output_path.format(step), data, labels, rec_ids)
        accuracies.step = step
        accuracies_list.append(accuracies)

    with open(siuts.accuracies_filepath, 'wb') as f:
        pickle.dump(accuracies_list, f, protocol=-1)

    print
    print "Highest segments level F1 scores"
    accuracies_list.sort(key=operator.attrgetter('seg_f1'), reverse=True)
    for acc in accuracies_list[:nr_in_mean]:
        print "{:6} {:1.4f} ".format(acc.step, acc.seg_f1)
    print "Mean of {0} segment level F1 scores: {1}".format(nr_in_mean,
                                                            np.mean([x.seg_f1 for x in accuracies_list[:nr_in_mean]]))

    print
    print "Highest recording level F1 scores:"
    accuracies_list.sort(key=operator.attrgetter('file_f1'), reverse=True)
    for acc in accuracies_list[:nr_in_mean]:
        print "{:6} {:1.4f} ".format(acc.step, acc.file_f1)
    print "Mean of {0} file level F1 scores: {1}".format(nr_in_mean,
                                                         np.mean([x.file_f1 for x in accuracies_list[:nr_in_mean]]))

    print
    print "Highest top 3 accuracies"
    accuracies_list.sort(key=operator.attrgetter('top3_acc'), reverse=True)
    for acc in accuracies_list[:nr_in_mean]:
        print "{:6} {:1.4f} ".format(acc.step, acc.top3_acc)
    print "Mean of {0} highest top-3 accuracies: {1}".format(nr_in_mean, np.mean(
        [x.top3_acc for x in accuracies_list[:nr_in_mean]]))
    print


if __name__ == "__main__":
    main()
