import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.python.platform import gfile
import siuts
import sys


def main(graph_path):
    num_classes = len(siuts.species_list)
    testing_data = siuts.load(siuts.testing_data_filepath)
    test_labels = siuts.reformat_labels(siuts.load(siuts.testing_labels_filepath))
    recording_ids = siuts.load(siuts.testing_rec_ids_filepath)

    print "Getting predictions..."

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
        for i in range(testing_data.shape[0] / siuts.test_batch_size):
            start = i * siuts.test_batch_size
            end = (i + 1) * siuts.test_batch_size
            if i == 0:
                testing_predictions = test_predictions_op[0].eval(feed_dict={tf_test_dataset: testing_data[start:end]})
            else:
                testing_predictions = np.concatenate((testing_predictions, test_predictions_op[0].eval(
                    feed_dict={tf_test_dataset: testing_data[start:end]})))

        test_labels = test_labels[:testing_predictions.shape[0]]

    predictions = np.argmax(testing_predictions, 1)
    labels = np.argmax(test_labels, 1)
    recording_ids = recording_ids[:testing_predictions.shape[0]]

    print
    print "----Results on segments level----"

    print "Accuracy: " + str(siuts.accuracy(testing_predictions, test_labels))
    print "AUC score (weighted): " + str(roc_auc_score(test_labels, testing_predictions, average="weighted"))
    print "F1 score (weighted): " + str(f1_score(labels, predictions, average='weighted'))

    seg_conf_matrix = confusion_matrix(labels, predictions)
    print
    print "Labels     Species name      Recall Precis. F1-score"
    print "----------------------------------------------------"
    for i in range(num_classes):
        TP = seg_conf_matrix[i][i]
        SUM = np.sum(seg_conf_matrix[i])
        if SUM == 0:
            recall = 0
            precision = 0
            f1 = 0
        else:
            recall = float(TP) / SUM * 100
            try:
                precision = float(TP) / np.sum(seg_conf_matrix[:, i]) * 100
            except ZeroDivisionError:
                precision = 0

            try:
                f1 = 2 * (recall * precision) / (recall + precision)
            except ZeroDivisionError:
                f1 = 0

        print "{:2d} {:^25} {:05.2f} | {:05.2f} | {:05.2f}".format(i, siuts.species_list[i], recall, precision, f1)
    print

    print "----Results on recordings level----"
    file_predictions = []
    file_labels = []
    for rec_id in recording_ids:
        rec_predictions = []
        test_label = []
        for i in range(len(recording_ids)):
            if recording_ids[i] == rec_id:
                rec_predictions.append(np.array(testing_predictions[i]))
                test_label = test_labels[i]
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

    file_acc = float(total) / len(file_predictions_mean)
    print "Accuracy: " + str(file_acc)

    file_predictions_mean = np.array(file_predictions_mean)
    file_labels = np.array(file_labels)
    rec_predictions = np.array([np.argmax(pred) for pred in file_predictions_mean])
    rec_labels = np.argmax(file_labels, 1)

    print "AUC score (weighted): " + str(roc_auc_score(file_labels, file_predictions_mean, average="weighted"))
    print "F1 score (weighted): " + str(f1_score(rec_labels, rec_predictions, average='weighted'))

    rec_conf_matrix = confusion_matrix(rec_labels, rec_predictions)
    print
    print "Prediction accuracy on recordings level"
    print "Labels     Species name       Recall Precis. F1-score"
    print "-----------------------------------------------------"
    for i in range(num_classes):
        TP = rec_conf_matrix[i][i]
        SUM = np.sum(rec_conf_matrix[i])
        try:
            recall = float(TP) / SUM * 100
        except ZeroDivisionError:
            recall = 0

        try:
            precision = float(TP) / np.sum(rec_conf_matrix[:, i]) * 100
        except ZeroDivisionError:
            precision = 0

        try:
            f1 = 2 * (recall * precision) / (recall + precision)
        except ZeroDivisionError:
            f1 = 0
        print "{:2d} {:^25} {:6.2f} | {:6.2f} | {:6.2f}".format(i, siuts.species_list[i], recall, precision, f1)

    print

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
    top3_acc = float(TPs) / len(file_predictions_mean)
    print "Top-3 accuracy: " + str(top3_acc)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print "You have to add path to the frozen graph as a command line argument"
