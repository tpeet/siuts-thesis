import tensorflow as tf
from tensorflow.python.platform import gfile
from operator import itemgetter
import numpy as np
import sys
import os
import warnings
import siuts

warnings.filterwarnings('ignore')


def reshape_segments(segments):
    segments = np.array(segments)
    return np.reshape(segments, [segments.shape[0], segments.shape[1], segments.shape[2], 1])


def main(wav_path, graph_path="siuts_model.pb"):
    if not os.path.isfile(graph_path):
        print "No model file found. Please specify trained model path as a second command line argument"
        return

    segments = reshape_segments(siuts.segment_wav(wav_path))
    with tf.Session() as persisted_sess:
        with gfile.FastGFile(graph_path, 'rb') as opened_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(opened_file.read())
            persisted_sess.graph.as_default()
            tf_sample = tf.placeholder(tf.float32, shape=(1, 64, 64, 1))
            predictions_op = tf.import_graph_def(graph_def,
                                                 input_map={"tf_one_prediction:0": tf_sample},
                                                 return_elements=['sm_one:0'])

        predictions = np.empty
        for i in range(len(segments)):
            if i == 0:
                predictions = predictions_op[0].eval(feed_dict={tf_sample: [segments[i]]})
            else:
                predictions = np.concatenate((predictions, predictions_op[0].eval(
                    feed_dict={tf_sample: [segments[i]]})))

    first_predictions = np.argmax(predictions, 1)
    print "Prediction for each segment:"
    for i, prediction in enumerate(first_predictions):
        print "{:^25} ({:05.2f}%)".format(siuts.species_list[prediction].replace("_", " "), max(predictions[i]) * 100)

    averaged_predictions = np.mean(predictions, axis=0)
    predictions_dict = {}
    for i, prediction in enumerate(averaged_predictions):
        predictions_dict[siuts.species_list[i]] = prediction
    print
    print "Predictions:"
    for species, probability in sorted(predictions_dict.items(), key=itemgetter(1), reverse=True):
        print "{:^25} ({:05.2f}%)".format(species.replace("_", " "), probability * 100)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print "You have to add path to the *.wav file as a command line argument"
