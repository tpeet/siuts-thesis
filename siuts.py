import os
import wave
import pylab
import numpy as np
from numpy.lib import stride_tricks
from sklearn.preprocessing import scale
import scipy.misc
import pickle

#
# General settings for selecting species and pre-processing properties
#

# List of species used in classification task. The index of the list is the label for each species
species_list = ['Parus_major', 'Coloeus_monedula', 'Corvus_cornix', 'Fringilla_coelebs',
                'Erithacus_rubecula', 'Phylloscopus_collybita', 'Turdus_merula', 'Cyanistes_caeruleus',
                'Emberiza_citrinella', 'Chloris_chloris', 'Turdus_philomelos', 'Phylloscopus_trochilus',
                'Sylvia_borin', 'Apus_apus', 'Passer_domesticus', 'Luscinia_luscinia', 'Sylvia_atricapilla',
                'Ficedula_hypoleuca', 'Sylvia_communis', 'Carpodacus_erythrinus']

# PlutoF species ID-s had to be handpicked, because the names didn't always correspond to the ones in Xeno-Canto.
# Each ID in this list corresponds to the species in species_list
plutoF_taxon_ids = [86560, 48932, 110936, 60814, 57887, 89499, 107910, 86555, 56209, 43289, 107914, 89514, 102321,
                    36397, 86608, 72325, 102319, 60307, 102323, 43434]

# Xeno-canto quality A is the best
acceptable_quality = ["A", "B"]

# Sample rate of the wave files
wav_framerate = 22050

# Frame size of the Fourier transform
fft_frame_size = 512

# Final segment size
resized_segment_size = 64

# Batch size in valuation phase
test_batch_size = 32

# How many samples of training data in one file. If lower than 16GB of RAM, a lower number should be used
samples_in_file = 4096

# Overlap by half of the segment size when performing segmentation
segmentation_hop_size = fft_frame_size / 4

# Directory path containing audio files, segments, etc...
data_dir = "data/"
# Directory where Xeno-canto audio files are downloaded
xeno_dir = data_dir + "xeno_recordings/"
# Directory where PlutoF audio file are downloade
plutoF_dir = data_dir + "plutof_recordings/"
# File path to the meta-data of xeno-canto recordings
xeno_metadata_path = data_dir + "xeno_metadata.pickle"
# File path to the meta-data of PlutoF recordings
plutof_metadata_path = data_dir + "plutof_metadata.pickle"
# Directory where xeno-canto converted WAV files are saved
xeno_wavs_dir = data_dir + "xeno_wavs/"
# Directory where PlutoF converted WAV files are saved
plutof_wavs_dir = data_dir + "plutof_wavs/"
# Directory where Xeno-canto segments are saved
xeno_segments_dir = data_dir + "xeno_segments/"
# Directory where PlutoF segments are saved
plutof_segments_dir = data_dir + "plutof_segments/"

# Directory where the input files for training and evaluation are saved
dataset_dir = data_dir + "dataset/"
# File path to the joined testing segments
testing_data_filepath = dataset_dir + "testing_data.pickle"
# File path to the joined testing labels
testing_labels_filepath = dataset_dir + "testing_labels.pickle"
# File path to the joined testing recording id's
testing_rec_ids_filepath = dataset_dir + "testing_rec_ids.pickle"
# File path to the joined validation segments
validation_data_filepath = dataset_dir + "validation_data.pickle"
# File path to the joined validation labels
validation_labels_filepath = dataset_dir + "validation_labels.pickle"
# File path to the joined validation recording id's
validation_rec_ids_filepath = dataset_dir + "validation_rec_ids.pickle"

# Directory where graph and checkpoint files are saved
checkpoints_dir = "checkpoints/"
# Directory where frozen graphs are located
frozen_graphs_dir = checkpoints_dir + "frozen_graphs/"
# File path to the accuracies objects for each step, where checkpoint was saved
accuracies_filepath = checkpoints_dir + "accuracies.pickle"


class Recording:
    segments_count = None

    def __init__(self, identifier, gen, sp, label, file_url):
        self.id = identifier
        self.gen = gen
        self.sp = sp
        self.label = label
        self.file_url = file_url

    def __repr__(self):
        return "id: {0}, name: {1}_{2}, label: {3}".format(self.id, self.gen, self.sp, self.label)

    def get_name(self):
        """Return the scientific name - <genus_species>"""
        return "{0}_{1}".format(self.gen, self.sp)

    def get_filename(self):
        """Return the filename withoud extension - <genus_species-id>"""
        return "{0}_{1}-{2}".format(self.gen, self.sp, self.id)


class Accuracy:
    def __init__(self):
        pass

    step = None
    seg_acc = None
    seg_auc = None
    seg_f1 = None
    seg_conf_matrix = None
    file_acc = None
    file_auc = None
    file_f1 = None
    file_conf_matrix = None
    top3_acc = None


def create_dir(path):
    (dirname, _) = os.path.split(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


def stft(sig, frame_size, overlap_fac=0.5, window=np.hanning):
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap_fac * frame_size))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1

    samples = np.append(samples, np.zeros(frame_size))

    frames = stride_tricks.as_strided(samples, shape=(cols, frame_size),
                                      strides=(samples.strides[0] * hop_size, samples.strides[0])).copy()
    frames *= win

    return np.fft.fft(frames)


def clean_spectrogram(transposed_spectrogram, coef=3):
    row_means = transposed_spectrogram.mean(axis=0)
    col_means = transposed_spectrogram.mean(axis=1)

    cleaned_spectrogram = []

    for col_index, column in enumerate(transposed_spectrogram):
        for row_index, pixel in enumerate(column):
            if pixel > coef * row_means[row_index] and pixel > coef * col_means[col_index]:
                cleaned_spectrogram.append(transposed_spectrogram[col_index])
                break
    return np.array(cleaned_spectrogram)


def scale_segments(segments):
    segment_size1 = len(segments[0])
    segment_size2 = len(segments[0][0])
    segment_count = len(segments)
    segments = np.reshape(segments, (segment_count, segment_size1 * segment_size2))
    scaled_segments = scale(segments, axis=1, with_mean=True, with_std=True, copy=True)
    return scaled_segments.reshape(segment_count, segment_size1, segment_size2, 1).tolist()


def segment_wav(wav_path):
    signal, fs = get_wav_info(wav_path)
    transposed_spectrogram = abs(stft(signal, fft_frame_size))[:, :fft_frame_size / 2]
    cleaned_spectrogram = clean_spectrogram(transposed_spectrogram)
    segments = []
    if cleaned_spectrogram.shape[0] > fft_frame_size / 2:
        for i in range(int(np.floor(cleaned_spectrogram.shape[0] / segmentation_hop_size - 1))):
            segment = cleaned_spectrogram[
                      i * segmentation_hop_size:i * segmentation_hop_size + cleaned_spectrogram.shape[1]]
            resized_segment = scipy.misc.imresize(segment,
                                                  (resized_segment_size, resized_segment_size),
                                                  interp='nearest')
            segments.append(resized_segment)
    return segments


def load(location):
    with open(location, 'rb') as opened_file:
        return pickle.load(opened_file)


def reformat_labels(labels):
    labels = (np.arange(len(species_list)) == labels[:, None]).astype(np.float32)
    return np.array(labels)


def accuracy(predictions, labels):
    return float((100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / len(predictions)))
