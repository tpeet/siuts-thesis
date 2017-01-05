import datetime
import os
import pickle
import time
from pydub import AudioSegment
import siuts
from siuts import create_dir


def main():
    start = time.time()
    create_dir(siuts.xeno_wavs_dir)
    create_dir(siuts.plutof_wavs_dir)

    print "Starting to convert training data to wav files"
    recordings = siuts.load(siuts.xeno_metadata_path)
    recordings_count = len(recordings)

    i = 0
    skipped_files = 0
    converted_files = 0
    for rec in recordings:
        if i % 1000 == 0:
            print "{0}/{1} | {2}".format(i, recordings_count, str(datetime.datetime.now()))
        i += 1
        file_path = "{0}{1}.wav".format(siuts.xeno_wavs_dir, rec.get_filename())

        if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
            try:
                sound = AudioSegment.from_mp3("{0}{1}.mp3".format(siuts.xeno_dir, rec.get_filename()))
                sound = sound.set_frame_rate(siuts.wav_framerate).set_channels(1)
                sound.export(file_path, format="wav")
                converted_files += 1
            except:
                print "Error on converting {0}".format(rec.get_filename())
        else:
            skipped_files += 1

    end = time.time()
    print "Converting training data from to wav files took {0} seconds. {1} recordings out of {2} were converted.".format(
        end - start, i, recordings_count)
    print "{0} files were already converted.".format(skipped_files)

    start = time.time()
    print ""
    print "Starting to convert testing data to wav files"
    recordings = siuts.load(siuts.plutof_metadata_path)
    recordings_count = len(recordings)

    i = 0
    skipped_files = 0
    converted_files = 0
    for rec in recordings:
        if i % 100 == 0:
            print "{0}/{1} | {2}".format(i, recordings_count, str(datetime.datetime.now()))

        file_path = "{0}{1}.wav".format(siuts.plutof_wavs_dir, rec.get_filename())
        i += 1
        if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
            try:
                sound = AudioSegment.from_file("{0}{1}.m4a".format(siuts.plutoF_dir, rec.get_filename()))
                sound = sound.set_frame_rate(siuts.wav_framerate).set_channels(1)
                sound.export(file_path, format="wav")
                converted_files += 1
            except:
                print "Error on {0}".format(rec.get_filename())
        else:
            skipped_files += 1

    end = time.time()
    print "Converting testing data from to wav files took {0} seconds. {1} recordings out of {2} were converted.".format(
        end - start, converted_files, recordings_count)
    print "{0} files were already converted.".format(skipped_files)


if __name__ == "__main__":
    main()
