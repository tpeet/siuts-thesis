import json
import urllib2
import urllib
import pickle
import os
import siuts
from siuts import create_dir, Recording


def main():
    create_dir(siuts.data_dir)
    species_set = siuts.species_list
    acceptable_quality = siuts.acceptable_quality

    all_recordings = []
    for species_name in species_set:
        species_split = species_name.split("_")
        genus = species_split[0]
        species = species_split[1]
        url = "http://www.xeno-canto.org/api/2/recordings?query={0}%20{1}".format(genus, species)
        json_data = json.load(urllib2.urlopen(url))
        recordings = []

        # if data is divided to several pages, then include them all
        page = int(json_data["page"])
        while int(json_data["numPages"]) - page >= 0:
            # creates list of Recordings objects from recordings of each page
            quality_recordings = [
                Recording(x['id'], x['gen'], x['sp'], species_set.index("{0}_{1}".format(x["gen"], x["sp"])), x["file"])
                for x in json_data["recordings"] if x["q"] in acceptable_quality]
            recordings = recordings + quality_recordings
            page += 1
            if int(json_data["numPages"]) - page >= 0:
                json_data = json.load(urllib2.urlopen(url + "&page=" + str(page)))
        all_recordings = all_recordings + recordings

    with open(siuts.xeno_metadata_path, 'wb') as f:
        pickle.dump(all_recordings, f, protocol=-1)
    print "Finished downloading and saving training meta-data"

    path = siuts.xeno_dir
    create_dir(path)
    recordings_count = len(all_recordings)

    i = 0
    for rec in all_recordings:
        file_path = path + rec.get_filename() + ".mp3"
        i += 1
        if i % 100 == 0:
            print "{0}/{1} downloaded".format(i, recordings_count)

        if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
            urllib.urlretrieve(rec.file_url, file_path)


if __name__ == "__main__":
    main()
