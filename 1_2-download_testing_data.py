import json
import urllib2
import pickle
import urllib
import math
import os
import siuts
from siuts import create_dir, Recording


def main():
    files_url_prefix = "https://files.plutof.ut.ee/"

    create_dir(siuts.plutoF_dir)

    taxon_ids = siuts.plutoF_taxon_ids

    taxon_url_temp = "https://api.plutof.ut.ee/v1/taxonomy/taxonnodes/{0}/"
    taxon_urls = [taxon_url_temp.format(x) for x in taxon_ids]

    recordings = []
    counter = 0
    url = "https://api.plutof.ut.ee/v1/public/taxonoccurrence/observation/observations/?mainform=15&page={}&page_size=100"
    json_data = json.load(urllib2.urlopen(url.format(1)))
    number_of_pages = int(math.ceil(float(json_data["collection"]["count"]) / 100))

    print "Starting to download audio recordings"

    for page in range(1, number_of_pages + 1):
        json_data = json.load(urllib2.urlopen(url.format(page)))
        print
        print "Downloading from page {}".format(page)
        items = json_data['collection']['items']
        for item in items:
            links = item['links']
            taxon_url = [x['href'] for x in links if 'rel' in x and x['rel'] == 'taxon_node'][0]

            # if species is part of our classification task
            if taxon_url in taxon_urls:
                audio_urls = [x['href'] for x in links if 'format' in x and 'audio' in x['format']]
                if len(audio_urls) > 0:
                    audio_url = audio_urls[0].replace("/public/", "/")
                    audio_data = json.load(urllib2.urlopen(audio_url))
                    file_url = files_url_prefix + audio_data["public_url"]

                    rec_id = audio_data["id"]
                    label = taxon_urls.index(taxon_url)
                    sp_name = siuts.species_list[label]
                    gen = sp_name.split("_")[0]
                    sp = sp_name.split("_")[1]

                    # use same file format as with xeno-canto recordings: <genus_species-id>
                    fname = "{}-{:06d}".format(sp_name, audio_data["id"])
                    file_path = "{}/{}.m4a".format(siuts.plutoF_dir, fname)

                    if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
                        urllib.urlretrieve(file_url, file_path)

                    recordings.append(Recording(rec_id, gen, sp, label, file_url))
                    counter += 1
                    if counter % 10 == 0:
                        print "Downloaded {} files".format(counter)

    with open(siuts.plutof_metadata_path, 'wb') as f:
        pickle.dump(recordings, f, protocol=-1)

    print ""
    print "Finished saving meta-data and downloaded {0} recordings".format(len(recordings))


if __name__ == "__main__":
    main()
