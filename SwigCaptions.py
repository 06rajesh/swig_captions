from pathlib import Path
import json
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import re

from NLGSentenceGenerator import NLGSentenceGenerator, SentenceObject


class SWiGCaptions:
    def __init__(self, batch_size=4):
        self.wn = wn
        self.generator = NLGSentenceGenerator()
        self.ps = PorterStemmer()
        self.batch_size = batch_size

    def noun2synset(self, noun):
        return wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*',
                                                                                        noun) else "'{}'".format(noun)

    def processed_synset(self, noun):
        syn = self.noun2synset(noun)
        splitted = syn.split(".")
        final = splitted[0]
        final = final.replace("_", " ")
        return final

    def sentence_object_from_frames(self, verb, role_values):
        subj = None
        obj = None
        place = None
        verb = self.ps.stem(verb)

        subject_roles = ["agent", "agents", "agenttype", "boaters", "substance", "seller", "victim", "farmer"]

        for role in role_values:
            if role in subject_roles:
                subj = self.processed_synset(role_values[role])
            elif role == "place":
                place = self.processed_synset(role_values[role])
            else:
                obj = self.processed_synset(role_values[role])

        if subj == None:
            for role in role_values:
                val = self.processed_synset(role_values[role])
                print(role + ": " + val)
            return  None

        sent = SentenceObject(verb, subj, object=obj, place=place)
        return sent


    def process_batch_data(self, batch_data):
        sentences = []
        samples = dict()

        skipped = 0

        for idx, key in enumerate(batch_data):
            verb = batch_data[key]['verb']
            frames = batch_data[key]['frames']
            sentence_keys = []
            for f in frames:
                s = self.sentence_object_from_frames(verb, f)
                if s == None:
                    skipped += 1
                else:
                    sentences.append(s)
                    sentence_keys.append(s.ToString())

            samples[key] = sentence_keys

        captions = self.generator.generate_batch(sentences, keyed=True)

        outputs = dict()
        for key in samples:
            caps = list()
            for k in samples[key]:
                caps.append(captions[k])
            outputs[key] = caps

        return outputs, skipped

    def read_and_generate_batch(self, filepath, batch_number):
        test_file = filepath
        current_batch = batch_number
        batch_data = {}

        with open(test_file) as f:
            test_json = json.load(f)

            range_start = self.batch_size * (current_batch - 1)
            range_end = range_start + self.batch_size

            for idx, key in enumerate(test_json):
                if (idx >= range_start) and (idx < range_end):
                    batch_data[key] = test_json[key]
                elif idx >= range_end:
                    break

        return self.process_batch_data(batch_data)
