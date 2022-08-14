from pathlib import Path
import json
from nltk.corpus import wordnet as wn
# from nltk.stem import PorterStemmer
import re
import math

from NLGSentenceGenerator import NLGSentenceGenerator, SentenceObject


class SWiGCaptions:
    def __init__(self, targetfile:Path, batch_size=4):
        self.wn = wn
        self.generator = NLGSentenceGenerator()
        # self.ps = PorterStemmer()
        self.batch_size = batch_size

        self.targetfile = targetfile
        self._json = {}
        self._itemkeys = []
        self.n_items = 0
        self.total_batch = 0

        self.load_json(targetfile)

        self.subject_roles = [
            "agent", "agents", "agenttype", "boaters", "substance", "seller",
            "victim", "farmer", "source", "buyer", "carrier", "eater", "experiencer",
            "farmer", "gatherers", "giver", "listener", "mourner", "perceiver",
            "performer", "seller", "sprouter", "individuals",
        ]

        self.object_roles = [
            "admired", "blocked", "bodypart", "boringthing", "caughtitem", "coagent",
            "coagentpart", "victim",
        ]

    def load_json(self, targetfile: Path):
        path = Path(targetfile)

        if not path.is_file():
            raise ValueError("Target JSON file does not exists")

        with open(targetfile) as f:
            self._json = json.load(f)
            self._itemkeys = list(self._json.keys())
            self.n_items = len(self._itemkeys)
            self.total_batch = math.ceil(self.n_items/self.batch_size)

    def noun2synset(self, noun):
        return wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*',
                                                                                        noun) else "'{}'".format(noun)

    def processed_synset(self, noun):
        syn = self.noun2synset(noun)
        splitted = syn.split(".")
        final = splitted[0]
        final = final.replace("_", " ")
        return final

    def verb_stemmer(self, verbword:str):
        if verbword.endswith("ing"):
            size = len(verbword)
            return verbword[:size-3]

        return verbword

    def get_role_values_with_empty_check(self, role_values:dict, selected_role:str):
        if role_values[selected_role] == '':
            return selected_role
        else:
            return self.processed_synset(role_values[selected_role])

    def debug_roles(self, role_value_dict, verb=None):
        if verb != None:
            print(verb)
        for role in role_value_dict:
            val = self.get_role_values_with_empty_check(role_value_dict, role)
            print(role + ": " + val)
        print("=====================================")

    def detect_subject_from_roles(self, roles):
        selected = None

        if len(roles) == 1:
            return roles[-1]

        for r in roles:
            if r in self.subject_roles:
                selected = r

        if selected == None:
            for r in roles:
                if r.endswith("er"):
                    selected = r

        return selected

    def detect_object_from_roles(self, roles):
        selected = None

        if len(roles) == 1:
            return roles[-1]

        for r in roles:
            if r in self.object_roles:
                selected = r

        if selected == None:
            for r in roles:
                if r.endswith("item") or r.endswith("items"):
                    selected = r

        return selected


    def sentence_object_from_frames(self, verb, role_values):
        subj = None
        obj = None
        verb = self.verb_stemmer(verb)
        compliments = []

        total_keys = len(role_values.keys())

        if total_keys == 1:
            for role in role_values:
                subj = self.get_role_values_with_empty_check(role_values, role)
            sent = SentenceObject(verb, subj)
        else:
            roles = list(role_values.keys())

            subject_role = self.detect_subject_from_roles(roles)
            subj = self.get_role_values_with_empty_check(role_values, subject_role)
            roles.remove(subject_role)

            object_role = self.detect_object_from_roles(roles)
            if object_role != None:
                obj = self.get_role_values_with_empty_check(role_values, object_role)
                roles.remove(object_role)

            # append all the other roles as compliments except place
            for r in roles:
                if r != "place":
                    compitem = self.get_role_values_with_empty_check(role_values, r)
                    compliments.append(compitem)

            # append place at the end of the compliments
            if "place" in roles:
                compitem = self.get_role_values_with_empty_check(role_values, "place")
                compliments.append(compitem)

            # remove duplicates of compliments
            # by placing the last item like place in the end
            filtered = []
            for c in compliments:
                if c not in filtered:
                    filtered.append(c)
                else:
                    filtered.remove(c)
                    filtered.append(c)

            if subj == None or verb == None:
                return None

            sent = SentenceObject(verb, subj, object=obj, compliments=filtered)
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
                    try:
                        sentence_keys.append(s.ToString())
                    except TypeError:
                        print(f)
                        exit()

            samples[key] = sentence_keys

        captions = self.generator.generate_batch(sentences, keyed=True)

        outputs = dict()
        for key in samples:
            caps = list()
            for k in samples[key]:
                caps.append(captions[k])
            outputs[key] = caps

        return outputs, skipped

    def read_and_generate_batch(self, batch_number):
        current_batch = batch_number
        batch_data = {}

        range_start = self.batch_size * (current_batch - 1)
        range_end = range_start + self.batch_size

        if range_end > self.n_items:
            range_end = self.n_items

        for i in range(range_start, range_end):
            key = self._itemkeys[i-1]
            batch_data[key] = self._json[key]

        return self.process_batch_data(batch_data)
