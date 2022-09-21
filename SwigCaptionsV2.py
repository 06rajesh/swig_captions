import json
import re
from typing import List
import math

# seed the pseudorandom number generator
import random

import cv2
from pathlib import Path
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import brown

from rolemaps import SUBJECT_ROLES, AGENT_ROLES, OBJECT_ROLES, ROLE_PREPOSITION_MAP, PLURAL_NOUNS, VERB_FORM_MAPS

DETERMINERS_LIST = ['a', 'an', 'the']

def show_img(imgpath:Path, annotation:dict):
    # read image
    img = cv2.imread(str(imgpath))

    result = img.copy()
    boxes = annotation['bb']
    for bkey in boxes:
        b = boxes[bkey]
        cv2.rectangle(result, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    # show thresh and result
    cv2.imshow("bounding_box", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def noun2synset(noun, trim=False):
    if noun == '':
        return noun

    synset = wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*', noun) \
        else "'{}'".format(noun)

    if trim:
        return synset.split('.')[0]
    else:
        return synset

def count_n_roles(annotation:dict):
    return len(annotation['bb'].keys())

class SwigCaptionGenerator:
    img_keys: List[str]
    annotations: dict
    wn: wn
    imgdir: Path
    vocabs: dict
    agent_roles: list
    agentlist: dict
    placelist: dict
    determiners: dict
    prepositions: dict
    subject_roles: list
    object_roles: list
    debug: bool


    def __init__(self, annotations:dict,  debug:bool = False, img_dir: str = './SWiG/images_512'):
        self.annotations = annotations
        self.img_keys = list(annotations.keys())
        self.imgdir = Path(img_dir)
        self.wn = wn
        self.vocabs = {}

        self.agentlist = {}
        self.placelist = {}
        self.determiners = {}
        self.prepositions = {}

        self.agent_roles = AGENT_ROLES
        self.subject_roles = SUBJECT_ROLES
        self.object_roles = OBJECT_ROLES

        self.debug = debug

        random.seed(1)
        self.preprocess_frames()

    def check_synonym(self, noun):
        synlist = wn.synsets(noun)

        if len(synlist) > 0:
            hypnyms = synlist[0].hypernyms()
            if len(hypnyms) > 0:
                randidx = random.randint(1, len(hypnyms))
                name = hypnyms[randidx-1].name()
                noun = name.split('.')[0]
                noun = noun.replace("_", " ")
                return noun

        return noun

    def format_noun(self, noun):

        noun = noun2synset(noun, trim=True)
        noun = noun.replace("_", " ")

        noun = noun.lower()
        return noun

    def remove_empty_roles(self, roles: dict):
        trimmed = dict()
        for r in roles:
            if roles[r] != '':
                trimmed[r] = roles[r]
        return trimmed

    def bb_list_to_dict(self, bblist:List):
        """
        bblist : List
            items: ['x1', 'y1', 'x2', 'y2']
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        """
        return {
            'x1': bblist[0], 'x2': bblist[2], 'y1': bblist[1], 'y2': bblist[3],
        }

    def preprocess_frames(self):

        all_agents = []
        all_places = []

        for key in self.img_keys:
            annotation = self.annotations[key]
            frames = annotation['frames']
            agentlist = []
            placelist = []
            for f in frames:
                for role in f:
                    roleval = f[role]
                    if roleval != '':
                        if roleval not in self.vocabs:
                            self.vocabs[roleval] = self.format_noun(roleval)
                        if role in self.agent_roles:
                            agentlist.append(roleval)
                            if roleval not in all_agents:
                                all_agents.append(roleval)
                        if role == 'place':
                            placelist.append(roleval)
                            if roleval not in all_places:
                                all_places.append(roleval)
            self.agentlist[key] = agentlist
            self.placelist[key] = placelist

        targets = [self.vocabs[v] for v in self.vocabs if len(self.vocabs[v].split()) == 1]
        targets = set(targets)
        determiners = self.compute_vocab_determiners(targets)

        # print(self.vocabs)

        for v in self.vocabs:
            vocab_val = self.vocabs[v]
            if vocab_val in determiners:
                self.determiners[v] = determiners[vocab_val]
            else:
                self.determiners[v] = 'the'


        places = [self.vocabs[p] for p in all_places]
        places = set(places)
        place_preps = self.compute_trigram_preposition(places, preplist=['at', 'in', 'on'])
        for p in place_preps:
            self.prepositions[p] = place_preps[p]

        preptarget = [self.vocabs[v].split()[0] for v in self.vocabs if v not in all_places and v not in all_agents]
        preptarget = set(preptarget)

        if self.debug:
            print(preptarget)

        vocab_preps = self.compute_trigram_preposition(preptarget)
        for p in vocab_preps:
            self.prepositions[p] = vocab_preps[p]

        if self.debug:
            print(self.prepositions)
            print(self.agentlist)
            print(self.placelist)


    def most_used_by_key(self, key_prep_dict):
        most_used_dict = dict()
        if len(key_prep_dict) > 0:
            sorted_choices = {k: v for k, v in sorted(key_prep_dict.items(), key=lambda item: item[1], reverse=True)}
            for choice in sorted_choices:
                splitted = choice.split("_")
                key = splitted[0]
                if key not in most_used_dict:
                    most_used_dict[key] = ' '.join(splitted[1:])
        return most_used_dict

    def compute_vocab_determiners(self, targetset:set):
        if len(targetset) == 0:
            return {}

        choices = dict()
        for (d, n) in nltk.bigrams(brown.tagged_words(tagset="universal")):
            if n[0].lower() in targetset:
                if d[0].lower() in DETERMINERS_LIST:
                    k = n[0].lower() + "_"+ d[0].lower()
                    if k in choices:
                        choices[k] += 1
                    else:
                        choices[k] = 1
        return self.most_used_by_key(choices)

    def compute_trigram_preposition(self, targetset:set, preplist:List=None):
        if len(targetset) == 0:
            return {}

        choices = dict()
        for (p, d, n) in nltk.trigrams(brown.tagged_words(tagset="universal")):
            if preplist:
                if n[0].lower() in targetset and p[0].lower() in preplist:
                    if d[1] == "DET" and p[1] == "ADP":
                        k = n[0].lower() + "_" + p[0].lower()
                        if k in choices:
                            choices[k] += 1
                        else:
                            choices[k] = 1
            else:
                if n[0].lower() in targetset:
                    if d[1] == "DET" and p[1] == "ADP":
                        k = n[0].lower() + "_" + p[0].lower()
                        if k in choices:
                            choices[k] += 1
                        else:
                            choices[k] = 1
        return self.most_used_by_key(choices)

    def detect_subject_from_frame(self, frame:dict, annot_key:str):
        all_keys:List[str] = list(frame.keys())
        agentlist = self.agentlist[annot_key]

        if len(all_keys) == 1:
            if all_keys[0] == 'place':
                if len(agentlist) > 0:
                    return ('agent', agentlist[0])
                else:
                    return (None, None)
            return (all_keys[0], frame[all_keys[0]])

        for k in all_keys:
            if k in self.agent_roles:
                return (k, frame[k] )

        for k in all_keys:
            if k in self.subject_roles:
                return (k, frame[k] )

        for k in all_keys:
            if k.endswith("er") or k.endswith("ers"):
                return (k, frame[k] )

        for k in all_keys:
            if k != 'place':
                return (k, frame[k])

    def detect_place_from_frame(self, frame: dict, ignore_keys:list, annot_key:str):
        place_roles = ['place']
        all_keys:List[str] = list(frame.keys())
        all_keys = [k for k in all_keys if k not in ignore_keys]

        placelist = self.placelist[annot_key]

        for k in all_keys:
            if k in place_roles:
                return (k, frame[k])

        if len(placelist) > 0:
            return ('place', placelist[0])

        return (None, None)

    def detect_object_from_frame(self, frame: dict, ignore_keys:list):
        all_keys:List[str] = list(frame.keys())
        all_keys = [k for k in all_keys if k not in ignore_keys]

        for k in all_keys:
            if k in self.object_roles:
                return (k, frame[k])

        for k in all_keys:
            if k.endswith("item") or k.endswith("items"):
                return (k, frame[k])

        return (None, None)

    def get_place_phrase(self, place:tuple):
        base_place_phrase = self.determiners[place[1]] + " " + self.vocabs[place[1]]

        place_n = self.vocabs[place[1]]
        if place_n in self.prepositions.keys():
            return self.prepositions[place_n] + " " + base_place_phrase

        return 'at ' + base_place_phrase

    def get_as_phrase(self, val:str, role: str, with_pp:bool=True):
        base_phrase = self.determiners[val] + " " + self.vocabs[val]

        if with_pp:
            if self.determiners[val] == 'the':
                base_phrase = self.vocabs[val]

            if role in ROLE_PREPOSITION_MAP:
                if ROLE_PREPOSITION_MAP[role] != '':
                    return ROLE_PREPOSITION_MAP[role] + " " + base_phrase
                else:
                    return base_phrase

            if self.debug:
                print(role + ':' + self.vocabs[val])

            if self.vocabs[val] in self.prepositions:
                return self.prepositions[self.vocabs[val]] + " " + base_phrase
        else:
            rand = random.random()
            if rand > 0.75:
                synval = self.check_synonym(self.vocabs[val])
                base_phrase = self.determiners[val] + " " + synval

        return base_phrase

    def is_plural(self, subject:str, subj_key:str=None):
        noun = subject.split()[-1]
        if subj_key and subj_key.endswith('s'):
            return True
        if noun in PLURAL_NOUNS:
            return True
        return False

    def get_verb_phrase(self, annot_key: str, subj_val:str, subj_key:str):
        if annot_key not in self.img_keys:
            raise Warning("Image key not available in provided annotations")

        is_plural = False
        if subj_val != None and subj_key != None:
            is_plural = self.is_plural(self.vocabs[subj_val], subj_key)

        annotation = self.annotations[annot_key]
        verb = annotation['verb']

        rand = random.random()
        if verb in VERB_FORM_MAPS and rand > 0.3:
            verb_forms = VERB_FORM_MAPS[verb]
            if is_plural:
                return verb_forms[0]
            else:
                return verb_forms[1]

        if is_plural:
            return 'are ' + verb
        else:
            return 'is ' + verb

    def construct_passive_sentence(self, object_phrase:str, subject_phrase: str, annot_key:str):
        annotation = self.annotations[annot_key]
        verb = annotation['verb']

        if verb not in VERB_FORM_MAPS:
            raise ValueError("Verb not found")

        verb_forms = VERB_FORM_MAPS[verb]
        verb_tense = verb_forms[0] + 'ed'
        if verb_forms[0].endswith('e'):
            verb_tense = verb_forms[0] + 'd'
        verb_phrase = "getting " + verb_tense + ' by'

        return object_phrase + " is " + verb_phrase + " " + subject_phrase


    def construct_sentence(self, subject:str, verb_phrase:str, object_phrase:str, compliments:List, place:str, annot_key:str):

        base_sentence = subject + " " + verb_phrase

        if object_phrase != "":
            annotation = self.annotations[annot_key]
            verb = annotation['verb']
            rand = random.random()
            if verb in VERB_FORM_MAPS and rand > 0.75:
                base_sentence = self.construct_passive_sentence(object_phrase, subject, annot_key)
            else:
                base_sentence = base_sentence + " " + object_phrase

        if len(compliments) > 0:
            compliment_phrases = " ".join(compliments)
            base_sentence =  base_sentence + " " + compliment_phrases

        if place != "":
            base_sentence = base_sentence + " " + place

        return base_sentence

    def process_frames(self, frame:dict, annot_key:str):
        f = self.remove_empty_roles(frame)
        ignored_keys = []

        sub_key, sub_val = self.detect_subject_from_frame(f, annot_key)
        if sub_key:
            ignored_keys.append(sub_key)

        place_key, place_val = self.detect_place_from_frame(f, ignored_keys, annot_key)
        if place_key:
            ignored_keys.append(place_key)

        object_key, object_val = self.detect_object_from_frame(f, ignored_keys)
        if object_key:
            ignored_keys.append(object_key)

        if sub_val:
            subject_phrase = self.get_as_phrase(sub_val, sub_key, with_pp=False)
        else:
            subject_phrase = "it"

        verb_phrase = self.get_verb_phrase(annot_key, sub_val, sub_key)

        object_phrase = ""
        if object_val:
            object_phrase = self.get_as_phrase(object_val, object_key, with_pp=False)

        compliments = [(k, f[k]) for k in f.keys() if k not in ignored_keys]
        compliments_list = [self.get_as_phrase(n, k, with_pp=True) for k, n in compliments]

        place_phrase = ""
        if place_val:
            place_phrase = self.get_place_phrase((place_key, place_val))

        sentence = self.construct_sentence(subject_phrase, verb_phrase, object_phrase, compliments_list, place_phrase, annot_key)
        return sentence

    def generate_sentences(self, annot_key:str):
        if annot_key not in self.img_keys:
            raise Warning("Image key not available in provided annotations")

        annotation = self.annotations[annot_key]
        frames = annotation['frames']
        if self.debug:
            print(frames)
        sentences = []
        for f in frames:
            s = self.process_frames(f, annot_key)
            sentences.append(s)

        return sentences

class SwigCaptionV2:
    batch_size:int
    target_file: Path

    _json:dict
    _item_keys:List
    n_items: int
    total_batch: int

    def __init__(self, targetfile:Path, batch_size:int):
        self.batch_size = batch_size

        self.target_file = targetfile
        self._json = {}
        self._item_keys = []
        self.n_items = 0
        self.total_batch = 0

        self.load_json(targetfile)

    def load_json(self, targetfile: Path):
        path = Path(targetfile)

        if not path.is_file():
            raise ValueError("Target JSON file does not exists")

        with open(targetfile) as f:
            self._json = json.load(f)
            self._item_keys = list(self._json.keys())
            self.n_items = len(self._item_keys)
            self.total_batch = math.ceil(self.n_items / self.batch_size)

    def process_batch_data(self, batch_data:dict):
        captionGen = SwigCaptionGenerator(batch_data)
        output = {}
        for key in batch_data:
            sentences = captionGen.generate_sentences(key)
            output[key] = sentences

        return output


    def read_and_generate_batch(self, batch_number):
        current_batch = batch_number
        batch_data = {}

        range_start = self.batch_size * (current_batch - 1)
        range_end = range_start + self.batch_size

        if range_end > self.n_items:
            range_end = self.n_items

        for i in range(range_start, range_end):
            key = self._item_keys[i-1]
            batch_data[key] = self._json[key]

        return self.process_batch_data(batch_data)

    def debug_batch_by_role_len(self, role_len:int):
        batch_data = {}
        count = 0
        for key in self._item_keys:
            annotation = self._json[key]
            bboxes = annotation['bb']

            if len(bboxes.keys()) == role_len:
                batch_data[key] = annotation
                count += 1

                if count >= self.batch_size:
                    break

        return self.process_batch_data(batch_data)

    def debug_batch_by_role(self, role:str, offset=0):
        batch_data = {}
        count = 0
        for key in self._item_keys:
            annotation = self._json[key]
            bboxes = annotation['bb']

            if role in bboxes.keys():
                if count >= offset:
                    batch_data[key] = annotation
                count += 1

                if count >= self.batch_size+offset:
                    break

        return self.process_batch_data(batch_data)

    def debug_roles(self):
        roles = {}

        for key in self._item_keys:
            annotation = self._json[key]
            bboxes = annotation['bb']

            map_keys = list(ROLE_PREPOSITION_MAP.keys())
            for r in bboxes.keys():
                if r not in map_keys and r not in OBJECT_ROLES:
                    if r in roles:
                        roles[r] += 1
                    else:
                        roles[r] = 1

        sorted_roles = {k: v for k, v in sorted(roles.items(), key=lambda item: item[1], reverse=True)}
        print(sorted_roles)
        print("Total {} roles found".format(len(roles)))

    def debug_verbs(self):
        verbs = {}

        for key in self._item_keys:
            annotation = self._json[key]
            verb = annotation['verb']

            if verb not in verbs:
                verbs[verb] = 1
            else:
                verbs[verb] += 1

        sorted_verbs = {k: v for k, v in sorted(verbs.items(), key=lambda item: item[1], reverse=True)}
        print(sorted_verbs)
        print("Total {} verbs found".format(len(verbs)))

# if __name__ == '__main__':
#     root = Path('./SWiG')
#     annotation_file = 'train.json'
#
#     imgdir = root / 'images_512'
#     annotationfile = root / 'SWiG_jsons' / annotation_file
#
#
#     capgen = SwigCaptionV2(annotationfile, 4)
#     # capgen.debug_verbs()
#     captions = capgen.debug_batch_by_role('item', offset=10)
#     print(captions)
#     # captions = capgen.read_and_generate_batch(4)
#     # print(captions)


    # with open(annotationfile) as f:
    #     all = json.load(f)
    #
    # count = 0
    # for key in all.keys():
    #     annotation = all[key]
    #     imgpath = imgdir / key
    #
    #     bboxes = annotation['bb']
    #
    #     if len(bboxes.keys()) == 4:
    #         gen = SwigCaptionGenerator(annotation, img_id=key)
    #         gen.generate_sentences()
    #         show_img(imgpath, annotation)
    #
    #         count += 1
    #
    #         if count >= 5:
    #             break
