from simplenlg import NLGFactory, Realiser, Lexicon, Feature, Tense
import nltk
from nltk.corpus import brown
from typing import List

class SentenceObject:
    def __init__(self, verb, subject, object, place):
        self.verb = verb
        self.subject = subject
        self.object = object
        self.place = place

        if subject == None:
            print(verb, subject, object, place)
            raise ValueError("Subject can not be empty")

    def ToString(self):
        key = self.verb + "_" + self.subject
        if self.object != None:
            key += "_" + self.object
        if self.place != None:
            key += "_" + self.place

        return key


class NLGSentenceGenerator:
    def __init__(self, decorate=True, decorate_compliments=True):
        lexi = Lexicon.getDefaultLexicon()
        self.factory = NLGFactory(lexi)
        self.realiser = Realiser()

        self.decorate = decorate
        self.decorate_compliments = decorate_compliments

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

    def get_trigram_preposition(self, target:str):
        choices = dict()
        for (p, d, n) in nltk.trigrams(brown.tagged_words(tagset="universal")):
            if n[0] == target:
                if n[1] == "NOUN" and d[1] == "DET" and p[1] == "ADP":
                    k = p[0] + "_" + d[0]
                    if k in choices:
                        choices[k] += 1
                    else:
                        choices[k] = 1

        most_used = ""
        if len(choices) > 0:
            sorted_choices = {k: v for k, v in sorted(choices.items(), key=lambda item: item[1])}
            most_used = list(sorted_choices.keys())[-1]
        return most_used

    def get_multi_trigram_preposition(self, targetset:set):
        choices = dict()
        for (p, d, n) in nltk.trigrams(brown.tagged_words(tagset="universal")):
            if n[0] in targetset:
                if n[1] == "NOUN" and d[1] == "DET" and p[1] == "ADP":
                    k = n[0] + "_" + p[0] + "_" + d[0]
                    if k in choices:
                        choices[k] += 1
                    else:
                        choices[k] = 1
        return self.most_used_by_key(choices)

    def get_bigram_preposition(self, target:str):
        choices = dict()
        for (p, n) in nltk.bigrams(brown.tagged_words(tagset="universal")):
            if n[0] == target:
                if n[1] == "NOUN" and p[1] == "ADP":
                    k = p[0]
                    if k in choices:
                        choices[k] += 1
                    else:
                        choices[k] = 1

        most_used = ""
        if len(choices) > 0:
            sorted_choices = {k: v for k, v in sorted(choices.items(), key=lambda item: item[1])}
            most_used = list(sorted_choices.keys())[-1]
        return most_used

    def get_multi_bigram_preposition(self, targetset:set):
        choices = dict()
        for (p, n) in nltk.bigrams(brown.tagged_words(tagset="universal")):
            if n[0] in targetset:
                if n[1] == "NOUN" and p[1] == "ADP":
                    k = n[0] + "_" + p[0]
                    if k in choices:
                        choices[k] += 1
                    else:
                        choices[k] = 1

        return self.most_used_by_key(choices)

    def get_determiner(self, target:str):
        choices = dict()
        for (d, n) in nltk.bigrams(brown.tagged_words(tagset="universal")):
            if n[0] == target and d[1] == "DET":
                k = d[0]
                if k in choices:
                    choices[k] += 1
                else:
                    choices[k] = 1
        most_used = ""
        if len(choices) > 0:
            sorted_choices = {k: v for k, v in sorted(choices.items(), key=lambda item: item[1])}
            most_used = list(sorted_choices.keys())[-1]
        return most_used

    def get_determiner_multi(self, targetset:set):
        choices = dict()
        for (d, n) in nltk.bigrams(brown.tagged_words(tagset="universal")):
            if n[0] in targetset:
                if d[1] == "DET":
                    k = n[0] + "_"+ d[0]
                    if k in choices:
                        choices[k] += 1
                    else:
                        choices[k] = 1
        return self.most_used_by_key(choices)

    def get_decorated_word(self, word:str):
        pre = ""
        if self.decorate:
            pre = self.get_determiner(word)

        if pre != "":
            pre = pre + " "
        return  pre + word

    def get_decorated_compliment(self, word:str):
        compliment = word
        if self.decorate_compliments:
            prepositions = self.get_trigram_preposition(word)
            if prepositions != "":
                compliment = prepositions + " " + word
        return compliment

    def generate(self, verb, subject, object=None, place=None):
        p = self.factory.createClause()
        p.setVerb(verb)
        p.setSubject(self.get_decorated_word(subject))

        if object != None:
            p.setObject(self.get_decorated_word(object))

        if place != None:
            p.setComplement(self.get_decorated_compliment(place))

        p.setFeature(Feature.TENSE, Tense.PRESENT)
        p.setFeature(Feature.PROGRESSIVE, True)

        output = self.realiser.realiseSentence(p)
        return output

    def generate_batch(self, sentences: List[SentenceObject], keyed=False):
        determiners = set()
        compliments = set()

        for s in sentences:
            determiners.add(s.subject)
            determiners.add(s.object)
            compliments.add(s.place)

        sub_prepos = self.get_determiner_multi(determiners)
        sub_complements = self.get_multi_trigram_preposition(compliments)

        outputs = []
        keyed_outputs = dict()

        for s in sentences:
            p = self.factory.createClause()
            p.setVerb(s.verb)

            if self.decorate and s.subject in sub_prepos:
                p.setSubject(sub_prepos[s.subject] + " " + s.subject)
            else:
                p.setSubject(s.subject)

            if s.object != None:
                if self.decorate and s.object in sub_prepos:
                    p.setObject(sub_prepos[s.object] + " " + s.object)
                else:
                    p.setObject(s.object)

            if s.place != None:
                if self.decorate_compliments and s.place in sub_complements:
                    p.setComplement(sub_complements[s.place] + " " + s.place)
                else:
                    p.setComplement(s.place)

            p.setFeature(Feature.TENSE, Tense.PRESENT)
            p.setFeature(Feature.PROGRESSIVE, True)

            output = self.realiser.realiseSentence(p)

            outputs.append(output)
            keyed_outputs[s.ToString()] = output

        if keyed:
            return keyed_outputs

        return outputs
