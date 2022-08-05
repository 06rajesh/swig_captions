from simplenlg import NLGFactory, Realiser, Lexicon, Feature, Tense
import nltk
from nltk.corpus import brown


class NLGSentenceGenerator:
    def __init__(self, decorate=True, decorate_compliments=True):
        lexi = Lexicon.getDefaultLexicon()
        self.factory = NLGFactory(lexi)
        self.realiser = Realiser()

        self.tigrams_set = nltk.trigrams(brown.tagged_words(tagset="universal"))
        self.bigrams_set = nltk.bigrams(brown.tagged_words(tagset="universal"))

        self.decorate = decorate
        self.decorate_compliments = decorate_compliments

    def get_trigram_preposition(self, target:str):
        choices = dict()
        for (p, d, n) in self.tigrams_set:
            if n[1] == "NOUN" and n[0] == target and d[1] == "DET" and p[1] == "ADP":
                k = p[0] + " " + d[0]
                if k in choices:
                    choices[k] += 1
                else:
                    choices[k] = 1

        most_used = ""
        if len(choices) > 0:
            sorted_choices = {k: v for k, v in sorted(choices.items(), key=lambda item: item[1])}
            most_used = list(sorted_choices.keys())[-1]
        return most_used

    def get_bigram_preposition(self, target:str):
        choices = dict()
        for (p, n) in self.bigrams_set:
            if n[1] == "NOUN" and n[0] == target and p[1] == "ADP":
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

    def get_determiner(self, target:str):
        choices = dict()
        for (d, n) in self.bigrams_set:
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