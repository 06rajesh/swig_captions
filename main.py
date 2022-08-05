from NLGSentenceGenerator import NLGSentenceGenerator


if __name__ == '__main__':

    generator = NLGSentenceGenerator()
    generated = generator.generate("chase", "mary", object="dog", place="forest")
    print(generated)