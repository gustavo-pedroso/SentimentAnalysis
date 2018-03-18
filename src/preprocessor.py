import string
import re
import html
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as sw
from nltk.tokenize import TweetTokenizer
import infer_spaces
import enchant


class Preprocessor:
    dic = enchant.Dict('en_US')
    stopwords = set(sw.words('english'))
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    punct = set(string.punctuation)
    special_char = ['$', '%', '&', '*', '(', ')', '_', '-', '+', '=', '{', '[', '}', ']', '~', '.', ',', ';']
    tokenizer = TweetTokenizer(reduce_len=True, preserve_case=False)

    def preprocess(self, doc):
        temp = doc.lower()
        temp = html.unescape(temp)
        sentence = []

        for word in self.tokenizer.tokenize(temp):

            word = re.sub('((http|https)://)?[a-zA-Z0-9./?:@\-_=#]+\.([a-zA-Z0-9&./?:@\-_=#])*', '', word)  # remove url

            word = re.sub('[:;][\-^]?[)\]Dp]|[cCq(\[][\-^]?[:;]', ' good ', word)  # replace good emoticons

            word = re.sub('[:;][\-^]?[(\[<{]|[>)\]\}][\-^]?[:;]', ' bad ', word)  # replace bad emoticons

            for s in self.special_char:
                word = word.replace(s, '')

            if '@' in word:
                word = ''

            if '#' in word:
                word = infer_spaces.infer_spaces(word.replace('#', ''))

            if word.strip() in self.stopwords:
                word = ''

            if word.strip() in self.punct:
                word = ''

            if len(word.strip()) < 2:
                word = ''

            if word != '' and ' ' not in word:
                if not self.dic.check(word):
                    sug = self.dic.suggest(word)
                    if len(sug) > 0:
                        word = sug[0]

            word = word.lower()

            word = self.stemmer.stem(word)

            sentence.append(word)

        temp = ' '.join(sentence)
        temp = ' '.join(temp.split())

        return temp


