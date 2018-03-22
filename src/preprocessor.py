import string
import re
import html
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as sw
from nltk.tokenize import TweetTokenizer
import src.infer_spaces as infer_spaces
import enchant
import src.conf as conf


class Preprocessor:
    dic = enchant.Dict('en_US')
    stopwords = set(sw.words('english'))
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    punct = set(string.punctuation)
    digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    special_char = ['$', '%', '&', '*', '(', ')', '_', '-', '+', '=', '{', '[', '}', ']', '~', '.', ',', ';']
    tokenizer = TweetTokenizer(reduce_len=True, preserve_case=False)

    english_names = set(map(lambda x: x.replace('\n', '').strip().lower(),
                        open(conf.project_path + 'data/' + 'english_names.txt', 'r').readlines()))

    def preprocess(self, doc):
        temp = doc.lower()
        temp = html.unescape(temp)
        sentence = []

        temp_sentence = []
        for word in self.tokenizer.tokenize(temp):
            if '#' in word:
                word = infer_spaces.infer_spaces(word.replace('#', ''))
            temp_sentence.append(word)
        temp = ' '.join(temp_sentence)
        temp = ' '.join(temp.split())

        for word in self.tokenizer.tokenize(temp):

            word = word.strip()

            word = re.sub('((http|https)://)?[a-zA-Z0-9./?:@\-_=#]+\.([a-zA-Z0-9&./?:@\-_=#])*', '', word)  # remove url

            word = re.sub('[:;][\-^]?[)\]Dp]|[cCq(\[][\-^]?[:;]', ' good ', word)  # replace good emoticons

            word = re.sub('[:;][\-^]?[(\[<{]|[>)\]\}][\-^]?[:;]', ' bad ', word)  # replace bad emoticons

            for s in self.special_char:
                word = word.replace(s, '')

            for d in self.digits:
                if d in word:
                    word = ''
                    break

            if word in self.english_names and not self.dic.check(word):
                word = ''

            if '@' in word:
                word = ''

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

        print(temp)
        return temp
