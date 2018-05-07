from pymongo import MongoClient
import src.conf as conf
import pickle
from src.classifier import Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from src.preprocessor import Preprocessor


client = MongoClient('localhost', 27017)
db = client['tcc']
collection = db['tweets']

events = collection.find({"classified": True})

dataset = pickle.load(open(conf.project_path + 'data\dataset_preprocessed.pickle', 'rb'))
dataset = [x for x in dataset if len(x[0].split()) > 0]

# positives = [x for x in dataset if x[1] == 'positive']
# negatives = [x for x in dataset if x[1] == 'negative']
#
# dataset = negatives + positives[0:len(negatives)]

vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, sublinear_tf=True, use_idf=True)
classifier = SVC(kernel='rbf', C=2.9, gamma=1)
p = Preprocessor()

clf = Classifier(vectorizer=vectorizer, classifier=classifier)

x_train = [x[0] for x in dataset]
y_train = [x[1] for x in dataset]

clf.train(x_train=x_train, y_train=y_train)

print(clf.predict(p.preprocess('''''')))

# for ev in events:
#     print(ev['text'].replace('\n', '') + ' --> ' + clf.predict(p.preprocess(ev['text'])))
