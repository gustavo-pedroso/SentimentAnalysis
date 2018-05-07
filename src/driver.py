import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from src.preprocessor import Preprocessor
from random import sample
from src.classifier import Classifier
import src.conf as conf
import numpy as np


preprocess = False
if preprocess:

    dataset_pickle = open(conf.project_path + 'data\dataset_cleared.pickle', 'rb')
    dataset = pickle.load(dataset_pickle)
    dataset_pickle.close()

    preprocessor = Preprocessor()

    dataset = list(map(lambda x: (preprocessor.preprocess(x[0]), x[1]), dataset))  # preprocess dataset
    preprocessed_dataset = open(conf.project_path + 'data\dataset_preprocessed.pickle', 'wb')
    pickle.dump(dataset, preprocessed_dataset)

dataset = pickle.load(open(conf.project_path + 'data\dataset_preprocessed.pickle', 'rb'))

dataset = [x for x in dataset if len(x[0].split()) > 0]

dataset = list(set(dataset))

classifiers = [(SVC(kernel='rbf', C=2.9, gamma=1), 'svm_rbf'),
               (SVC(kernel='linear'), 'svm_linear')]
               # (KNeighborsClassifier(), 'knn'),
               # (MultinomialNB(), 'naive_bayes'),
               # (Perceptron(), 'perceptron')]

vectorizers = [(TfidfVectorizer(min_df=0.0, max_df=1.0, sublinear_tf=True, use_idf=True), 'tfidf')]
               # (CountVectorizer(min_df=0.0, max_df=1.0), 'count'),
               # (HashingVectorizer(), 'hash')]


c = Classifier(classifier=classifiers[1][0], vectorizer=vectorizers[0][0])
x = list(map(lambda a: a[0], dataset))
y = list(map(lambda a: a[1], dataset))

c.train(x_train=x, y_train=y)

from sklearn.pipeline import make_pipeline
import eli5

pipe = make_pipeline(vectorizers[0][0], classifiers[1][0])
pipe.fit(x, y)


file = open('C:/Users/Gustavo/Desktop/batata.html', 'wb')
file.write(eli5.show_weights(classifiers[1][0], vec=vectorizers[0][0], top=1000).data.encode('utf8'))
file.close()







# cl = Classifier(SVC(), vectorizers[0][0])
# c_avg = []
# gamma_avg = []
# for i in range(0, 10):
#     dataset1 = sample(dataset, len(dataset))
#     x = list(map(lambda a: a[0], dataset1))
#     y = list(map(lambda a: a[1], dataset1))
#     r = cl.svc_param_selection(x, y, 5, "grid")
#     print(r)
#     c_avg.append(r['C'])
#     gamma_avg.append(r['gamma'])
#
# print(numpy.average(c_avg))
# print(numpy.average(gamma_avg))


# runs = 100
# for clf in classifiers:
#     for vec in vectorizers:
#         if clf[1] == 'naive_bayes' and vec[1] == 'hash':
#             continue
#         print(vec[1] + ' and ' + clf[1])
#         mr_avg = []
#         acc_avg = []
#
#         file_path = conf.project_path + 'results/' + vec[1] + '-' + clf[1] + '.txt'
#         file = open(file_path, 'w')
#
#         for i in range(0, runs):
#             dataset = sample(dataset, len(dataset))
#             c = Classifier(clf[0], vec[0])
#             r = c.evaluate_score(dataset, 10)
#             mr_avg.append(r[0])
#             acc_avg.append(r[1])
#             print(str(r[0]) + ',' + str(r[1]))
#             file.write(str(r[0]) + ',' + str(r[1]) + '\n')
#         file.close()
#         print(numpy.mean(mr_avg), numpy.mean(acc_avg))
