import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from preprocessor import Preprocessor
from random import sample
from classifier import Classifier
import numpy

preprocess = False
if preprocess:

    dataset_pickle = open(r'data\dataset_cleared.pickle', 'rb')

    dataset = pickle.load(dataset_pickle)
    dataset_pickle.close()

    preprocessor = Preprocessor()

    positives = list(filter(lambda x: x[1] == 'positive', dataset))  # filter positives
    negatives = list(filter(lambda x: x[1] == 'negative', dataset))  # filter negatives

    min_examples = min(len(positives), len(negatives))

    positives = positives[0:min_examples]
    negatives = negatives[0:min_examples]

    dataset = positives + negatives

    dataset = list(map(lambda x: (preprocessor.preprocess(x[0]), x[1]), dataset))  # preprocess dataset
    preprocessed_dataset = open(r'data\dataset_preprocessed.pickle', 'wb')
    pickle.dump(dataset, preprocessed_dataset)

dataset = pickle.load(open(r'data\dataset_preprocessed.pickle', 'rb'))

classifiers = [(SVC(kernel='rbf', C=1, gamma=1), 'svm_rbf'),
               (SVC(kernel='linear'), 'svm_linear'),
               (KNeighborsClassifier(), 'knn'),
               (MultinomialNB(), 'naive_bayes')]

vectorizers = [(TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True), 'tfidf'),
               (CountVectorizer(max_df=0.95, min_df=2), 'count'),
               (HashingVectorizer(), 'hash')]

'''
cl = Classifier(SVC(), (TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)))

x = list(map(lambda a: a[0], dataset))
y = list(map(lambda a: a[1], dataset))

c_avg = []
gamma_avg = []

for i in range(0, 10):
    r = cl.svc_param_selection(x, y, 10, "grid")
    print(r)
    c_avg.append(r['C'])
    gamma_avg.append(r['gamma'])

print(numpy.average(c_avg))
print(numpy.average(gamma_avg))
'''


runs = 2
for clf in classifiers:
    for vec in vectorizers:
        if clf[1] == 'naive_bayes' and vec[1] == 'hash':
            continue
        print(vec[1] + ' and ' + clf[1])
        mr_avg = []
        acc_avg = []

        file_path = 'results/' + vec[1] + '-' + clf[1] + '.txt'
        file = open(file_path, 'w')

        for i in range(0, runs):
            dataset = sample(dataset, len(dataset))
            c = Classifier(clf[0], vec[0])
            r = c.evaluate_score(dataset, 10)
            mr_avg.append(r[0])
            acc_avg.append(r[1])
            file.write(str(r[0]) + ',' + str(r[1]) + '\n')
        file.close()
        print(numpy.mean(mr_avg), numpy.mean(acc_avg))
