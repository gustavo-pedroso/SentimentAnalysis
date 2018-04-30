from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from src.preprocessor import Preprocessor
from random import sample
import numpy as np

import src.conf as conf
import pickle

test = open(conf.project_path + '/data/SemEval2016-task4-test.subtask-BD.txt', 'r').readlines()
gold = open(conf.project_path + '/data/SemEval2016_task4_subtaskB_test_gold.txt', 'r').readlines()

p = Preprocessor()

preprocess = False

if preprocess:
    dataset = [p.preprocess(t.split('\t')[3].replace('\n', '')) for t in test]
    pickle.dump(dataset, open(conf.project_path + 'data/test_preprocessed.pickle', 'wb'))
else:
    dataset = pickle.load(open(conf.project_path + 'data/test_preprocessed.pickle', 'rb'))


labels = [t.split('\t')[2].replace('\n', '') for t in gold]

test_dataset = [(example, label) for example, label in zip(dataset, labels)]
train_dataset = pickle.load(open(conf.project_path + 'data\dataset_preprocessed.pickle', 'rb'))

train_dataset = [x for x in train_dataset if len(x[0].split()) > 0]
train_dataset = list(set(train_dataset))

print(len(train_dataset))

x_test = [x[0] for x in test_dataset]

train_dataset = [x for x in train_dataset if x[0] not in x_test]

m_avg = []
a_avg = []

for i in range(0, 10):
    train_dataset1 = sample(train_dataset, len(train_dataset))[0:4346]

    x_train = [x[0] for x in train_dataset1]
    y_train = [x[1] for x in train_dataset1]

    x_test = [x[0] for x in test_dataset]
    y_test = [x[1] for x in test_dataset]

    vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, sublinear_tf=True, use_idf=True)
    classifier = SVC(kernel='rbf', C=1.1, gamma=1)

    train_vectors = vectorizer.fit_transform(x_train)
    classifier.fit(train_vectors, y_train)

    x_test = vectorizer.transform(x_test)

    y_pred = classifier.predict(x_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    macroaveraged_recall = 1/2*((tp / (tp + fn)) + (tn / (tn + fp)))
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    m_avg.append(macroaveraged_recall)
    a_avg.append(accuracy)
    print(macroaveraged_recall, accuracy)

print(np.average(m_avg), np.average(a_avg))
