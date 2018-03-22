from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy
from sklearn.metrics import confusion_matrix
import scipy


class Classifier:

    def __init__(self, classifier, vectorizer):
        self.classifier = classifier
        self.vectorizer = vectorizer

    def train(self, x_train, y_train):
        train_vectors = self.vectorizer.fit_transform(x_train)
        self.classifier.fit(train_vectors, y_train)
        print('classifier trained on', len(y_train), 'examples')

    def evaluate_score(self, data, folds):
        kf = KFold(n_splits=folds)
        mr_avg = []
        acc_avg = []
        for train, test in kf.split(data):

            train_data = numpy.array(data)[train]
            x_train = list(map(lambda x: x[0], train_data))
            y_train = list(map(lambda x: x[1], train_data))

            test_data = numpy.array(data)[test]
            x_test = list(map(lambda x: x[0], test_data))
            y_test = list(map(lambda x: x[1], test_data))

            train_vectors = self.vectorizer.fit_transform(x_train)
            self.classifier.fit(train_vectors, y_train)

            x_test = self.vectorizer.transform(x_test)

            y_pred = self.classifier.predict(x_test)

            nn, np, pn, pp = confusion_matrix(y_test, y_pred).ravel()

            macroaveraged_recall = 1/2*((pp / (pp + np)) + (nn / (nn + pn)))
            accuracy = (pp + nn) / (pp + nn + np + pn)

            if macroaveraged_recall is not None:
                mr_avg.append(macroaveraged_recall)
                acc_avg.append(accuracy)

            return numpy.mean(mr_avg), numpy.mean(acc_avg)

    def predict(self, example):
        example = self.vectorizer.transform([example])
        return self.classifier.predict(example)

    def svc_param_selection(self, train_data, train_labels, nfolds, searchType):
        train_vectors = self.vectorizer.fit_transform(train_data)

        if searchType == "grid":
            cs = [0.9, 1, 1.1]
            gammas = [0.9, 1, 1.1]
            param_grid = {'C': cs, 'gamma': gammas}
            grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, scoring="recall_macro")
        elif searchType == "random":
            cs = scipy.stats.expon(scale=100)
            gammas = scipy.stats.expon(scale=0.1)
            param_grid = {'C': cs, 'gamma': gammas}
            grid_search = RandomizedSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, scoring="recall_macro")

        grid_search.fit(train_vectors, train_labels)
        grid_search.best_params_
        return grid_search.best_params_


