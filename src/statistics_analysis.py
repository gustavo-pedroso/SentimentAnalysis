from scipy import stats
import os
import numpy as np
import src.conf as conf

samples_avg = set()

results_path = conf.project_path + '/results/'

for file1 in os.listdir(results_path):
    for file2 in os.listdir(results_path):
        if file1 != file2:  # and file1 == 'tfidf-svm_rbf.txt':
            f1 = open(results_path + file1)
            f2 = open(results_path + file2)

            sample_acc = list(map(lambda x: float(x.replace('\n', '').split(',')[1]), f1.readlines()))
            f1.close()
            f1 = open(results_path + file1)
            sample1 = list(map(lambda x: float(x.replace('\n', '').split(',')[0]), f1.readlines()))
            sample2 = list(map(lambda x: float(x.replace('\n', '').split(',')[0]), f2.readlines()))

            sample1_avg = np.average(sample1)
            sample2_avg = np.average(sample2)

            samples_avg.add((file1.replace('.txt', ''), sample1_avg, np.average(sample_acc)))

            d = stats.ttest_ind(sample1, sample2)

            '''print('Test Between:')
            print(file1.replace('.txt', '') + ' avg = ' + str(sample1_avg))
            print(file2.replace('.txt', '') + ' avg = ' + str(sample2_avg))
            print(d)
            print(d.pvalue)
            print('-------------------------------------------------------------')'''

            print(file1.replace('.txt', ''), file2.replace('.txt', ''), d.pvalue)

for s in sorted(samples_avg, key=lambda x: x[1], reverse=True):
    print(s)
