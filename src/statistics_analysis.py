from scipy import stats
import os
import numpy as np

samples_avg = set()

for file1 in os.listdir('results'):
    for file2 in os.listdir('results'):
        if file1 != file2:
            f1 = open('results/' + file1)
            f2 = open('results/' + file2)

            sample1 = list(map(lambda x: float(x.replace('\n', '').split(',')[0]), f1.readlines()))
            sample2 = list(map(lambda x: float(x.replace('\n', '').split(',')[0]), f2.readlines()))

            sample1_avg = np.average(sample1)
            sample2_avg = np.average(sample2)

            samples_avg.add((np.average(sample1), file1.replace('.txt', '')))

            d = stats.ttest_ind(sample1, sample2)

            print('Test Between:')
            print(file1.replace('.txt', '') + ' avg = ' + str(sample1_avg))
            print(file2.replace('.txt', '') + ' avg = ' + str(sample2_avg))
            print(d.pvalue)
            print('-------------------------------------------------------------')

for s in sorted(samples_avg, reverse=True):
    print(s)
