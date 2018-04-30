import pickle
import re
import src.conf as conf

dataset_file = open(conf.project_path + '/data/dataset1.pickle', 'rb')
dataset_file1 = open(conf.project_path + '/data/dataset_cleared1.pickle', 'wb')


dataset = pickle.load(dataset_file)
dataset_cleared = set()

# for d in dataset:
#     text = d[0]
#     between_quotes = re.findall(r'"([^"]*)"', text)
#
#     if len(between_quotes) > 0:
#         text = between_quotes[0]
#     else:
#         text = ''
#
#     text = text.replace('"', '')
#     if text != '':
#         dataset_cleared.add((text, d[1]))

for d in dataset:
    if d[1] != 'positive' and d[1] != 'negative':
        print(d[1])

dataset_cleared = list(dataset_cleared)

pickle.dump(dataset, dataset_file1)
dataset_file1.close()




