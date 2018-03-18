import pickle
from test import TitleParser
from urllib.request import urlopen
import urllib


lines = open(r'C:\Users\gusta\Desktop\data.txt', 'r').readlines()

ids_labels = []

print(len(lines))

for line in lines:
    line = line.split('\t')
    tweet_id = line[0]
    label = line[1].replace('\n', '')

    if label != 'neutral':
        ids_labels.append((tweet_id, label))

dataset_file = open(r'C:\Users\gusta\Desktop\dataset.pickle', 'wb')
dataset = []

curr = 0
total = len(ids_labels)

tp = TitleParser()
for inst in ids_labels:
    print(str(curr) + ' from ' + str(total) + ' -> dataset length = ' + str(len(dataset)))
    curr += 1
    tweet_id = str(inst[0])
    url = 'https://twitter.com/statuses/' + tweet_id

    try:
        html_string = str(urlopen(url).read())
        tp.feed(html_string)
        dataset.append((tp.title, inst[1]))
    except Exception:
        print(tweet_id)
        continue

pickle.dump(dataset, dataset_file)
dataset_file.close()
