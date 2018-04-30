import pickle
import src.conf as conf
from urllib.request import urlopen
import re

lines = open(conf.project_path + '/data/data1.txt', 'r').readlines()

ids_labels = []

print(len(lines))

for line in lines:
    line = line.split('\t')
    tweet_id = line[0]
    label = line[2].replace('\n', '')

    if label != 'neutral':
        ids_labels.append((tweet_id, label))

dataset_file = open(conf.project_path + '/data/dataset1.pickle', 'wb')
dataset = []

curr = 0
total = len(ids_labels)


for inst in ids_labels:
    print(str(curr) + ' from ' + str(total) + ' -> dataset length = ' + str(len(dataset)))
    curr += 1
    tweet_id = str(inst[0])
    url = 'https://twitter.com/statuses/' + tweet_id

    try:
        html_string = str(urlopen(url).read())
        data = re.findall(r'<title>.*?<\/title>', html_string)[0]
        data = re.findall(r'&quot;.*?&quot;', data)[0].replace('&quot;', '')
        dataset.append((data, inst[1]))
    except Exception:
        print(tweet_id)
        continue

pickle.dump(dataset, dataset_file)
dataset_file.close()
