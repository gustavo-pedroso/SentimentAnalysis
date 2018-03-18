import pickle
import re

dataset_file = open(r'C:\Users\gusta\Desktop\dataset.pickle', 'rb')
dataset_file1 = open(r'C:\Users\gusta\Desktop\dataset_cleared.pickle', 'wb')


dataset = pickle.load(dataset_file)
dataset_cleared = set()

for d in dataset:
    text = d[0]
    between_quotes = re.findall(r'"([^"]*)"', text)

    if len(between_quotes) > 0:
        text = between_quotes[0]
    else:
        text = ''

    text = text.replace('"', '')
    if text != '':
        dataset_cleared.add((text, d[1]))

for d in dataset_cleared:
    if d[1] != 'positive' and d[1] != 'negative':
        print(d[1])

dataset_cleared = list(dataset_cleared)

pickle.dump(dataset_cleared, dataset_file1)
dataset_file1.close()




