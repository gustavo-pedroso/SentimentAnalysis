from pymongo import MongoClient


client = MongoClient('localhost', 27017)
db = client['tcc']
collection = db['tweets']

for doc in collection.find({"classified": True}).limit(10):
    print(doc['text'])

