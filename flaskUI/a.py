# Question 1
from pymongo import MongoClient

client = MongoClient('mongodb://user:password@notjyzh.duckdns.org:27017/?authSource=test', 27017)
db = client["kenji2008"]
# cursor =client.list_database_names()

# for db in cursor:
#     print(db)

#create collection
collection = db["crudeoil"]

#first document
cursor = collection.find({'price per barrel'})
for document in cursor:
    print(document)