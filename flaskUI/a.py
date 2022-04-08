# Question 1
from pymongo import MongoClient
import auth

from timeit import default_timer as timer

start = timer()

# connection
client = MongoClient(auth.connMongo)
# database
db = client["kenji2008"]

# list
quarterList = []
templist = []
tariffList = []
pricePerBarrelList = []
mainCostList = []
# collection
colCrudeOil = db["crudeoil"]
colMaintenance = db["maintenance"]
colTariff = db["tariff"]
colWeather = db["weather"]

#cursor crude oil
cursorCrudeOil = colCrudeOil.find({
    'quarter': {
        '$gt': '0'
    },
    'price_per_barrel': {
        '$gt': '0'
    }
})
for document in cursorCrudeOil:
    pricePerBarrelList.append(document['price_per_barrel'])
    quarterList.append(document['quarter'])

# cursor for temp
cursorTemp = colWeather.find({'temperature': {'$gt': '0'}})
for document in cursorTemp:
    templist.append(document['temperature'])

# cursor for tariff
cursorTariff = colTariff.find({'tariff_per_kwh': {'$gt': '0'}})
for document in cursorTariff:
    tariffList.append(document['tariff_per_kwh'])

# cursor for maintanence
cursorMain = colMaintenance.find({'cost': {'$gt': '0'}})
for document in cursorMain:
    print(document)
    mainCostList.append(document['cost'])

print(quarterList)
# print(templist)
# print(tariffList)
# print(pricePerBarrelList)
# print(mainCostList)

[(2015.1, 27.37, 23.29, 50.13, 21.85), (2015.2, 28.77, 20.87, 41.43, 20.85), (2015.3, 28.73, 22.41, 59.81, 19.25), (2015.4, 28.13, 20.35, 52.79, 22.85), (2016.1, 28.37, 19.5, 39.76, 23.35), (2016.2, 29.07, 17.68, 44.14, 22.35), (2016.3, 28.53, 19.27, 47.45, 24.35), (2016.4, 27.7, 19.13, 50.79, 25.35), (2017.1, 27.07, 20.2, 37.09, 21.75), (2017.2, 28.27, 21.39, 39.09, 23.75), (2017.3, 28.13, 20.72, 44.44, 22.75), (2017.4, 27.33, 20.3, 43.1, 26.75), (2018.1, 27.1, 21.56, 47.45, 22.8), (2018.2, 28.53, 22.15, 45.78, 20.8), (2018.3, 28.33, 23.65, 36.75, 23.8), (2018.4, 27.6, 24.13, 43.44, 21.8), (2019.1, 28.3, 23.85, 40.43, 21.75), (2019.2, 28.77, 22.79, 36.75, 23.75), (2019.3, 29.03, 24.22, 43.1, 22.75), (2019.4, 27.6, 23.43, 42.1, 21.75), (2020.1, 28.07, 24.24, 30.07, 25.58)]

[(2015.1, 27.37, 23.29, 50.13, 21.85), (2015.2, 28.77, 20.87, 41.43, 20.85), (2015.3, 28.73, 22.41, 59.81, 19.25), (2015.4, 28.13, 20.35, 52.79, 22.85), (2016.1, 28.37, 19.5, 39.76, 23.35), (2016.2, 29.07, 17.68, 44.14, 22.35), (2016.3, 28.53, 19.27, 47.45, 24.35), (2016.4, 27.7, 19.13, 50.79, 25.35), (2017.1, 27.07, 20.2, 37.09, 21.75), (2017.2, 28.27, 21.39, 39.09, 23.75), (2017.3, 28.13, 20.72, 44.44, 22.75), (2017.4, 27.33, 20.3, 43.1, 26.75), (2018.1, 27.1, 21.56, 47.45, 22.8), (2018.2, 28.53, 22.15, 45.78, 20.8), (2018.3, 28.33, 23.65, 36.75, 23.8), (2018.4, 27.6, 24.13, 43.44, 21.8), (2019.1, 28.3, 23.85, 40.43, 21.75), (2019.2, 28.77, 22.79, 36.75, 23.75), (2019.3, 29.03, 24.22, 43.1, 22.75), (2019.4, 27.6, 23.43, 42.1, 21.75), (2020.1, 28.07, 24.24, 30.07, 25.58), (2020.2, 28.63, 23.02, 24.39, 26.58), (2020.3, 27.9, 19.6, 22.39, 28.58), (2020.4, 27.57, 21.4, 18.71, 29.58), (2021.1, 27.07, 20.8, 27.73, 30.6), (2021.2, 28.47, 22.6, 33.08, 29.6), (2021.3, 28.17, 22.63, 35.75, 28.6), (2021.4, 27.93, 22.63, 40.13, 27.6)]
# ...
end = timer()
print("Time taken: " + str(end - start) + " seconds")