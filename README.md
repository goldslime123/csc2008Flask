# csc2008 Database Project

# How to run: 
#### 1. Navigate to flaskUI ```cd flaskUI```.
#### 2. Install environment ```py -m venv env```.
#### 3. Install flask ```pip install flask```.
#### 4. You need to turn on your virtual environment, at directory \flaskUI, type ```env\Scripts\Activate```.
#### 5. Finally run your flask ```flask run```.

# Notes:
There will be red markers on the html file. Do not worry! It is working fine! Its red because it is not a javascript thing.


# How to connect to maria DB and create table with csv data (hdb example):
### Reference link: 
- https://www.youtube.com/watch?v=3hXk9sXBgt8
- https://mariadb.com/resources/blog/how-to-connect-python-programs-to-mariadb/

#### 1. Download all the CSV files
![image](https://user-images.githubusercontent.com/53167249/161244397-4e1d454a-0758-47e2-bf76-fcaa697313af.png)
#### 2. Open maria command prompt and install maria DB python (pip3 install mariadb)
![image](https://user-images.githubusercontent.com/53167249/161245416-3a511068-3cd6-430b-bd35-04ca22e8d341.png)
#### 3. Open maria MySQL Client, check what user you are in, will need it later (status)
![image](https://user-images.githubusercontent.com/53167249/161245865-bf0eab1b-436d-48e3-8377-8672e2dac893.png)
#### 4. Create a database to put all your CSV tables and go into the database (csvfiles)
![image](https://user-images.githubusercontent.com/53167249/161246312-e81f3cd0-6f93-46a1-9a55-20ec6a6ee707.png)
#### 5. Then create necessary tables based on the csv files. One CSV file, one table.
![image](https://user-images.githubusercontent.com/53167249/161246741-33be36ab-cc3a-4bfd-a14d-311793eed164.png)
![image](https://user-images.githubusercontent.com/53167249/161247930-00a604cd-6dfd-405e-9b9c-ed85b0fe909e.png)
CREATE TABLE hdb(id int primary key, quarter varchar(255), premiseType varchar(255), averageElectricConsumption decimal(5,2), averageBill decimal(5,2));
#### 6. Save the data in the CSV to the table in the database, retrieve the path of the specific CSV file and replace 'C:/CSC2008DATA/hdb.csv'
![image](https://user-images.githubusercontent.com/53167249/161248762-4fee558f-f8d9-477f-b2ab-94d21100fe5c.png)
### LOAD DATA LOCAL INFILE 'C:/CSC2008DATA/hdb.csv' INTO TABLE hdb FIELDS TERMINATED BY ',' LINES TERMINATED BY '\r\n' IGNORE 1 LINES (id, quarter, premiseType, averageElectricConsumption, averageBill);
#### 7. Connect to your mariaDB from python (is already inside the code)
![image](https://user-images.githubusercontent.com/53167249/161249150-a1a62190-130d-43ed-90b5-620c0e046472.png)
#### 8. Execute the SQL statements using cursor (the line with the red mark is needed to convert the data receive to float)
![image](https://user-images.githubusercontent.com/53167249/161249443-87cc0142-a2b1-49f8-95dc-8250fd794df1.png)
