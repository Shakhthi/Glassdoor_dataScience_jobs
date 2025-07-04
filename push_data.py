import sys
import os

import json

from src.logging.logger import logging
from src.exception.exception_handler import ExceptionHandler

import mysql.connector
import pymongo

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv("MONGO_URI")

IP = os.getenv("MYSQL_HOST")
PORT = os.getenv("PORT")
PASSCODE = os.getenv("MYSQL_PASSCODE")

class loadData:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ExceptionHandler(e, sys)
        
    def csv_to_json_converter(self, train_file_path):
        try:
            logging.info(f"csv to json conversion initiated for {train_file_path}")
            train_data = pd.read_csv(train_file_path)
            train_data.reset_index(drop = True, inplace = True)
            records = list(json.loads(train_data.T.to_json()).values())

            logging.info("csv to json conversion successfull")
            return records
            
        except Exception as e:
            raise ExceptionHandler(e, sys)
        
    def push_data_into_mongoDB(self, database, collection, records):
        try:
            logging.info("pushing data to MongoDB")
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGODB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            logging.info(f"data pushed into MongoDB collection {self.collection} successfully")

            return len(self.records)
        except Exception as e:
            raise ExceptionHandler(e, sys)
        
    def push_data_into_mysql(self, records):
        try:
            logging.info("pushing data into MySQL initiated.")
            self.records = records.to_dict(orient='records')
            mysql_connector = mysql.connector.connect(
                host = IP,
                port = PORT,
                user = "root",
                password = PASSCODE,
                #database = "ds_jobs"
            )

            cursor = mysql_connector.cursor()
            db = "CREATE DATABASE IF NOT EXISTS ds_jobs;"
            cursor.execute(db)
            cursor.execute("USE ds_jobs")

            logging.info("Creating a table if it does not exist")
            create_table_query =""" CREATE TABLE IF NOT EXISTS Jobs (
                                            company VARCHAR(225),
                                            job_title TEXT,
                                            company_rating FLOAT,
                                            job_description TEXT,
                                            location VARCHAR(225),
                                            salary_avg_estimate FLOAT,
                                            salary_estimate_payperiod VARCHAR(225),
                                            company_size VARCHAR(225),
                                            company_founded INT,
                                            employment_type VARCHAR(225),
                                            industry VARCHAR(225),
                                            sector VARCHAR(255),
                                            revenue VARCHAR(225),
                                            career_opportunities_rating FLOAT,
                                            comp_and_benefits_rating FLOAT,
                                            culture_and_values_rating FLOAT,
                                            senior_management_rating FLOAT,
                                            work_life_balance_rating FLOAT )
                                """
            cursor.execute(create_table_query)



            insert_query = """ INSERT INTO 
                                        jobs (company, job_title, company_rating, job_description,
                                                location, salary_avg_estimate, salary_estimate_payperiod,
                                                company_size, company_founded, employment_type,
                                                industry, sector, revenue,
                                                career_opportunities_rating, comp_and_benefits_rating,
                                                culture_and_values_rating, senior_management_rating,
                                                work_life_balance_rating) 
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
            values = [(record["company"],
                       record["job_title"],
                       record["company_rating"],
                       record["job_description"],
                       record["location"],
                       record["salary_avg_estimate"],
                       record["salary_estimate_payperiod"],
                       record["company_size"],
                       record["company_founded"],
                       record["employment_type"],
                       record["industry"],
                       record["sector"],
                       record["revenue"],
                       record["career_opportunities_rating"],
                       record["comp_and_benefits_rating"],
                       record["culture_and_values_rating"],
                       record["senior_management_rating"],
                       record["work_life_balance_rating"]) 
                      for record in self.records]
            
            cursor.executemany(insert_query, values)
            mysql_connector.commit()

            logging.info(f"Successfully inserted {len(values)} records into MySQL")
        
            cursor.close()
            mysql_connector.close()
        except Exception as e:
            if mysql_connector.is_connected():
                cursor.close()
                mysql_connector.close()
            raise ExceptionHandler(e, sys)
        
if __name__ == "__main__":
    data_loader = loadData()
    data_file_path = r"D:\4.Data Wrangling\Datasets\Ds_jobs\Glassdoor_job_postings.csv"
    train_records = data_loader.csv_to_json_converter(train_file_path=data_file_path)
    print(f"Number of records to insert: {len(train_records)}")
    data_loader.push_data_into_mongoDB(database="MK", collection="ds_jobs", records=train_records)
    #test_data = pd.read_csv(test_file_path)
    #data_loader.push_data_into_mysql(test_data)
