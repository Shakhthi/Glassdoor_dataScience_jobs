import os
import sys

from src.exception.exception_handler import ExceptionHandler
from src.logging.logger import logging

import mysql.connector

from dotenv import load_dotenv
load_dotenv()

IP = os.getenv("MYSQL_HOST")
PORT = os.getenv("PORT")
PASSCODE = os.getenv("MYSQL_PASSCODE")

try: 
    logging.info("Connecting to MySQL...")
    connection = mysql.connector.connect(
        host = IP,
        port = PORT,
        user = "root",
        password = PASSCODE
    )

    if connection.is_connected():
        cursor = connection.cursor()

        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        
        for db in databases:
            print(f"- {db[0]}")

except mysql.connector.Error as e:
    raise ExceptionHandler(e, sys)

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        logging.info("MySQL connection is closed")
