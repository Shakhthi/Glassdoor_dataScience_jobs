import os
import sys
import certifi

from src.exception.exception_handler import ExceptionHandler
from src.logging.logger import logging

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv("MONGO_URI")

# Create a new client and connect to the server
client = MongoClient(MONGODB_URL, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    logging.info("Connecting to MongoDB...")
    # Use certifi to get the correct CA certificates for SSL connections

    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    raise ExceptionHandler(e, sys)