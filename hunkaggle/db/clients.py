'''
Created on Jul 26, 2014

@author: c3h3
'''

from pymongo import MongoClient
from ..settings import MONGOHOST,MONGOPORT

mdb_client = MongoClient(host=MONGOHOST, port=MONGOPORT)


