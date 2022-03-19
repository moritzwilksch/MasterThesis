import logging

from pymongo import MongoClient


class MongoLogger:
    def __init__(self, level: int = logging.INFO):
        ...
