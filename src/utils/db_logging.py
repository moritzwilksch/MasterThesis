import logging
import os

from pymongo import MongoClient
from pymongo.collection import Collection
from rich.console import Console


class MongoLogger:
    def __init__(
        self,
        host: str,
        port: int,
        db: str,
        coll: str,
        user: str = None,
        passwd: str = None,
        level: int = logging.INFO,
    ):
        if user is None:
            user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
        if passwd is None:
            passwd = os.getenv("MONGO_INITDB_ROOT_PASSWORD")

        client = MongoClient(
            f"mongodb://{user}:{passwd}@{host}:{port}", authSource="admin"
        )
        _db = client[db]
        self.coll: Collection = _db[coll]
        self.level = level

        self.console = Console()
        self.color_map = {
            logging.DEBUG: "green",
            logging.INFO: "blue",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
        }
        self.level_map = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARN",
            logging.ERROR: "ERROR",
        }

    # ----------------------- private methods -----------------------
    def _print_message(self, message: str, level: int) -> None:
        self.console.print(
            f"[{self.color_map.get(level)}][{self.level_map.get(level)}][/]\t{message}"
        )

    def _log_message(self, message: str, level: int) -> None:
        self.coll.insert_one({"msg": message, "lvl": level})
        if level >= self.level:
            self._print_message(message=message, level=level)

    # ----------------------- public methods -----------------------
    def debug(self, message: str) -> None:
        self._log_message(message=message, level=logging.DEBUG)

    def info(self, message: str) -> None:
        self._log_message(message=message, level=logging.INFO)

    def warning(self, message: str) -> None:
        self._log_message(message=message, level=logging.WARNING)

    def error(self, message: str) -> None:
        self._log_message(message=message, level=logging.ERROR)


logger = MongoLogger(
    host="65.108.135.187",
    port=27017,
    db="logging",
    coll="thesis",
    level=logging.DEBUG,
)
