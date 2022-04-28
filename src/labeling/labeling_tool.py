import os

import polars as pl
from bson import ObjectId
from pymongo import MongoClient
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.style import Style
from rich.text import Text

from src.utils.logging import log


class MongoConnector:
    """A connector to MongoDB where the data are stored."""

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        db: str,
        collection: str,
        authSource: str = "admin",
    ) -> None:
        self.client = MongoClient(
            f"mongodb://{user}:{password}@{host}:{port}/{db}", authSource=authSource
        )
        self.db = self.client[db]
        self.collection = self.db[collection]

    def fill_db_from_frame(self, df: pl.DataFrame) -> None:
        """
        Fills the database with the data from the given dataframe.

        Args:
            df: The pl.DataFrame to fill the database with. Columns "entropy" and "label" are initialized with 1 and null respectively.
        """
        df = df.with_columns(
            [
                pl.lit(1).alias("entropy"),  # don't use 0: NaNs in sampling
                pl.lit(None).alias("label"),
            ]
        )
        rows = list(df.to_dicts())
        self.collection.insert_many(rows)
        log.info(
            f"Imported data to MongoDB. Total rows in frame: {df.height}, total docs in DB: {self.collection.count_documents({})}"
        )

    def get_all_data_as_frame(self) -> pl.DataFrame:
        """
        Returns all data from the database as a pl.DataFrame.

        Returns:
            A pl.DataFrame with all data from the database.
        """
        result = list(self.collection.find({}))
        for rr in result:
            rr["_id"] = str(rr["_id"])

        return pl.from_dicts(result)


class DataLabeler:
    """Class for data labeling logic."""

    def __init__(self, conn: MongoConnector) -> None:
        self.conn = conn
        self.console = Console()
        self.inmem_df = self.conn.get_all_data_as_frame()
        self.legend = "1...POS\t2...NEU\t3...NEG"

    def show_doc_by_id(self, doc_id: str, status: str):
        """
        Shows the document with the given id and status to the user.

        Args:
            doc_id: The MongoDB _id of the document to show (as a string).
            status: The status to show to the user.
        """

        self.console.clear()

        sample = self.inmem_df.filter(pl.col("_id") == doc_id).to_dicts()[0]

        p = Panel(
            sample.get("tweet"),
            title=Text(f"id = {sample['_id']}"),
            border_style=Style(dim=True),
        )

        self.console.print(p)
        self.console.print(Align(status, align="center"))
        self.console.print(
            Align("[grey]" + self.legend + "[/]", align="center"), style=Style(dim=True)
        )

        self.console.print("\n" * 2)
        choice = Prompt.ask("Choose", choices=["1", "2", "3"])
        return choice

    def sample_one_from_frame(self) -> str:
        """
        Samples examples that are a) unlabeled and b) have a high entropy.

        Returns:
            The _id of the sampled example as a string.
        """
        unlabeled_df = self.inmem_df.filter(pl.col("label").is_null())

        highest_entropy = (
            unlabeled_df.select(pl.col("entropy").max()).to_numpy().ravel()[0]
        )
        return (
            unlabeled_df.filter(pl.col("entropy") == highest_entropy)
            .sample(n=1)
            .select("_id")
            .to_series()
            .to_list()[0]
        )

    def update_one_doc(self, doc_id, label) -> None:
        """
        Updates a single document in the database and in-memory frame.

        Args:
            doc_id: The MongoDB _id of the document to update.
            label: The label to update the document with.
        """
        update_result = self.conn.collection.update_one(
            {"_id": ObjectId(doc_id)}, {"$set": {"label": label}}
        )
        if update_result.matched_count != 1:
            raise ValueError(
                f"Update failed for _id = {doc_id}. Matched count: {update_result.matched_count}"
            )

        self.inmem_df = self.inmem_df.with_column(
            pl.when(pl.col("_id") == doc_id)
            .then(label)
            .otherwise(pl.col("label"))
            .alias("label")
        )

    def run(self):
        while True:
            total_docs = df.height
            labeled_docs = (
                self.inmem_df.select(pl.col("label").is_not_null().cast(pl.Int16).sum())
                .to_numpy()
                .ravel()[0]
            )
            status = f"{labeled_docs} docs labeled, {total_docs} docs in DB, {total_docs - labeled_docs} docs remaining."

            if labeled_docs % 3 == 0 and labeled_docs > 0:
                log.warning("Please train active learning model now!")
                exit(0)  # TODO: train AL model here and update entropy in Mongodb

            id_to_be_labeled = self.sample_one_from_frame()
            label = self.show_doc_by_id(id_to_be_labeled, status=status)
            self.update_one_doc(id_to_be_labeled, label)


if __name__ == "__main__":
    user = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWD")
    conn = MongoConnector(
        user,
        password,
        host="localhost",
        port=27017,
        db="data_labeling",
        collection="dev_coll",
        authSource="data_labeling",
    )

    df = pl.DataFrame(
        {
            "user": ["a", "b", "c", "d", "e", "f", "g"],
            "tweet": ["AASDF", "BASDF", "CASDF", "DASDF", "EASDF", "FASDF", "GASDF"],
        }
    )

    # conn.collection.drop()
    conn.fill_db_from_frame(df)
    print(conn.get_all_data_as_frame())

    labeler = DataLabeler(conn)
    labeler.run()
