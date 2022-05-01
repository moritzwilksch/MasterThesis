import json

import polars as pl
import requests
from jsonpath_ng import parse


class StocktwitsScraper:
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size

        self.json_paths = {
            "body": parse("messages.[*].body"),
            "created_at": parse("messages.[*].created_at"),
            "id": parse("messages.[*].id"),
            "sentiment": parse("messages.[*].entities.sentiment"),
        }

        self.base_url = "https://api.stocktwits.com/api/2/streams/symbol/"
        self.headers = {
            "cookie": "session_visits_count=2",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:100.0) Gecko/20100101 Firefox/100.0",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://stocktwits.com/",
            "Origin": "https://stocktwits.com",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "TE": "trailers",
        }

    def _fetch_one_batch(self, ticker: str, max: str = None):

        # with open("data/raw/stocktwits_stub.json") as f:
        #     return json.load(f)

        url = f"{self.base_url}/{ticker.upper()}.json"
        querystring = {"filter": "top", "limit": str(self.batch_size)}
        if max is not None:
            querystring["max"] = max

        response = requests.request(
            "GET", url, headers=self.headers, params=querystring
        )
        return response.json()

    def _get_one_batch_as_df(self, ticker: str, max: str = None) -> pl.DataFrame:
        data = self._fetch_one_batch(ticker, max=max)
        parsed_data = dict()
        for key, json_path in self.json_paths.items():
            # for sentiment key only: yields {"basic": <SENTI>} or None
            # thus: catch this case
            if key == "sentiment":
                parsed_data[key] = [
                    m.value.get("basic") if m.value is not None else None
                    for m in json_path.find(data)
                ]
            else:
                parsed_data[key] = [m.value for m in json_path.find(data)]

        return pl.from_dict(parsed_data)

    def run(self, ticker: str) -> pl.DataFrame:
        dfs = []
        max = None
        counter = 0

        while True:
            print(f"Fetching batch {counter}")

            df = self._get_one_batch_as_df(ticker, max=max)
            counter += 1
            dfs.append(df)
            all_data = pl.concat(dfs)

            if all_data.height > 5000:
                break

            max = all_data.select(pl.col("id").min()).to_numpy().ravel()[0]
            print(max)
            print(all_data)

            print(
                all_data.select(
                    pl.col("created_at")
                    .str.strptime(pl.Datetime, r"%Y-%m-%dT%H:%M:%SZ", strict=False)
                    .min()
                )
                .to_numpy()
                .ravel()[0]
            )

        return all_data


if __name__ == "__main__":
    # with open("data/raw/stocktwits_stub.json", "w") as f:
    #     json.dump(StocktwitsScraper().run("TSLA"), f)

    result = StocktwitsScraper(batch_size=100).run("TSLA")
    result.to_parquet("data/raw/stocktwits_tsla.parquet")
    print(result)
