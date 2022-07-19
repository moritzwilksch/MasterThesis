#%%
import numpy as np
import pandas as pd

df = pd.read_excel("data/semeval/Finance Microblog Scores v1.0.xlsx")

#%%
def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(columns={"Message": "text", "sentiment score": "score"})


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("id").agg({"score": "mean", "source": "first", "text": "first"})


def assign_label(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        label=np.select(
            condlist=[
                data.score >= 0.04,
                data.score <= -0.047,
            ],
            choicelist=[1, 3],
            default=2,
        )
    )


def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    return data.convert_dtypes()


clean = (
    df.copy()
    .pipe(rename_columns)
    .pipe(remove_duplicates)
    .pipe(assign_label)
    .pipe(fix_dtypes)
)

#%%
clean.to_parquet("data/semeval/semeval_clean.parquet")
