from pydoc import TextDoc

import polars as pl


class Preprocessor:
    TEXTCOL = "text"
    CASHTAG_REGEX = r"\$[a-zA-Z]+"
    MENTION_REGEX = r"@[A-Za-z0-9_]+"

    def __init__(
        self,
        symbols: bool = True,
        numbers: bool = True,
        cashtags: bool = True,
        mentions: bool = True,
        lowercase: bool = True,
    ):
        self.symbols = symbols
        self.numbers = numbers
        self.cashtags = cashtags
        self.mentions = mentions
        self.lowercase = lowercase

    def fix_symbols(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.col(self.TEXTCOL)
            .str.replace_all(r"&gt;", ">")
            .str.replace_all(r"&lt;", "<")
            .str.replace_all(r"&amp;", "&")
        )

    def prep_numbers(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(pl.col(self.TEXTCOL).str.replace_all(r"\d", "9"))

    def prep_cashtags(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.col(self.TEXTCOL).str.replace_all(self.CASHTAG_REGEX, "$TICK")
        )

    def prep_mentions(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.col(self.TEXTCOL).str.replace_all(self.MENTION_REGEX, "@USER")
        )

    def prep_lowercase(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(pl.col(self.TEXTCOL).str.to_lowercase())

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.symbols:
            df = self.fix_symbols(df)
        if self.numbers:
            df = self.prep_numbers(df)
        if self.cashtags:
            df = self.prep_cashtags(df)
        if self.mentions:
            df = self.prep_mentions(df)
        if self.lowercase:
            df = self.prep_lowercase(df)
        return df
