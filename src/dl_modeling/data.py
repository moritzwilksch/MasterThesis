import pytorch_lightning as ptl
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextPreprocessor:
    def __init__(
        self,
    ):
        self.tokenizer = get_tokenizer("basic_english")


class TweetDataSet(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.data = df.reset_index(drop=True)  # will the index fck things up?
        self.data["label"] = self.data["label"].map(
            {"0": "2", "1": "1", "2": "2", "3": "3"}
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, "text"], int(self.data.loc[idx, "label"])


class TweetDataModule(ptl.LightningDataModule):
    def __init__(self, split_idx):
        super().__init__()

        self.split_idx = split_idx

        self.all_data: pd.DataFrame = pd.read_parquet(
            "data/labeled/labeled_tweets.parquet"
        )
        self.xtrainval, self.xtest, self.ytrainval, self.ytest = train_test_split(
            self.all_data["text"],
            self.all_data["label"],
            shuffle=True,
            random_state=42,
            test_size=0.25,  # hold-out test set
        )
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # these map split_idx i -> data indeces for split i
        self.split_idxs_train = {}
        self.split_idxs_val = {}

        for split_idx, (train_idx, val_idx) in enumerate(
            self.kfold.split(self.xtrainval)
        ):
            self.split_idxs_train[split_idx] = train_idx
            self.split_idxs_val[split_idx] = val_idx

        # for text processing, set tokenizer and vocab built on train split
        self.tokenizer, self.vocab = self.get_tokenizer_for_split()

    def train_dataloader(self, batch_size: int = 32):
        df_for_split = pd.concat([self.xtrainval, self.ytrainval], axis=1).iloc[
            self.split_idxs_train[self.split_idx]
        ]

        return torch.utils.data.DataLoader(
            TweetDataSet(df_for_split), batch_size=batch_size, collate_fn=self.collate_fn
        )

    def val_dataloader(self, batch_size: int = 32):
        df_for_split = pd.concat([self.xtrainval, self.ytrainval], axis=1).iloc[
            self.split_idxs_val[self.split_idx]
        ]
        return torch.utils.data.DataLoader(
            TweetDataSet(df_for_split), batch_size=batch_size, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            TweetDataSet(pd.concat([self.xtest, self.ytest], axis=1)), batch_size=32
        )

    def get_tokenizer_for_split(self):
        tokenizer = get_tokenizer("basic_english")

        def yield_tokens(data_iter):
            for texts, _ in data_iter:
                for document in texts:
                    yield tokenizer(document)

        # get train data as df
        df_for_split = pd.concat([self.xtrainval, self.ytrainval], axis=1).iloc[
            self.split_idxs_train[self.split_idx]
        ]

        # build vocab - only based on train set
        vocab = build_vocab_from_iterator(
            yield_tokens(
                torch.utils.data.DataLoader(
                    TweetDataSet(df_for_split),
                )
            ),
            specials=["<unk>"],
        )
        vocab.set_default_index(vocab["<unk>"])
        return tokenizer, vocab  # vocab is the trained object

    def collate_fn(self, batch):
        for text, label in batch:
            return torch.Tensor(self.vocab(self.tokenizer(text))).long(), label


if __name__ == "__main__":
    dm = TweetDataModule(split_idx=0)
    for x, y in dm.train_dataloader():
        print(x, y)
        break
