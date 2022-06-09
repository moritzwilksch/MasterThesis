import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import polars as pl
from src.utils.preprocessing import Preprocessor

prepper = Preprocessor()


class TweetDataSet(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.data = df.reset_index(drop=True)  # will the index fck things up?
        self.data["label"] = self.data["label"].map(
            {"0": "1", "1": "0", "2": "1", "3": "2"}  # off-by-one!
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, "text"], int(self.data.loc[idx, "label"])


class TweetDataModule(ptl.LightningDataModule):
    def __init__(self, split_idx, batch_size: int):
        super().__init__()

        self.split_idx = split_idx
        self.batch_size = batch_size

        self.all_data = pl.read_parquet(
            "data/labeled/labeled_tweets.parquet"
        )
        self.all_data = prepper.process(self.all_data).to_pandas()

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

    def train_dataloader(self):
        df_for_split = pd.concat([self.xtrainval, self.ytrainval], axis=1).iloc[
            self.split_idxs_train[self.split_idx]
        ]

        return torch.utils.data.DataLoader(
            TweetDataSet(df_for_split),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        df_for_split = pd.concat([self.xtrainval, self.ytrainval], axis=1).iloc[
            self.split_idxs_val[self.split_idx]
        ]
        return torch.utils.data.DataLoader(
            TweetDataSet(df_for_split),
            batch_size=1024,
            collate_fn=self.collate_fn,
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
        text_tensors = []
        labels = []
        seq_lens = []

        for text, label in batch:
            tokens = self.vocab(self.tokenizer(text))
            text_tensors.append(torch.Tensor(tokens).long())
            labels.append(label)
            seq_lens.append(len(tokens))

        padded_sequences = nn.utils.rnn.pad_sequence(text_tensors)
        return padded_sequences, seq_lens, torch.Tensor(labels).long()


if __name__ == "__main__":
    dm = TweetDataModule(split_idx=0, batch_size=32)
    # for x, _,  y in dm.train_dataloader():
    #     print(x, y)
    #     break

    print(dm.tokenizer("$MU well I'm glad I stayed short on this piece of shiiiiit lol:D 😂😂 $10.23 incoming #savagetrading "))
