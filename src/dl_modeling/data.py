import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torchtext
from sklearn.model_selection import KFold, train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from src.utils.preprocessing import Preprocessor

prepper = Preprocessor()


class TweetDataSet(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.data = df.reset_index(drop=True)  # will the index fck things up?

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, "text"], int(self.data.loc[idx, "label"])


class BertTensorDataSet(torch.utils.data.Dataset):
    def __init__(self, data_prefix: str):
        tensors = []
        for ii in range(20):
            tensors.append(torch.load(f"data/distilbert/{data_prefix}_{ii}.pt"))
        self.all_data = torch.cat(tensors, dim=0).clone().detach()
        self.labels = (
            pl.read_parquet("data/labeled/labeled_tweets.parquet")["label"]
            .to_pandas()
            .map({"0": 1, "1": 0, "2": 1, "3": 2})  # off-by-one!
            .to_numpy()
        )

    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        return self.all_data[idx], self.labels[idx]


class BERTTensorDataModule(ptl.LightningDataModule):
    def __init__(self, split_idx, data_prefix: str, batch_size: int = 64):
        super().__init__()
        self.dataset = BertTensorDataSet(data_prefix=data_prefix)
        self.split_idx = split_idx
        self.batch_size = batch_size

        self.trainval_idxs, self.test_idxs = train_test_split(
            np.arange(len(self.dataset)),
            shuffle=True,
            random_state=42,
            test_size=0.25,  # hold-out test set
        )
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # these map split_idx i -> data indeces for split i
        self.split_idxs_train = {}
        self.split_idxs_val = {}

        for split_idx, (train_idx, val_idx) in enumerate(
            self.kfold.split(self.trainval_idxs)
        ):
            self.split_idxs_train[split_idx] = self.trainval_idxs[train_idx]
            self.split_idxs_val[split_idx] = self.trainval_idxs[val_idx]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(
                self.split_idxs_train[self.split_idx]
            ),
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(self.dataset, self.split_idxs_val[self.split_idx]),
            batch_size=1024,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(self.dataset, self.test_idxs),
            batch_size=1024,
        )

    def trainval_dataloader_for_retraining(self):
        train_idxs, val_idxs = train_test_split(
            self.trainval_idxs, test_size=0.1, shuffle=True
        )
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idxs),
            num_workers=4,
        ), torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_idxs),
            num_workers=1,
        )


class TweetDataModule(ptl.LightningDataModule):
    def __init__(
        self, split_idx, batch_size: int, model_type: str = "recurrent", all_data=None
    ):
        super().__init__()
        if split_idx == "retrain":
            RETRAIN = True
            split_idx = 0
        else:
            RETRAIN = False

        print(f"Tokenizer retrain: {RETRAIN}")
        self.split_idx = split_idx
        self.batch_size = batch_size
        self.collate_fn_to_use = (
            self.collate_fn
            if model_type == "recurrent"
            else self.transformer_collate_fn
            if model_type == "transformer"
            else None
        )

        self.all_data = pl.read_parquet("data/labeled/labeled_tweets.parquet")

        # Override for benchmarking on FinSoMe
        if all_data is not None:
            self.all_data = all_data

        self.all_data = self.all_data.with_column(
            pl.when(pl.col("label") == "0")
            .then(pl.lit("2"))
            .otherwise(pl.col("label"))
            .cast(pl.Int32)
            .alias("label")
        ).with_column(pl.col("label") - 1)

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

        from textaugment import EDA

        t = EDA()

        def augment_text(text: str) -> str:
            text = t.synonym_replacement(text, n=2)
            return text

        dftrain = pd.concat(
            [
                self.xtrainval.iloc[self.split_idxs_train[self.split_idx]],
                self.ytrainval.iloc[self.split_idxs_train[self.split_idx]],
            ],
            axis=1,
        )
        
        self.dfval = pd.concat(
            [
                self.xtrainval.iloc[self.split_idxs_val[self.split_idx]],
                self.ytrainval.iloc[self.split_idxs_val[self.split_idx]],
            ],
            axis=1,
        )
        augmented = pl.from_pandas(dftrain).select(
            [pl.col("text").apply(augment_text).alias("text"), pl.col("label")]
        )

        self.dftrain = pd.concat([dftrain, augmented.to_pandas()]).reset_index(drop=True)

    

        if RETRAIN:
            self.tokenizer = torchtext.data.functional.sentencepiece_numericalizer(
                torchtext.data.functional.load_sp_model(
                    f"outputs/tokenizers/retraining_trainval.model"
                )
            )
        else:
            sp_model = torchtext.data.functional.load_sp_model(
                f"outputs/tokenizers/split_{split_idx}.model"
            )
            self.tokenizer = torchtext.data.functional.sentencepiece_numericalizer(
                sp_model=sp_model
            )

    def train_dataloader(self):
        df_for_split = self.dftrain

        return torch.utils.data.DataLoader(
            TweetDataSet(df_for_split),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_to_use,
            num_workers=4,
        )

    def val_dataloader(self):
        df_for_split = self.dfval
        return torch.utils.data.DataLoader(
            TweetDataSet(df_for_split),
            batch_size=1024,
            collate_fn=self.collate_fn_to_use,
            num_workers=1,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            TweetDataSet(pd.concat([self.xtest, self.ytest], axis=1)),
            batch_size=1024,
            collate_fn=self.collate_fn_to_use,
        )

    def trainval_dataloader_for_retraining(self):
        df_for_split = pd.concat([self.xtrainval, self.ytrainval], axis=1)
        train, val = train_test_split(df_for_split, test_size=0.1, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(
            TweetDataSet(train),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_to_use,
            num_workers=2,
        )
        val_dataloader = torch.utils.data.DataLoader(
            TweetDataSet(val),
            batch_size=1024,
            collate_fn=self.collate_fn_to_use,
            num_workers=2,
        )

        return train_dataloader, val_dataloader

    def all_dataloader(self):
        return torch.utils.data.DataLoader(
            TweetDataSet(self.all_data),
            batch_size=1024,
            collate_fn=self.collate_fn_to_use,
            num_workers=2,
        )

    def collate_fn(self, batch):
        text_tensors = []
        labels = []
        seq_lens = []

        for text, label in batch:
            # tokens = self.vocab(self.tokenizer(text))
            tokens = list(self.tokenizer([text]))[0]
            text_tensors.append(torch.Tensor(tokens).long())
            labels.append(label)
            seq_lens.append(len(tokens))

        padded_sequences = nn.utils.rnn.pad_sequence(text_tensors)
        return padded_sequences, seq_lens, torch.Tensor(labels).long()

    def transformer_collate_fn(self, batch):
        text_tensors = []
        labels = []
        seq_lens = []
        masks = []

        for text, label in batch:
            # tokens = self.vocab(self.tokenizer(text))
            tokens = list(self.tokenizer([text]))[0]
            text_tensors.append(torch.Tensor(tokens).long())
            labels.append(label)
            seq_lens.append(len(tokens))

        max_seq_len = max(seq_lens)
        for l in seq_lens:
            mask = torch.full((max_seq_len,), True)
            mask[:l] = False
            masks.append(mask.bool())

        padded_sequences = nn.utils.rnn.pad_sequence(text_tensors)

        # repeat mask tensor for each head
        masks = torch.stack(masks, dim=0)
        masks = torch.cat([masks] * 1, dim=0)

        return padded_sequences, masks, torch.Tensor(labels).long()
