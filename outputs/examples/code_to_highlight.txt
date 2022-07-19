pip install pyfin-sentiment


from pyfin_sentiment.model import SentimentModel

# the model only needs to be downloaded once
SentimentModel.download("small")

model = SentimentModel("small")
model.predict(["Long $TSLA!!", "Selling my $AAPL position"])
# array(['1', '3'], dtype=object)