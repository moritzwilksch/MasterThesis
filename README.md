# 🎓 Master Thesis <img width=90 align="right" src="https://www.uni-potsdam.de/fileadmin/projects/zim/images/logos/Unilogo.svg">
> My Master Thesis. WIP.  

***See the actual write up in [this other repo](https://github.com/moritzwilksch/MasterThesisWriting)***

# The Project
> Developing a sentiment analysis model for financial social media posts

## The Problem
There is loads of research on sentiment analysis models for social media posts (Hutto & Gilbert, 2014; Barbierie et al., 2020) and on sentiment analysis of financial texts like news and corporate filings (Loughran & McDonald, 2011; Araci, 2019). However, the research on financial social media posts (think StockTwits, Reddit r/wallstreetbets, and Twitter) is limited.

## The Status-Quo
Researchers often utilize sentiment models from the adjacent domains of *finance* or *generic social media*. Therefore, be benchmark the most common models: VADER (Hutto & Gilbert, 2014), NTUSD-Fin (Chen et al., 2018), FinBERT (Araci, 2019), and TwitterRoBERTa (Barbierie et al., 2020)

## The Solution
We collect and label 10,000 tweets and train a varietiy of sentiment analysis models comparing their performance and compute footprints. The detailed methodology can be found [here](https://www.github.com/moritzwilksch/MasterThesisWriting). The final models will be open-sourced and availabe for anyone to use as [pyFin-sentiment](https://www.github.com/moritzwilksch/pyfin-sentiment): a python package for sentiment analysis of financial social media posts.

# Performance
## On Tweets
> Out-of-sample ROC AUC of proposed and existing models on the collected dataset of 10,000 tweets.
<img width="1697" alt="image" src="https://user-images.githubusercontent.com/58488209/182564752-176f8c68-4f34-4ba2-9a3f-160444d02d96.png">

## On StockTwits Posts
> Out-of-sample ROC AUC of proposed and existing models on a dataset of StockTwits posts.  

Using the Fin-SoMe dataset compiled by Chen et al. (2020)
<img width="1774" alt="image" src="https://user-images.githubusercontent.com/58488209/182565301-5ea19d97-c2fb-4a00-9629-49e9360acdec.png">

## Resourcefulness
Measured as inference time per sample (ms) on a system with an AMD Ryzen 5 3600 CPU and 64GB of RAM
<img width="1774" alt="image" src="https://user-images.githubusercontent.com/58488209/183087415-705820a4-238a-40d4-b150-0af63ee7371d.png">

# `pyFin-Sentiment`
This work set out to publish a ***usable model artifact*** to provide future research with more accurate sentiment assessments. We therefore publish the proposed logistc regression model in an easy-to-use python library called [pyFin-Sentiment](https://github.com/moritzwilksch/pyfin-sentiment)

# References
1) Araci, D. (2019). Finbert: Financial sentiment analysiswith pre-trained language models. arXiv preprint arXiv:1908.10063
2) Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L. (2020). Tweeteval: Uniﬁed benchmark and comparative evaluation for tweet classiﬁcation. arXiv preprint arXiv:2010.12421.
3) Chen, C.-C., Huang, H.-H., & Chen, H.-H. (2018). Ntusd-ﬁn: a market sentiment dictionary for ﬁnancial social media data applications. In Proceedings of the 1st ﬁnancial narrative processing workshop (fnp 2018).
3) Chen, C.-C., Huang, H.-H., & Chen, H.-H. (2020). Issues and perspectives from 10,000 annotated ﬁnancial social media data. In Proceedings of the 12th language resources and evaluation conference (pp. 6106–6110).
4) Hutto, C., &Gilbert, E. (2014). Vader: Aparsimonious rule-based model for sentiment analysis of social media text. InProceedings ofthe international aaai conference on web andsocial media (Vol. 8, pp. 216–225).
5) Loughran, T.,&McDonald, B. (2011).When is aliabilitynotaliability? textual analysis, dictionaries, and 10-ks. The Journal ofﬁnance, 66(1), 35–65.
