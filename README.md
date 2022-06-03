# 🎓 Master Thesis <img width=90 align="right" src="https://www.uni-potsdam.de/fileadmin/projects/zim/images/logos/Unilogo.svg">
> My Master Thesis. WIP.  

***See the actual write up in [this other repo](https://github.com/moritzwilksch/MasterThesisWriting)***

# The Project
> Developing a sentiment analysis model for financial social media posts

## The Problem
There is loads of research on sentiment analysis models for social media posts (Hutto & Gilbert, 2014; Barbierie et al., 2020) and on sentiment analysis of financial texts like news and corporate filings (Loughran & McDonald, 2011; Araci, 2019). However, the research on financial social media posts (think StockTwits, Reddit r/wallstreetbets, and Twitter) is limited.

## The Solution
We collect and label 10,000 tweets and train a varietiy of sentiment analysis models comparing their performance and compute footprints. The detailed methodology can be found [here](https://www.github.com/moritzwilksch/MasterThesisWriting). The final models will be open-sourced and availabe for researchers and practitionars as [pyFin-sentiment](https://www.github.com/moritzwilksch/pyfin-sentiment), a python package for sentiment analysis of financial social media posts.