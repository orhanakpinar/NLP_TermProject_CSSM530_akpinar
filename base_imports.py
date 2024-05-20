
import pandas as pd
from scrapy import Selector
import requests
import time
from bs4 import BeautifulSoup
import re
from sklearn.metrics import f1_score, accuracy_score
import spacy
from spacy.lang.tr.stop_words import STOP_WORDS as tr_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from gensim import corpora, models
import gensim
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math
import unicodedata
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os
import shutil
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np

