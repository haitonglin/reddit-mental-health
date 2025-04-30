
import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from datasets import load_dataset

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(post):
    post = re.sub(r'\n', ' ', post)
    post = re.sub(r'[^\x00-\x7f]', '', post)
    post = post.lower().split()
    post = [word for word in post if word not in stop_words]
    post = [word for word in post if not word.startswith('http') and not word.startswith('@')]
    post = [word.translate(str.maketrans('', '', string.punctuation)) for word in post]
    post = [word for word in post if word]
    return post

def load_data(path):
    ds = load_dataset(path)
    df = ds['train'].to_pandas()
    df = df.dropna(subset=['body'])
    df = df[~df['body'].isin(['[removed]', '[deleted]'])].reset_index(drop=True)
    return df

# backup
def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
