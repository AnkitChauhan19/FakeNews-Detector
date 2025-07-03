import pandas as pd
import re
import spacy
import string
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

tqdm.pandas()

# Loading true and fake news datasets
true_df = pd.read_csv('Data/True.csv')
fake_df = pd.read_csv('Data/Fake.csv')

# Adding the true/fake label to the datasets
true_df['label'] = 1
fake_df['label'] = 0

# Combining the two datasets and dropping rows with null values
combined_df = pd.concat([true_df, fake_df], ignore_index=True)
combined_df.dropna(inplace=True)

# Combining title and text columns and dropping irrelevant columns
combined_df['news_text'] = combined_df['title'] + ' ' + combined_df['text']
combined_df.drop(columns=['title', 'text', 'date'], inplace=True)

# Creating a new dataframe to store cleaned data
df_clean = pd.DataFrame()

df_clean['label'] = combined_df['label']

# Function to clean the news text
def clean_text(text):
    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text

df_clean['text'] = combined_df['news_text'].progress_apply(clean_text)

# Function for POS tagging and lemmatizing text
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        if token.pos_ in ['NOUN'] and not token.is_stop: # Filtering stopwords and keeping nouns
            lemmatized_tokens.append(token.lemma_) # Storing lemmatized text

    return " ".join(lemmatized_tokens)

df_clean['lemm_text'] = df_clean['text'].progress_apply(lemmatize_text)

# Saving cleaned data frame
df_clean.to_csv('clean_data.csv', index=False)

print(df_clean.shape)