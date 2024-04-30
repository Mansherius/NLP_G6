from transformers import pipeline
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the model

# Load the dataset
data = pd.read_csv(r'.\Eminem_Lyrics.csv', sep='\t', comment='#', encoding="ISO-8859-1")
#df = pd.read_csv("spotify_millsongdata.csv")
df = data.head(20)

def clean_text(text, max_length=578):
    cleaned_text = ' '.join(text.split())
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.translate(str.maketrans("", "", string.punctuation))
    cleaned_text = ''.join([i for i in cleaned_text if not i.isdigit()])
    cleaned_text = ''.join([i for i in cleaned_text if i.isalpha() or i.isspace()])
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(cleaned_text)
    cleaned_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
    
    padded_text = cleaned_text.ljust(max_length)
    
    return padded_text

# Clean the text
df['text'] = df['Lyrics'].apply(clean_text)

# Now we have the dataset that we want to annotate

