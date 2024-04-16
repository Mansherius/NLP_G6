import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf

# Load data
df = pd.read_csv('your_data.csv')

# Split text data into individual posts
df['posts'] = df['posts'].apply(lambda x: [post.strip() for post in x.split('|||')])

# Prepare labels
labels = df['MBTI_classification'].values

# Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text and pad sequences
max_length = 128  # You can adjust this based on your requirements
input_ids = []
attention_masks = []

for posts in df['posts']:
    encoded_dict = tokenizer.encode_plus(
        posts,                      # Text to encode
        add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
        max_length=max_length,      # Pad & truncate all sentences
        padding='max_length',
        truncation=True,
        return_attention_mask=True, # Construct attention masks
        return_tensors='tf'         # Return TensorFlow tensors
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)

# Split data into train, validation, and test sets
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42)

