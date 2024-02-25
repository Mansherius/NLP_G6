import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# The file paths here are variable and can be changed depending on what files we are comparing
file1_path = '/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset_formatted.csv'
file2_path = '/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset_Processed_nltk.csv'

# Read CSV files into pandas dataframes
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Calculate the number of rows for each Sentence_ID in both dataframes
rows_count_df1 = df1.groupby('Sentence_ID').size().reset_index(name='count_df1')
rows_count_df2 = df2.groupby('Sentence_ID').size().reset_index(name='count_df2')

# Merge dataframes based on Sentence_ID
merged_df = pd.merge(rows_count_df1, rows_count_df2, on='Sentence_ID', how='inner')

# Filter dataframes to include only common Sentence_IDs with the same number of rows
common_sentence_ids = merged_df[merged_df['count_df1'] == merged_df['count_df2']]['Sentence_ID']

df1_common = df1[df1['Sentence_ID'].isin(common_sentence_ids)]
df2_common = df2[df2['Sentence_ID'].isin(common_sentence_ids)]

# Now we have a dictionary of the tags
# This is because the names of the tags in spaCy and the names of the tags in NLTK/NER_Dataset_formatted.csv are different and we need to map them to each other
# This will ensure that we do not disregard correctly tagged words

# Dictionary with the conversions of the tags
ner_to_Spacy = {
    "B-geo": "GPE",
    "I-geo": "GPE",
    "B-gpe": "GPE",
    "I-gpe": "GPE",
    "B-tim": "DATE",
    "I-tim": "DATE",
    "B-org": "ORG",
    "I-org": "ORG",
    "B-per": "PERSON",
    "I-per": "PERSON",
    "B-art": "WORK_OF_ART",
    "I-art": "WORK_OF_ART",
}

ner_to_nltk = {
    "B-geo": ["LOCATION"],
    "I-geo": ["LOCATION"],
    "B-gpe": ["GPE"],
    "I-gpe": ["GPE"],
    "B-tim": ["B_DATE"],
    "I-tim": ["I_DATE"],
    "B-org": ["ORGANIZATION"],
    "I-org": ["ORGANIZATION"],
    "B-per": ["PERSON"],
    "I-per": ["PERSON"],
    "O": ["O"]
}

# Use the replace method to update tags in df1_common
df1_common['Tag'].replace(ner_to_nltk, inplace=True) # Change this depending on whether you are running the code for nltk or spacy

# Extract tags for comparison
tag1 = df1_common['Tag'].tolist()
tag2 = df2_common['Tag'].tolist()

# Combine tags from both dataframes
combined_tags = list(set(tag1).union(set(tag2)))
        
# Use LabelEncoder to convert tags to numeric labels
label_encoder = LabelEncoder()
label_encoder.fit(combined_tags)

tag1_encoded = label_encoder.transform(tag1)
tag2_encoded = label_encoder.transform(tag2)

# Calculate accuracy
accuracy = accuracy_score(tag1_encoded, tag2_encoded)
print(f'Tag-wise Accuracy for NER: {accuracy * 100:.2f}%')

# Show the mismatched tags
mismatched_tags = [(label_encoder.classes_[t1], label_encoder.classes_[t2]) for t1, t2 in zip(tag1_encoded, tag2_encoded) if t1 != t2]
mismathced_labels_df = pd.DataFrame(mismatched_tags, columns=['True_Label', 'Predicted_Label'])
mismathced_labels_df.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/tag_comparison_nltk/mismatched_labels_NER_nltk.csv')

# Generate confusion matrix
conf_matrix = confusion_matrix(tag1_encoded, tag2_encoded)

# Save confusion matrix to a file
conf_matrix_df = pd.DataFrame(conf_matrix, columns=label_encoder.classes_, index=label_encoder.classes_)
conf_matrix_df.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/tag_comparison_nltk/confusion_matrix_NER_nltk.csv')
