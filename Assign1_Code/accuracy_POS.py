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

# Extract pos for comparison
pos1 = df1_common['POS'].tolist()
pos2 = df2_common['POS'].tolist()

# Combine tags from both dataframes
combined_tags = list(set(pos1).union(set(pos2)))

# Use LabelEncoder to convert tags to numeric labels
label_encoder = LabelEncoder()
label_encoder.fit(combined_tags)

pos1_encoded = label_encoder.transform(pos1)
pos2_encoded = label_encoder.transform(pos2)

# Calculate accuracy
accuracy = accuracy_score(pos1_encoded, pos2_encoded)
print(f'Tag-wise Accuracy for POS: {accuracy * 100:.2f}%')

# Generate confusion matrix
conf_matrix = confusion_matrix(pos1_encoded, pos2_encoded)
print('\nConfusion Matrix for POS:')
print(conf_matrix)

# Save confusion matrix to a file
conf_matrix_df = pd.DataFrame(conf_matrix, columns=label_encoder.classes_, index=label_encoder.classes_)
conf_matrix_df.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/POS_comparison_nltk/confusion_matrix_POS_nltk.csv')

# Find and save mismatched labels to a file
mismatched_labels = [(label_encoder.classes_[t1], label_encoder.classes_[t2]) for t1, t2 in zip(pos1_encoded, pos2_encoded) if t1 != t2]
mismatched_labels_df = pd.DataFrame(mismatched_labels, columns=['True_Label', 'Predicted_Label'])
mismatched_labels_df.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/POS_comparison_nltk/mismatched_labels_POS_nltk.csv')
