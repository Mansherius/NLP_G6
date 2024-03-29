import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# We have to process the NER_dataset with this NLP model
# The NER_dataset is in the following path: NLP - 6th Sem/NER_dataset.csv

# The NER_dataset is a csv file with the following columns:
# 1. Sentence
# 2. Word
# 3. POS
# 4. Tag

# We have to process the NER_dataset with the NLP model using POS_tagger and NER_tagger for each sentence in the dataset

import pandas as pd
import numpy as np
from ast import literal_eval

# Open the NER_dataset
# The data is of the following format
# It is a string of an array
# The elements of the array are all the individiual words in the sentence that are separated by commas
# As the NLP model processes sentences, we have to first convert the string of an array to an actual array 
# Then use join to add all the individual words in the array to form a sentence and convert that to a string
# Then we need to process each sentence using the NLP model

# Open the NER_dataset using pandas
data = pd.read_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset.csv')
# Shape of the data -> (47959, 4)


# print(data.head())

# Creating a function to process the data from the dataset into a meangingful format
def processSentence(df, row):
    # Convert the string of an array to an actual array
    sentence = np.array(literal_eval(df['Word'][row]))
    sentence = ' '.join(sentence)
    return sentence

'''
# Now we Process the data. For now let us only work on the head of the dataframe
for i in range(1):
    sentence = processSentence(data, i)
    print(sentence)
    doc = nlp(sentence)
    
    # If there is no NER tag, then the tag is set to 'O'
    # If there is a NER tag, then the tag is set to the NER tag
    for token in doc:
        txt = token.text
        pos = token.tag_
        ner = token.ent_type_
        if ner == '':
            ner = 'O'

        print(txt, pos, ner)
'''
# Now that we know this works, we have to process the entire dataset as well as output the results to a new csv file in a concise format
# The dataframe will have the following columns:
# 1. Sentence_ID
# 2. Word
# 3. POS
# 4. Tag

# The Sentence_ID will be the index of the dataframe

newDF = pd.DataFrame(columns=['Sentence_ID', 'Word', 'POS', 'Tag'])

for i in range(data.shape[0]):
    sentence = processSentence(data, i)
    doc = nlp(sentence)
    
    rows = [{'Sentence_ID': data['Sentence_ID'][i], 'Word': token.text, 'POS': token.tag_, 'Tag': token.ent_type_ if token.ent_type_ else 'O'} for token in doc]

    # Concatenating the rows to the newDF
    newDF = pd.concat([newDF, pd.DataFrame(rows)], ignore_index=True)
    # Check to ensure the programming is still running as it has a long execution time
    print("running row ", i)

# Now we have to output the newDF to a new csv file
newDF.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset_Processed_spaCy.csv', index=False)

# Now the code to process the personal dataset that we have curated
'''
text_file_path = '/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/nlp_dataset.txt'

# Read the text file and create a DataFrame
with open(text_file_path, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

df = pd.DataFrame({'Sentence': sentences})

customDF = pd.DataFrame(columns=['Sentence_No', 'Word', 'POS', 'Tag'])

for i in range(df.shape[0]):
    sentence = df['Sentence'][i]
    doc = nlp(sentence)
    
    rows = [{'Sentence_No': i, 'Word': token.text, 'POS': token.tag_, 'Tag': token.ent_type_ if token.ent_type_ else 'O'} for token in doc]

    # Concatenating the rows to the newDF
    customDF = pd.concat([customDF, pd.DataFrame(rows)], ignore_index=True)
    # Check to ensure the programming is still running as it has a long execution time
    print("running row ", i)

# Now we have to output the newDF to a new csv file
customDF.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/Custom_Dataset_Processed_spaCy.csv', index=False)
'''