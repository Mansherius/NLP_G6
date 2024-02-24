import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from ast import literal_eval

# Open the NER_dataset
data = pd.read_csv("/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset.csv")

'''
# Creating a function to process the data from the dataset into a meaningful format
def processSentence(df, row):
    # Convert the string of an array to an actual array
    sentence = np.array(literal_eval(df['Word'][row]))
    # This array is already tokenized and does not need to be tokenized again
    return sentence

# Now we have to process the entire dataset
# We will create a new dataframe to store the results

newDF = pd.DataFrame(columns=['Sentence_ID', 'Word', 'POS', 'Tag'])

# Now we process the entire dataset
for i in range(data.shape[0]):
    sentence = processSentence(data, i)
    # No need for stopword removal as the POS tagger will take care of that
    # stop_words = set(stopwords.words('english'))
    # Ensuring that the words are not present in the stopword list
    words = [w for w in sentence]

    # Now we create a new row that will be appended to the dataframe and will contain the POS and entity tags for each word in the sentence
    # We will use the nltk POS tagger for this

    rows = [{'Sentence_ID': data['Sentence_ID'][i], 'Word': word, 'POS': pos, 'Tag': 'O'} for word, pos in nltk.pos_tag(words)]

    # Now we have to change the value of the 'Tag' column to the NER tag if it is present
    # We will use the nltk NER tagger for this

    ner_tags = nltk.ne_chunk(nltk.pos_tag(words))

    for j, chunk in enumerate(ner_tags):
        if isinstance(chunk, nltk.Tree):
            # If the current token is part of a named entity, set the 'Tag' column accordingly
            for k in range(len(chunk.leaves())):
                rows[j + k]['Tag'] = chunk.label()

    # Now we append the rows to the dataframe
    newDF = pd.concat([newDF, pd.DataFrame(rows)], ignore_index=True)

    # check to see the program is still running as it has a long execution time
    print("running row ", i)

# Now we have to output the results to a new csv file
newDF.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset_Processed_nltk.csv', index=False)
'''

# Now the code to process the personal dataset that we have curated

text_file_path = '/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/nlp_dataset.txt'
# Read the text file and create a DataFrame
with open(text_file_path, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

df = pd.DataFrame({'Sentence': sentences})

# Now we process the entire dataset

# Now we have sentnces that have not been tokenized

newDF = pd.DataFrame(columns=['Sentence_ID', 'Word', 'POS', 'Tag'])

# Now we process the entire dataset

for i in range(df.shape[0]):
    sentence = df['Sentence'][i]
    # No need for stopword removal as the POS tagger will take care of that
    # stop_words = set(stopwords.words('english'))
    # Ensuring that the words are not present in the stopword list
    words = word_tokenize(sentence)

    # Now we create a new row that will be appended to the dataframe and will contain the POS and entity tags for each word in the sentence
    # We will use the nltk POS tagger for this

    rows = [{'Sentence_ID': i, 'Word': word, 'POS': pos, 'Tag': 'O'} for word, pos in nltk.pos_tag(words)]

    # Now we have to change the value of the 'Tag' column to the NER tag if it is present
    # We will use the nltk NER tagger for this

    ner_tags = nltk.ne_chunk(nltk.pos_tag(words))

    for j, chunk in enumerate(ner_tags):
        if isinstance(chunk, nltk.Tree):
            # If the current token is part of a named entity, set the 'Tag' column accordingly
            for k in range(len(chunk.leaves())):
                rows[j + k]['Tag'] = chunk.label()

    # Now we append the rows to the dataframe
    newDF = pd.concat([newDF, pd.DataFrame(rows)], ignore_index=True)

    # check to see the program is still running as it has a long execution time
    print("running row ", i)

# Now we have to output the results to a new csv file
newDF.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/Custom_Dataset_Processed_nltk.csv', index=False)