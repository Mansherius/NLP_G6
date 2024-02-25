'''
This python file is used to convert the dataset into a format that is the same as all of our other outputs so that the accuracy model has an easy time during comparison
'''

import pandas as pd
import numpy as np
from ast import literal_eval

# Open the NER_dataset
# The data is of the following format
# 1. Sentence_ID
# 2. Word (a string of an array that has all the tokenised words)
# 3. POS
# 4. Tag

# Open the NER_dataset using pandas

data = pd.read_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset.csv')
# Shape of the data -> (47959, 4)

# Now we need the data to be outputted as 
# 1. Sentence_ID
# 2. Word (Singluar tokenised word)
# 3. POS
# 4. Tag

# Creating a function to process the data from the dataset into a meangingful format
def processSentence(df, row):
    # Convert the string of an array to an actual array
    sentence = np.array(literal_eval(df['Word'][row]))
    # We will keep this as an array so that we can go through it one by one in the main loop
    return sentence

def processPOS(df, row):
    # Convert the string of an array to an actual array
    pos = np.array(literal_eval(df['POS'][row]))
    # We will keep this as an array so that we can go through it one by one in the main loop
    return pos

def processTag(df, row):
    # Convert the string of an array to an actual array
    tag = np.array(literal_eval(df['Tag'][row]))
    # We will keep this as an array so that we can go through it one by one in the main loop
    return tag

# Now we have to process the entire dataset
# We will create a new dataframe to store the results

newDF = pd.DataFrame(columns=['Sentence_ID', 'Word', 'POS', 'Tag'])

for i in range(data.shape[0]):
    sentence = processSentence(data, i)
    # No need for stopword removal as the POS tagger will take care of that
    # stop_words = set(stopwords.words('english'))
    # Ensuring that the words are not present in the stopword list
    words = [w for w in sentence]

    # Changing the format of the rest of the tags

    pos = processPOS(data, i)
    tag = processTag(data, i)

    poss = [p for p in pos]
    tagg = [t for t in tag]

    # Now we create a new row that will be appended to the dataframe and will contain the POS and entity tags for each word in the sentence
    # We already have all the tags in the NER dataset but they are in an array format and we need to separate all of them

    rows = [{'Sentence_ID': data['Sentence_ID'][i], 'Word': word, 'POS': pos, 'Tag': tag} for word, pos, tag in zip(words, poss, tagg)]

    # Now we append the rows to the dataframe
    newDF = pd.concat([newDF, pd.DataFrame(rows)], ignore_index=True)

    # check to see the program is still running as it has a long execution time
    print("running row ", i)

# Now we have to output the results to a new csv file

newDF.to_csv('/Users/manshersingh/Documents/Ashoka Coursework/NLP - 6th Sem/NER_Dataset_formatted.csv', index=False)