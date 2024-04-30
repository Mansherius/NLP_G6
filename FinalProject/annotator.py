from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import pandas as pd

# Load the model
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

# Load the dataset
# df = pd.read_csv("Eminem_Lyrics.csv", sep='\t', comment='#', encoding="ISO-8859-1")
df = pd.read_csv(".\EminemLyricsCleaned.csv", sep='\t', comment='#', encoding="ISO-8859-1")

# Now we need to go through each song's lyrics one by one and use the model to annotate them with the corresponding emotion
emotion = pipeline('sentiment-analysis', 
                    model='arpanghoshal/EmoRoBERTa')

# We will save the annotated lyrics in a new column in the dataframe
df['Emotion'] = df['Lyrics'].apply(emotion)
df['Emotion'] = df['Emotion'].apply(lambda x: x[0]['label'])

# Drop unnecessary columns - link
df = df.drop(columns=['Album_URL'])
df = df.drop(columns=['Views'])
df = df.drop(columns=['Release_date'])

# Save the annotated dataframe to a new CSV file
df.to_csv('EminemLyricsAnnotated.csv', sep='\t', index=False)