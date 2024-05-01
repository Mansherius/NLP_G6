from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm

# Load the model
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

# Load the dataset
# df = pd.read_csv("Eminem_Lyrics.csv", sep='\t', comment='#', encoding="ISO-8859-1")
df = pd.read_csv("SpotifyLyricsCleaned.csv", sep='\t', comment='#', encoding="ISO-8859-1")

# Now we need to go through each song's lyrics one by one and use the model to annotate them with the corresponding emotion
emotion = pipeline('sentiment-analysis', 
                    model='arpanghoshal/EmoRoBERTa')

# truncate the lyrics to 512 tokens so that when we call the model it doesn't run out of memory
def truncate(text):
    return text[:512]

# Annotate the lyrics with the corresponding emotion
df['text'] = df['text'].apply(truncate)

# We will save the annotated lyrics in a new column in the dataframe
tqdm.pandas(desc="Annotating Lyrics")
df['Emotion'] = df['text'].progress_apply(emotion)
df['Emotion'] = df['Emotion'].apply(lambda x: x[0]['label'])

# Drop unnecessary columns - link
df = df.drop(columns=['link'])

# Save the annotated dataframe to a new CSV file
df.to_csv('SpotifyLyricsAnnotated.csv', sep='\t', index=False)