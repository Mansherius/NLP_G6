'''
This is our main file where we will create our predictive model using BERT and train it on the annotated datasets that we have created.
Furthermore, this model will have the following uses :
Predict the emotion of the lyrics of a song for the following:
    - Music Recommendation System:
        - Recommend songs based on the emotion of the lyrics
        - Take an input from the user that can either be a song, a mood/emotion, or a a combination of both
        - Recommend songs based on the input by running our model on the input, classifying the emotion of the input and going through our database to suggest songs with similar emotions
        - We can also use topic modelling to recommend songs based on the topic of the lyrics and use that on the subset of the songs that have the same emotion as the input

    - Playlist generator:
        - Take in an input of a file that contains all their songs in a specific csv format
        - Run our model on the lyrics of each song and classify the emotion of the lyrics
        - Club the songs into various groups based on the emotion of the lyrics
        - Create playlists based on the emotions of the lyrics and add some songs from our database that have the same emotion as the songs in the input file (equivalent of smart shuffle)        
'''

# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, AdamW
import tensorflow as tf
from tqdm import tqdm  # Import tqdm for progress bar
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
df = pd.read_csv("SpotifyLyricsAnnotated.csv", sep='\t', comment='#', encoding="ISO-8859-1")
df = df.head(1200)  # Use 8000 rows for training

# Define a function to split lyrics into chunks
def split_lyrics_into_chunks(lyrics, max_chunk_length):
    chunks = []
    words = lyrics.split()
    current_chunk = []
    current_length = 0
    for word in words:
        word_length = len(tokenizer.tokenize(word))
        if current_length + word_length <= max_chunk_length:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Tokenization and chunking
max_chunk_length = 128
tokenized_lyrics = []
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
for index, row in df.iterrows():
    lyric_chunks = split_lyrics_into_chunks(row['text'], max_chunk_length)
    for chunk in lyric_chunks:
        tokenized_chunk = tokenizer(chunk, truncation=True, padding='max_length', max_length=max_chunk_length, return_tensors='pt')
        tokenized_lyrics.append({
            'input_ids': tokenized_chunk['input_ids'],
            'attention_mask': tokenized_chunk['attention_mask'],
            'emotion': row['Emotion']  # Include emotion label for each chunk
        })

# Concatenate tokenized chunks
input_ids = torch.cat([chunk['input_ids'] for chunk in tokenized_lyrics], dim=0)
attention_masks = torch.cat([chunk['attention_mask'] for chunk in tokenized_lyrics], dim=0)
emotions = [chunk['emotion'] for chunk in tokenized_lyrics]

# Convert emotion labels to numerical format
emotion_to_index = {emotion: index for index, emotion in enumerate(set(emotions))}
numerical_emotions = [emotion_to_index[emotion] for emotion in emotions]

# Convert numerical emotions to their emotions  
index_to_emotion = {index: emotion for emotion, index in emotion_to_index.items()}

# Store the mappings in a .py file called "mappings.py"
# If mapping.py already exists, it will be overwritten
with open('mappings.py', 'w') as f:
    f.write(f'emotion_to_index = {emotion_to_index}\n')
    f.write(f'index_to_emotion = {index_to_emotion}')



# Convert numerical emotions to tensor
emotion_tensor = torch.tensor(numerical_emotions)

# Create TensorDataset
dataset = TensorDataset(input_ids, attention_masks, emotion_tensor)

# Define sentiment analysis model
class SentimentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Access the pooled output of RoBERTa
        output = self.drop(pooled_output)
        return self.out(output)


# Initialize model
numClasses = len(df['Emotion'].unique())  # Number of unique emotion classes
model = SentimentClassifier(num_classes=numClasses)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Split dataset into train and validation sets
train_size = 0.8
train_indices, val_indices = train_test_split(list(range(len(dataset))), train_size=train_size, shuffle=True)

# Define train and validation samplers
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

# Define dataloaders
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_preds = 0
    total_preds = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    for batch in progress_bar:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct_preds += torch.sum(predicted == labels).item()
        total_preds += len(labels)
        progress_bar.set_postfix({'loss': train_loss / len(progress_bar), 'accuracy': correct_preds / total_preds})

    # Evaluation
    model.eval()
    val_loss = 0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct_preds += torch.sum(predicted == labels).item()
            total_preds += len(labels)

    val_loss /= len(val_loader)
    accuracy = correct_preds / total_preds

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}, Accuracy: {accuracy}')

# At this point the model is trained and ready to be used for inference
# However, since we intend on using the trained model in another script

# Since we used torch, we will have to save the model and the tokenizer

# Save the model
torch.save(model.state_dict(), 'MusicClassifier.pth')

# Save the tokenizer
tokenizer.save_pretrained('tokenizer')