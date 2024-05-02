'''
This is going to act as our driver program. we will call the model that we have trained in classifier.py and use it here
We will take in an input from the user that can either be a song, a mood/emotion, or a a combination of both
We will then run our model on the input, classify the emotion of the input and go through our database to suggest songs with similar emotions
We can also use topic modelling to recommend songs based on the topic of the lyrics and use that on the subset of the songs that have the same emotion as the input
We will also use the model to classify the emotion of the lyrics of the input song and then recommend songs with the same emotion from our database (call it smart shuffle)
'''

'''
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
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import mappings # Import the mappings file
import tqdm as tqdm

# For us to be able to use the model defined in classifier.py, we need to import it here and define it's structure
# Define the model architecture
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

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('tokenizer')

# Define your model architecture again
model = SentimentClassifier(num_classes=27)

# Load the saved parameters into the model
model.load_state_dict(torch.load('MusicClassifier.pth'))

# Set the model to evaluation mode
model.eval()

# Define a function to predict the emotion of the lyrics
def predict_emotion(text):
    # Tokenize the input text
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    # Make the prediction
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    # Get the predicted label
    label_index = torch.argmax(output, dim=1).item()
    
    # Convert numerical label to emotion
    predicted_emotion = mappings.index_to_emotion[label_index]
    
    # Return the predicted emotion
    return predicted_emotion



# Function to recommend songs based on the emotion of the lyrics
def music_recommendation():
    # Load the dataset
    df = pd.read_csv("SpotifyLyricsAnnotated.csv", sep='\t', comment='#', encoding="ISO-8859-1")

    # Take the input from the user
    print("Some important information before you proceed:")
    print("1. You can enter mood/emotion, or some song lyrics that capture the kind of emotion you are feeling.")
    print("2. The model will classify the emotion of the input and recommend songs with lyrics of the same emotion.")
    user_input = input("Enter the song name, mood/emotion, or a combination of both: ")
    user_input = user_input.lower()

    # Classify the emotion of the user input
    user_input_emotion = predict_emotion(user_input)
    print("The emotion of the input is: ", user_input_emotion)

    # Get the songs with the same emotion as the user input
    songs = df[df['Emotion'] == user_input_emotion]

    print("Here are some artists whose lyrics you might like, select one or more to get recommendations:")
    # Print a bullet point list of the artists
    for artist in songs['artist'].unique():
        print(f"• {artist}")

    # Take the input from the user
    while True:
        artist = input("Enter the artist name: ")
        if artist in songs['artist'].unique():
            # Get the songs of the artist
            artist_songs = songs[songs['artist'] == artist]
            print("Here are some songs by the artist with the same emotion:")
            for song in artist_songs['song'].unique():
                print(f"• {song}")
            break
        else:
            print("Invalid artist name! Please try again.")

# Function to generate playlists based on the emotion of the lyrics
def playlist_generator():
    # Load the dataset
    df = pd.read_csv("SpotifyLyricsAnnotated.csv", sep='\t', comment='#', encoding="ISO-8859-1")

    # Take the input from the user
    print("Some important information before you proceed:")
    print("1. You need to provide a file that contains all your songs in a specific csv format.")
    print("2. The model will classify the emotion of the lyrics of each song and create playlists based on the emotions of the lyrics.")
    file_path = input("Enter the path of the file: (Be accurate with the path, it should be in the format 'path/to/file.csv')")

    # Load the user's songs
    user_songs = pd.read_csv(file_path, sep='\t', comment='#', encoding="ISO-8859-1")

    # Classify the emotion of the lyrics of each song
    # Add a progress bar using tqdm
    user_songs['Emotion'] = ''
    for index, row in tqdm.tqdm(user_songs.iterrows(), total=len(user_songs)):
        user_songs.at[index, 'Emotion'] = predict_emotion(row['Lyrics'])

    # Club the songs into various groups based on the emotion of the lyrics
    playlists = user_songs.groupby('Emotion')

    # Create playlists based on the emotions of the lyrics
    print("Here are the playlists based on the emotions of the lyrics:")
    for emotion, songs in playlists:
        # Encode the emotion string to UTF-8
        emotion_utf8 = emotion.encode('utf-8', 'ignore').decode('utf-8')
        print(f"\nPlaylist for {emotion_utf8}:")
        for song in songs['Song_Name'].unique():
            # Encode the song name string to UTF-8
            song_utf8 = song.encode('utf-8', 'ignore').decode('utf-8')
            print(f"• {song_utf8}")


# Main function
def main():
    # Take in the user inputs after a basic welcome screen
    print("Welcome to our Final NLP Project!")
    print("We have two functionalities that you can use:")
    print("1. Music Recommendation System")
    print("2. Playlist Generator")

    while True:
        # Ask the user to choose between the two functionalities
        choice = int(input("Enter 1 for Music Recommendation System and 2 for Playlist Generator: "))
        if choice == 1:
            music_recommendation()
        elif choice == 2:
            playlist_generator()
        else:
            print("Invalid choice! Please try again.")

        # Ask the user if they want to continue using the program
        cont = input("Do you want to continue using the program? (y/n): ")
        if cont.lower() != 'y':
            break

    print("Thank you for using our program! Have a great day!")

# Call the main function

if __name__ == "__main__":
    main()