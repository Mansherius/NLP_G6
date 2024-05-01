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
from transformers import pipeline
from tqdm import tqdm  # Import tqdm for progress bar
from classifier import model, tokenizer

# Function to recommend songs based on the emotion of the lyrics
def music_recommendation():
    # Load the dataset
    df = pd.read_csv("SpotifyLyricsAnnotated.csv", sep='\t', comment='#', encoding="ISO-8859-1")

    # Load the emotion pipeline

    # Find out how this works to see if ['label] is the correct way to get the emotion
    emotion = pipeline('sentiment-analysis', 
                        model=model)

    # Take the input from the user
    print("Some important information before you proceed:")
    print("1. You can enter mood/emotion, or some song lyrics that capture the kind of emotion you are feeling.")
    print("2. The model will classify the emotion of the input and recommend songs with lyrics of the same emotion.")
    user_input = input("Enter the song name, mood/emotion, or a combination of both: ")
    user_input = user_input.lower()

    # Classify the emotion of the user input
    user_input_emotion = emotion(user_input)[0]['label']

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

    # Load the emotion pipeline
    emotion = pipeline('sentiment-analysis', 
                        model=model)

    # Take the input from the user
    print("Some important information before you proceed:")
    print("1. You need to provide a file that contains all your songs in a specific csv format.")
    print("2. The model will classify the emotion of the lyrics of each song and create playlists based on the emotions of the lyrics.")
    file_path = input("Enter the path of the file: (Be accurate with the path, it should be in the format 'path/to/file.csv')")

    # Convert the file to a CSV if required
    if file_path.endswith('.xlsx'):
        user_songs = pd.read_excel(file_path)
        file_path = file_path.replace('.xlsx', '.csv')
        user_songs.to_csv(file_path, sep='\t', index=False)
    
    # Load the user's songs
    user_songs = pd.read_csv(file_path, sep='\t', comment='#', encoding="ISO-8859-1")

    # Classify the emotion of the lyrics of each song
    user_songs['Emotion'] = user_songs['text'].apply(lambda x: emotion(x)[0]['label'])

    # Club the songs into various groups based on the emotion of the lyrics
    playlists = user_songs.groupby('Emotion')

    # Create playlists based on the emotions of the lyrics
    print("Here are the playlists based on the emotions of the lyrics:")
    for emotion, songs in playlists:
        print(f"\nPlaylist for {emotion}:")
        for song in songs['song'].unique():
            print(f"• {song}")


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