# Import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Ensure necessary NLTK data is downloaded
nltk.download(['punkt', 'wordnet'])

# 1. Load the Data
def load_data(database_filepath='sqlite:///DisasterResponse.db'):
    # Create a connection to the SQLite database
    engine = create_engine(database_filepath)
    
    # Read the data from the SQL table into a DataFrame
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Define the feature and target variables
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    
    return X, Y, category_names

# 2. Tokenization function to process the text
def tokenize(text):
    """
    Tokenizes and processes the input text.

    Args:
    text (str): The input text to tokenize.

    Returns:
    list: A list of clean tokens (words).
    """
    # Replace URLs with a placeholder string
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each token, normalize to lowercase, and remove whitespace
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

    return clean_tokens

# 3. Build the machine learning pipeline
def build_pipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline

# 4. Main execution
if __name__ == "__main__":
    # Load data
    X, Y, category_names = load_data('sqlite:///DisasterResponse.db')
    
    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Build the pipeline
    pipeline = build_pipeline()
    
    # Train the model
    print("Training the model...")
    pipeline.fit(X_train, Y_train)
    print("Model training complete.")
    
    # Save the trained model as a pickle file
    model_filepath = 'classifier.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Model saved to {model_filepath}.")
