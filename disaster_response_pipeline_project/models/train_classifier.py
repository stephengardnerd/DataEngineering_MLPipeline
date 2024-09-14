import subprocess
import os

# Install necessary packages
os.system('pip install pandas sqlalchemy nltk scikit-learn')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install necessary packages
try:
    import pandas
except ImportError:
    install('pandas')

try:
    import sqlalchemy
except ImportError:
    install('sqlalchemy')

try:
    import nltk
except ImportError:
    install('nltk')

try:
    import sklearn
except ImportError:
    install('scikit-learn')

import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

nltk.download('omw-1.4')
nltk.download(['punkt_tab', 'wordnet'])

# 1. Load the Data
def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Args:
    database_filepath: str. Filepath for the SQLite database containing the data.

    Returns:
    X: pandas DataFrame. Feature data (messages).
    Y: pandas DataFrame. Target data (categories).
    category_names: Index. Names of the target categories.
    """
    print(f"Attempting to load data from database at: {database_filepath}")
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    print(f"Data successfully loaded from {database_filepath}")
    return X, Y, category_names

# 2. Tokenization function to process the text
def tokenize(text):
    """
    Tokenize and lemmatize text data.

    Args:
    text: str. The text to be tokenized and lemmatized.

    Returns:
    clean_tokens: list. A list of the cleaned tokens.
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

    return clean_tokens

# 3. Build the machine learning pipeline
def build_pipeline():
    """
    Build a machine learning pipeline.

    Returns:
    pipeline: sklearn Pipeline. A machine learning pipeline object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

# Define and perform grid search
def perform_grid_search(pipeline, X_train, Y_train):
    """
    Perform grid search to optimize model parameters.

    Args:
    pipeline: sklearn Pipeline. The machine learning pipeline to be optimized.
    X_train: pandas DataFrame. Training data for features.
    Y_train: pandas DataFrame. Training data for target labels.

    Returns:
    cv: sklearn GridSearchCV object. The optimized model after grid search.
    """
    parameters = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [2, 4],
        'vect__max_df': [0.75],
        'tfidf__use_idf': [True]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)

    print("Starting grid search...")
    cv.fit(X_train, Y_train)
    print("Grid search complete.")
    
    return cv

# 4. Main execution
if __name__ == "__main__":
    # Ask the user for the database file location
    database_filepath = input("Please enter the file path of the DisasterResponse.db file: ")
    print(f"Loading data...\n    DATABASE: {database_filepath}")
    X, Y, category_names = load_data(database_filepath)
    
    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build the pipeline
    pipeline = build_pipeline()
    
    # Perform grid search
    cv = perform_grid_search(pipeline, X_train, Y_train)
    
    # Evaluate the model
    print("Evaluating the model...")
    Y_pred = cv.predict(X_test)
    for i, category in enumerate(category_names):
        print(f"Category: {category}\n")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("-" * 60)
    print("Model evaluation complete.")

    # Ask the user for the pickle file save location
    model_filepath = input("Please enter the file path to save the trained model pickle file: ")
    with open(model_filepath, 'wb') as file:
        pickle.dump(cv, file)
    print(f"Trained model saved to {model_filepath}.")
