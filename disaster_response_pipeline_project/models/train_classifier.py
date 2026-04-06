

import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from utils import tokenize

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

# 2. Build the machine learning pipeline
def build_pipeline():
    """
    Build a machine learning pipeline.

    Returns:
    pipeline: sklearn Pipeline. A machine learning pipeline object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # tokenize imported from utils
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
    if len(sys.argv) == 3:
        database_filepath = sys.argv[1]
        model_filepath = sys.argv[2]
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

        # Save the model
        with open(model_filepath, 'wb') as file:
            pickle.dump(cv, file)
        print(f"Trained model saved to {model_filepath}.")
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
