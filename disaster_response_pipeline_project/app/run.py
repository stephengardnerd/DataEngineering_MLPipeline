import subprocess
import sys

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        sys.exit(1)

# List of required packages
required_packages = [
    'pandas',
    'plotly',
    'joblib',
    'nltk',
    'flask',
    'sqlalchemy',
    'scikit-learn'
]
# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import json
import plotly
import pandas as pd
from joblib import load
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def get_file_locations():
    # Ask the user for the database file location
    db_file = input("Please enter the database file location (e.g., ../data/YourDatabaseName.db): ")
    # Ask the user for the pickle file location
    model_file = input("Please enter the pickle file location (e.g., ../models/your_model_name.pkl): ")
    return db_file, model_file

# Get the file locations from the user
db_file, model_file = get_file_locations()

# Load data
engine = create_engine(f'sqlite:///{db_file}')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
model = load(model_file)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    
    # First visualization: Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Second visualization: Distribution of Message Categories
    category_names = df.columns[4:]  # Assuming the first 4 columns are not categories
    category_counts = df[category_names].sum().sort_values(ascending=False)
    
    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=category_counts.index,
                    values=category_counts.values
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories'
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
