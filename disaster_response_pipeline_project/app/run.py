import sys
import json
import plotly
import pandas as pd
from joblib import load
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine
sys.path.insert(0, '..')
from utils import tokenize

app = Flask(__name__)

def get_file_locations():
    """
    Prompts the user to input the file paths for the database and the model pickle file.

    Returns:
    db_file (str): The file path of the SQLite database.
    model_file (str): The file path of the model pickle file.
    """
    db_file = sys.argv[1] if len(sys.argv) > 1 else "../data/DisasterResponse.db"
    model_file = sys.argv[2] if len(sys.argv) > 2 else "../models/classifier.pkl"
    return db_file, model_file

# Get the file locations from the user
db_file, model_file = get_file_locations()

# Load data
engine = create_engine(f'sqlite:///{db_file}')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
model = load(model_file)

@app.route('/')
@app.route('/index')
def index():
    """
    Renders the main page of the web app, displaying two visualizations:
    1. Distribution of Message Genres.
    2. Distribution of Message Categories.

    Returns:
    rendered HTML template for the index page.
    """
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
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

@app.route('/go')
def go():
    """
    Handles the user query and displays the model results.

    Returns:
    rendered HTML template for the go page with the classification results.
    """
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
    """
    Main function to run the Flask app.
    """
    app.run(host='0.0.0.0', port=3001, debug=False)

if __name__ == '__main__':
    main()
