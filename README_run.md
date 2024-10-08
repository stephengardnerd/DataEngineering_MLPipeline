# Disaster Response Pipeline: Run.py Script

## Overview

This project is part of a data science pipeline designed to analyze and classify disaster-related messages. The aim is to categorize these messages so that they can be effectively routed to the appropriate disaster response agencies. The pipeline processes text data, trains a machine learning model, and uses a Flask web app to display visualizations of the data.

## Project Components

1. **Data Processing (`process_data.py`)**:
   - Merges and cleans the disaster response messages and categories datasets.
   - Saves the cleaned data into an SQLite database.

2. **Model Training (`train_classifier.py`)**:
   - Loads the cleaned data from the SQLite database.
   - Trains a machine learning model to classify the messages into multiple categories.
   - Saves the trained model as a pickle file.

3. **Web App (`run.py`)**:
   - Loads the trained model and cleaned data.
   - Runs a Flask web app that visualizes the distribution of message genres and categories.
   - Allows users to input a message and see the predicted categories.

## Installation

### Prerequisites

Ensure you have Python installed on your machine. You will also need `pip` to install the necessary Python packages.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/disaster_response_pipeline.git
cd disaster_response_pipeline
```

### 2. Set Up the Environment

You can use Conda to create a virtual environment:

```bash
conda create --name disaster_env python=3.9
conda activate disaster_env
```

### 3. Install Dependencies

Before running the scripts, install the necessary dependencies:

```bash
pip install pandas sqlalchemy nltk scikit-learn
```

Alternatively, you can use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Script

### 1. Running the Web App (`run.py`)

This script runs a Flask web app that allows users to classify new messages and visualize data.

#### Run the Script:

```bash
python run.py
```

### 2. Provide File Paths

The script will prompt you to enter the file paths for the database and the pickle file:

- **For the database file path,** enter: `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data/disaster_Response.db`
- **For the model file save location,** enter: `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/models/classifier.pkl`

### 3. Example Input Message

Here is an example of a disaster recovery message you can input to test the classifier:

```plaintext
"There is a significant need for clean drinking water in the flood-affected areas. Please send water purification tablets and bottled water as soon as possible."
```

### 4. Visualizations

Once the model is trained and the app is running, the following visualizations will be available:

#### Screenshot #1: 
**Distribution of Message Genres**
![Screenshot #1](path/to/your/screenshot1.png)

#### Screenshot #2:
**Distribution of Message Categories**
![Screenshot #2](path/to/your/screenshot2.png)

## Contribution

Feel free to fork this repository, make improvements, and submit pull requests.

