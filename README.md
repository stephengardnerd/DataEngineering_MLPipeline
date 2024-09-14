# Disaster Response Pipeline Project

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

## Usage

### 1. Process the Data (`process_data.py`)

This script processes the disaster response data and saves the cleaned data into an SQLite database.

#### Run the Script:

```bash
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

### 2. Train the Model (`train_classifier.py`)

This script trains the machine learning model and saves the trained model as a pickle file.

#### Run the Script:

```bash
python train_classifier.py
```

You will be prompted to enter the file paths for the database and the pickle file:

Example:

```plaintext
Please enter the file path of the DisasterResponse.db file: data/DisasterResponse.db
Please enter the file path to save the trained model pickle file: models/classifier.pkl
```

### 3. Run the Web App (`run.py`)

This script runs a Flask web app that allows users to classify new messages and visualize data.

#### Run the Script:

```bash
python run.py
```

### Example Input Message

To test the classifier, you can input a message like:

```plaintext
"There is a significant need for clean drinking water in the flood-affected areas. Please send water purification tablets and bottled water as soon as possible."
```

### Visualizations

Once the model is trained and the app is running, the following visualizations will be available:

#### Screenshot #1: 
**Distribution of Message Genres**
![Screenshot #1](https://github.com/stephengardnerd/DataEngineering_MLPipeline/blob/main/disaster_response_pipeline_project/DisasterRecovery%20Plot.png)

#### Screenshot #2:
**Distribution of Message Categories**
![Screenshot #2](https://github.com/stephengardnerd/DataEngineering_MLPipeline/blob/main/disaster_response_pipeline_project/DisasterRecovery%20Plot2.png)

## File Descriptions

- **`process_data.py`**: Script for cleaning and processing disaster response data.
- **`train_classifier.py`**: Script for training and saving the machine learning model.
- **`run.py`**: Flask web app script that visualizes data and allows message classification.
- **`requirements.txt`**: List of dependencies required for running the project.
- **`data/`**: Directory containing the input datasets.
- **`models/`**: Directory where the trained model will be saved.

## Acknowledgements

This project was completed as part of the Udacity Data Engineering Nanodegree program. The dataset used in this project is provided by [Figure Eight](https://www.figure-eight.com/).

## Contact

For any questions or issues, please reach out through GitHub.

```

This README file provides a comprehensive guide to setting up and running the scripts in your project. It includes installation instructions, usage examples, and details about the project's components. You can copy this content into a `README.md` file in your GitHub repository, replace placeholders with actual links and your username, and include screenshots of the visualizations as needed.
