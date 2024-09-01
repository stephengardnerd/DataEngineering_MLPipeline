# Disaster Response Pipeline

This repository contains the final project for the Udacity Data Engineering Nanodegree. The project involves building a machine learning pipeline to categorize disaster response messages into various categories. The pipeline processes text data from disaster-related messages and categorizes them to facilitate quick responses by disaster relief organizations.

## Project Overview

The Disaster Response Pipeline project is designed to help organizations classify messages received during disasters. The pipeline:
- **Processes text data** from messages and categories datasets.
- **Cleans and prepares the data** for machine learning by splitting category data into separate columns, converting them to binary values, and removing duplicates.
- **Builds a machine learning model** using a `RandomForestClassifier` wrapped in a `MultiOutputClassifier` to predict multiple categories for each message.
- **Saves the model** as a pickle file for future use.

## Components

### 1. Data Preprocessing

- **Merge Datasets**: The `disaster_messages.csv` and `disaster_categories.csv` datasets are merged on the `id` column.
- **Clean Data**: Categories are split into 36 individual binary columns, and duplicate rows are removed.
- **Save Clean Data**: The cleaned data is saved into an SQLite database (`DisasterResponse.db`).

### 2. Tokenization and Lemmatization

- **Custom Tokenizer**: A custom tokenizer is implemented to process the text data. The tokenizer:
  - Replaces URLs with a placeholder.
  - Tokenizes the text into words.
  - Lemmatizes each word to its base form.
  - Normalizes the text to lowercase and removes whitespace.

### 3. Machine Learning Pipeline

- **Pipeline Components**:
  - `CountVectorizer`: Converts text data into a matrix of token counts.
  - `TfidfTransformer`: Transforms the count matrix to a normalized term-frequency or TF-IDF representation.
  - `MultiOutputClassifier`: A wrapper that allows using multiple classifiers, one for each output category.
  - `RandomForestClassifier`: The model used for classifying each category.
- **Model Training**: The model is trained on the disaster data and evaluated on test data.

### 4. Model Saving

- **Save Model**: The trained model is saved as a pickle file (`classifier.pkl`) for later use.

## Repository Contents

- **process_data.py**: Script to process the data, clean it, and save it to an SQLite database.
- **train_classifier.py**: Script to load the data, train the machine learning model, and save the trained model as a pickle file.
- **README.md**: This file, providing an overview of the project.
- **data/**: Directory containing the input datasets.
- **models/**: Directory where the trained model will be saved.

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/stephengardnerd/DataEngineering_MLPipeline.git
```

### 2. Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### 3. Run Data Processing Script

Process the data and save it to the SQLite database:

```bash
python process_data.py
```

### 4. Run Model Training Script

Train the model and save it as a pickle file:

```bash
python train_classifier.py
```

### Note

The final trained model file (`classifier.pkl`) is over 900 MB in size and could not be uploaded to GitHub due to file size limitations. Please run the script locally to generate the model file.

## Acknowledgments

This project was completed as part of the Udacity Data Engineering Nanodegree program.

## Contact

For any questions or issues, please reach out through GitHub.

---

