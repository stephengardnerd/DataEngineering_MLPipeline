# Disaster Response Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Libraries](https://img.shields.io/badge/Tools-Flask%20%7C%20Pandas%20%7C%20Scikit--Learn-yellow)

## Overview
This repository contains a full end-to-end data engineering and machine learning pipeline that classifies disaster-related messages in real time. The pipeline facilitates the rapid triage of incoming messages (from social media, texts, etc.) to immediately route them to the correct disaster response agencies. This repository demonstrates proficiency in ETL data manipulation, Natural Language Processing (NLP), and deploying a Random Forest Classifier via a Flask web application.

## Key Features
- **ETL Pipeline**: Extracts text data and categories from raw resources, cleans features via Pandas, and loads them into a normalized SQLite database.
- **ML Pipeline**: A robust `scikit-learn` pipeline using a custom NLP tokenizer, TF-IDF vectorization, and a `RandomForestClassifier` optimized via `GridSearchCV`.
- **Flask Web Application**: A web frontend used to dynamically test incoming messages against the model and visualize the disaster distribution using Plotly.

## Tech Stack
| Component | Technology | Purpose |
| --- | --- | --- |
| ETL Pipeline | `Pandas`, `SQLAlchemy` | Ingesting, cleaning, and storing disaster data. |
| ML Pipeline | `NLTK`, `Scikit-Learn` | NLP tokenization and model training. |
| Web Application | `Flask`, `Plotly`| Providing an accessible front-end interface and data visualizations. |

## Quick Start (Installation & Execution)

### 1. Set Up the Environment
Ensure you have Python installed. You can set up your isolated environment using pip and the provided `requirements.txt`.

```bash
git clone https://github.com/stephengardnerd/DataEngineering_MLPipeline.git
cd DataEngineering_MLPipeline

# Install requirements
pip install -r requirements.txt
```

### 2. Run the ETL Pipeline
The ETL script merges messages and categories, cleans the data, and stores it in an SQLite database.
```bash
python disaster_response_pipeline_project/data/process_data.py \
    disaster_response_pipeline_project/data/disaster_messages.csv \
    disaster_response_pipeline_project/data/disaster_categories.csv \
    disaster_response_pipeline_project/data/DisasterResponse.db
```

### 3. Run the ML Pipeline
This script trains the model, performs a grid search, and saves it as a pickle file.
```bash
python disaster_response_pipeline_project/models/train_classifier.py \
    disaster_response_pipeline_project/data/DisasterResponse.db \
    disaster_response_pipeline_project/models/classifier.pkl
```

### 4. Run the Web App
Finally, execute the Flask application.
```bash
cd disaster_response_pipeline_project/app
python run.py ../data/DisasterResponse.db ../models/classifier.pkl
```
The application will launch on `http://0.0.0.0:3001/` (with debugging disabled for production readiness).

## Algorithm Justification
The model utilizes a `RandomForestClassifier` nested under a `MultiOutputClassifier`. This approach is selected because disaster response messages typically trigger multiple overlapping categories simultaneously (e.g., both "medical help" and "water"). The random forest algorithm provides an excellent balance of accuracy against over-fitting when dealing with highly sparse, high-dimensional TF-IDF matrices without needing immense compute resources.

## Author 
Stephen D. Gardner
