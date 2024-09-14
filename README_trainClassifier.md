# Disaster Response Pipeline: Train Classifier Script

## Overview

This script is designed to train a machine learning model to classify disaster-related messages into multiple categories. The model helps in routing these messages to the appropriate disaster response services. The trained model is saved as a pickle file for future use.

## File Paths

The following are the example file paths used in this guide:

- **Database File:** `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data/disaster_Response.db`
- **Model Save Location:** `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/models/classifier.pkl`

## Prerequisites

Before running the script, ensure that Python is installed on your machine along with `pip` for managing Python packages.

### Installation

You can manually install the required Python libraries, or let the script handle it automatically.

#### Manual Installation

To install the required dependencies manually, run:

```bash
pip install pandas sqlalchemy nltk scikit-learn
```

#### Automatic Installation

The script includes a function that checks for the required packages and installs them if they are not already installed.

### Running the Script

Follow these steps to run the `train_classifier.py` script:

1. **Clone the Repository:**

   If you haven't done so already, clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/disaster_response_pipeline.git
   cd disaster_response_pipeline
   ```

2. **Navigate to the Script Directory:**

   Ensure that you're in the directory where the `train_classifier.py` script is located:

   ```bash
   cd /Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project
   ```

3. **Execute the Script:**

   Run the script using Python:

   ```bash
   python train_classifier.py
   ```

4. **Provide File Paths:**

   The script will prompt you to enter the file paths for the database and the pickle file:

   - **For the database file path,** enter: `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data/disaster_Response.db`
   - **For the model file save location,** enter: `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/models/classifier.pkl`

5. **Script Workflow:**

   The script will perform the following steps:
   
   - **Load the Data:** Reads data from the specified SQLite database.
   - **Tokenize Text:** Cleans and tokenizes the text data for analysis.
   - **Build Pipeline:** Constructs a machine learning pipeline that includes a vectorizer, transformer, and classifier.
   - **Perform Grid Search:** Optimizes the model’s hyperparameters using GridSearchCV.
   - **Evaluate the Model:** Tests the model’s performance on a test dataset and prints out classification reports for each category.
   - **Save the Model:** Saves the trained model as a pickle file at the specified location.

6. **Expected Output:**

   - The script will output classification reports for each category, including precision, recall, and F1-score metrics.
   - A trained model will be saved as a `.pkl` file at the location you specified.

### Example Input Message

To test the classifier after training, you can input a message like:

```plaintext
"There is a significant need for clean drinking water in the flood-affected areas. Please send water purification tablets and bottled water as soon as possible."
```

### Troubleshooting

- **File Not Found Error:** Double-check that the file paths provided are correct and that the files exist at those locations.
- **Python Import Errors:** Ensure that the required packages (`pandas`, `sqlalchemy`, `nltk`, `scikit-learn`) are installed.

## Acknowledgements

This project is part of the Data Engineering Nanodegree Program by Udacity. The dataset used is provided by [Figure Eight](https://www.figure-eight.com/).

## Contact

For any questions or issues, please reach out through GitHub.



