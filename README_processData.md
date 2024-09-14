Here’s how you can run the `process_data.py` script. The instructions assume you have the necessary Python environment set up and that the `process_data.py` script is stored in the correct directory.

## README Instructions for `process_data.py`

### Project Overview

The `process_data.py` script is used to process disaster response messages and categories datasets. It loads, cleans, and merges these datasets, and then stores the cleaned data in an SQLite database. This processed data will later be used for building a machine learning model to classify disaster-related messages.

### File Locations

In this example, we assume the following file paths:

- **Messages File:** `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data/disaster_messages.csv`
- **Categories File:** `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data/disaster_categories.csv`
- **SQLite Database:** `/Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data/DisasterResponse.db`

### Prerequisites

Ensure that you have Python 3.x installed and that you have installed the required libraries:

```bash
pip install pandas sqlalchemy
```

### Running the Script

To run the script, follow these steps:

1. **Navigate to the Directory**: 
   First, open your terminal or command prompt and navigate to the directory containing your script:

   ```bash
   cd /Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data
   ```

2. **Run the Script**:
   Execute the `process_data.py` script by passing the required arguments:

   ```bash
   python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
   ```

   This command does the following:
   - **Loads the Data**: Reads the messages and categories datasets.
   - **Cleans the Data**: Splits the categories into individual columns, converts the values to binary, and removes duplicates.
   - **Saves the Data**: Stores the cleaned data in an SQLite database (`DisasterResponse.db`).

3. **Expected Output**:
   - The script will print messages to the console indicating the progress, such as loading data, cleaning data, and saving data to the database.
   - Once the process is complete, the cleaned data will be stored in the specified SQLite database.

### Script Usage Example

Here’s what a typical usage of the script looks like in your terminal:

```bash
cd /Users/Mach15/Documents/GitHub/DataEngineering_MLPipeline/disaster_response_pipeline_project/data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

Upon successful execution, you should see output indicating that the data has been loaded, cleaned, and saved to the database.

### Troubleshooting

- **File Not Found Error**: If you encounter a file not found error, ensure that the file paths provided are correct.
- **Python Import Errors**: Ensure all required Python packages (`pandas`, `sqlalchemy`) are installed.

### Acknowledgments

This project is part of the Data Engineering Nanodegree Program by Udacity. The dataset used is provided by [Figure Eight](https://www.figure-eight.com/).

### Contact

For any questions or issues, please reach out through GitHub.

---

