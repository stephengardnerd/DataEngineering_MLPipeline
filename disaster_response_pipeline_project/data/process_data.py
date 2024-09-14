import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets, merge them on 'id'.

    Args:
    messages_filepath (str): Filepath for the csv file containing the messages dataset.
    categories_filepath (str): Filepath for the csv file containing the categories dataset.

    Returns:
    df (pandas.DataFrame): DataFrame containing the merged content of messages and categories datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataframe by splitting categories into separate columns,
    converting category values to binary (0 or 1), removing observations with
    category values of 2, and removing duplicates.

    Args:
    df (pandas.DataFrame): DataFrame containing the merged content of messages and categories datasets.

    Returns:
    df (pandas.DataFrame): DataFrame containing cleaned data with each category as a separate column.
    """
    # Split the categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names for the categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to integers and drop rows with '2'
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    # Drop rows with any category value equal to 2
    categories = categories[categories.isin([0, 1]).all(axis=1)]
    
    # Drop the original categories column from the dataframe
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataset into an SQLite database.

    Args:
    df (pandas.DataFrame): DataFrame containing cleaned data.
    database_filename (str): Filename for the output database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    print(f"Data has been successfully saved to the database '{database_filename}'.")

def main():
    """
    Main function that executes the ETL pipeline:
    - Loads data from the specified filepaths.
    - Cleans the data.
    - Saves the cleaned data to an SQLite database.

    The script expects three arguments:
    1. Filepath for the messages dataset (CSV).
    2. Filepath for the categories dataset (CSV).
    3. Filepath for the SQLite database where the cleaned data will be saved.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
