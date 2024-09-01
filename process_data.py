# process_data.py

# Import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data():
    # Load the messages dataset
    messages = pd.read_csv('messages.csv')

    # Display the first few rows of the messages dataframe
    print("Messages Dataframe:")
    print(messages.head())

    # Load the categories dataset
    categories = pd.read_csv('categories.csv')

    # Display the first few rows of the categories dataframe
    print("\nCategories Dataframe:")
    print(categories.head())

    # Merge messages and categories datasets on the common 'id' column
    df = pd.merge(messages, categories, on='id')

    # Display the first few rows of the combined dataset
    print("Combined DataFrame:")
    print(df.head())

    return df

def clean_data(df):
    # Create a dataframe of the 36 individual category columns
    # Split the categories column into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Display the first few rows of the categories DataFrame to verify the split
    print("Split categories DataFrame:")
    print(categories.head())

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    print("Category column names extracted:")
    print(category_colnames)

    # Rename the columns of the categories DataFrame with new column names
    categories.columns = category_colnames

    # Display the new column names
    print("New category column names:")
    print(categories.columns)

    # Iterate through the category columns to convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        
    # Display the first few rows of the updated categories DataFrame
    print("Updated categories DataFrame:")
    print(categories.head())

    # Drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Display the first few rows of the updated df DataFrame
    print("Updated df DataFrame:")
    print(df.head())

    # Check the number of duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")

    # Drop duplicates
    df = df.drop_duplicates()

    # Check the number of duplicates again
    duplicate_count_after = df.duplicated().sum()
    print(f"Number of duplicate rows after removal: {duplicate_count_after}")

    return df

def save_data(df, database_filename):
    # Save the clean dataset into an SQLite database
    engine = create_engine(f'sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    print(f"Data has been successfully saved to the database 'DisasterResponse'.")

def main():
    df = load_data()
    df = clean_data(df)
    save_data(df, 'DisasterResponse.db')

if __name__ == "__main__":
    main()
