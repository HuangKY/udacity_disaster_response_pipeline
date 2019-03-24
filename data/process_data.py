import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load message data and change the categories column 
    into separated category columns with 0 & 1 values.

    INPUT: messages_filepath, categories_filepath
    OUTPUT: concatenated dataframe
    '''
    # Load the two main datasets: message & category
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge to one big dataset
    df = pd.merge(messages, categories, how='inner', on='id')

    # Split categories into separated columns
    categories = df.categories.str.split(";", expand=True)

    # use the first row to get the name of the categories, and then rename the columns
    row = categories.iloc[0]
    category_colnames = row.str.split('-',expand=True)[0]
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    return df
    

def clean_data(df):
    '''
    Remove duplicates in a given dataframe.

    INPUT: dataframe
    OUTPUT: cleaned dataframe
    '''    
    # Remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filepath):
    '''
    Save a dataframe into a SQL database with a table name called "main_data".

    INPUT: dataframe, database_filepath
    OUTPUT: None
    '''    
    database_path = 'sqlite:///' + database_filepath

    engine = create_engine(database_path)
    df.to_sql('main_data', engine, index=False)  


def main():
    '''
    Execute an ETL process to a give message data.

    INPUT: None
    OUTPUT: None
    '''    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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