# Process the data and save it in a database
# python process_data.py --messages_filename disaster_messages.csv --categories_filename disaster_categories.csv --database_filename ../../db.sqlite3

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import argparse

CATEGORIES_FILENAME = 'disaster_categories.csv'
MESSAGES_FILENAME = 'disaster_messages.csv'
DATABASE_FILENAME = '../db.sqlite3'
TABLE_NAME = 'disaster_message'


def load_data(messages_filename, categories_filename):
    '''
    Load the data from the input files

    Args:
        categories_filename (str): categories filename
        messages_filename (str): messages filename

    Returns:
        df (pandas.DataFrame): dataframe containing the uncleaned dataset
    '''
    messages = pd.read_csv(messages_filename)
    #messages.head()
    categories = pd.read_csv(categories_filename)
    #categories.head()
    df = pd.merge(messages, categories, on = 'id')
    #df.head()
    return df


def clean_data(df):
    '''
    Clean the data

    Args:
        df (pandas.DataFrame): dataframe containing the uncleaned dataset

    Returns:
        df (pandas.DataFrame): dataframe containing the cleaned dataset
    '''
    categories = df.categories.str.split(pat = ';', expand = True)
    #categories.head()
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    #print(category_colnames)
    categories.columns = category_colnames
    #categories.head()
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    #categories.head()
    df = df.drop('categories', axis = 1)
    #df.head()
    df = pd.concat([df, categories], axis = 1)
    #df.head()
    df = df.drop_duplicates()
    #df.duplicated().sum()
    return df
    

def save_data(df, database_filename):
    '''
    Save the data into the database. The destination table name is TABLE_NAME

    Args:
        df (pandas.DataFrame): dataframe containing the dataset
        database_filename (str): database filename
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(TABLE_NAME, engine, index = False)  


def parse_input_arguments():
    '''
    Parse the command line arguments

    Returns:
        categories_filename (str): categories filename. Default value CATEGORIES_FILENAME
        messages_filename (str): messages filename. Default value MESSAGES_FILENAME
        database_filename (str): database filename. Default value DATABASE_FILENAME
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline Process Data")
    parser.add_argument('--messages_filename', type = str, default = MESSAGES_FILENAME, help = 'Messages dataset filename')
    parser.add_argument('--categories_filename', type = str, default = CATEGORIES_FILENAME, help = 'Categories dataset filename')
    parser.add_argument('--database_filename', type = str, default = DATABASE_FILENAME, help = 'Database filename to save cleaned data')
    args = parser.parse_args()
    #print(args)
    return args.messages_filename, args.categories_filename, args.database_filename


def process(messages_filename, categories_filename, database_filename):
    '''
    Process the data and save it in a database

    Args:
        categories_filename (str): categories filename
        messages_filename (str): messages filename
        database_filename (str): database filename
    '''
    # print(messages_filename)
    # print(categories_filename)
    # print(database_filename)
    # print(os.getcwd())

    print('Loading data...\n    Messages: {}\n    Categories: {}'.format(messages_filename, categories_filename))
    df = load_data(messages_filename, categories_filename)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    Database: {}'.format(database_filename))
    save_data(df, database_filename)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    messages_filename, categories_filename, database_filename = parse_input_arguments()
    process(messages_filename, categories_filename, database_filename)
