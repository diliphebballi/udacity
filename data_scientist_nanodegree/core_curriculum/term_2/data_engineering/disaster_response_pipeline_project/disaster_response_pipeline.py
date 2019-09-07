# Disaster Response Pipeline to classify input message
# python .\disaster_response_pipeline.py

import os
import pickle
import classifier.train_classifier
import data.process_data
import argparse


CATEGORIES_FILENAME = 'data/disaster_categories.csv'
MESSAGES_FILENAME = 'data/disaster_messages.csv'
DATABASE_FILENAME = 'db.sqlite3'
MODEL_PICKLE_FILENAME = 'classifier/trained_classifier.pkl'
DEFAULT_TEST_MESSAGE = 'Storm at sacred heart of Jesus'


#TODO: check if table is present in the database
#def check_database_data(database_filename):


def get_category_names(database_filename = DATABASE_FILENAME):
    '''
    Return category names

    Args:
        database_filename (str): database filename. Default value DATABASE_FILENAME

    Returns:
        category_names (list): list of the category names
    '''
    df = classifier.train_classifier.get_df_from_database(database_filename)
    return list(df.columns[4:])


def get_genre_distribution(database_filename = DATABASE_FILENAME):
    '''
    Return message genre distribution

    Args:
        database_filename (str): database filename. Default value DATABASE_FILENAME

    Returns:
        genre_distribution (dict): dictionary of message genre distribution (genre, count)
    '''
    df = classifier.train_classifier.get_df_from_database(database_filename)
    return df.groupby('genre').count()['message'].to_dict()


def get_top_n_categories(database_filename = DATABASE_FILENAME, n = 0):
    '''
    Return the top n message categories

    Args:
        database_filename (str): database filename. Default value DATABASE_FILENAME
        n (int): number of categories to be considered. Default value 0 (In case of 0 all the categories will be considered)

    Returns:
        top_n_categories (dict): dictionary of the n top categories (category, count)
    '''
    df = classifier.train_classifier.get_df_from_database(database_filename)
    if n == 0:
        n = len(get_category_names(database_filename))
    return df.iloc[:, 4:].sum().sort_values(ascending = False)[1:n].to_dict()


def get_predicted_category_names(category_predicted):
    '''
    Return the names of the categories corresponding to predicted one in input

    Args:
        category_predicted (list): list of predicted category

    Returns:
        predicted_category_names (list): list of predicted category names
    '''
    category_names = get_category_names(DATABASE_FILENAME)
    return [category_names[i] for i in range(len(category_predicted)) if category_predicted[i] == 1]


def load_pipeline(categories_filename = CATEGORIES_FILENAME, messages_filename = MESSAGES_FILENAME, database_filename = DATABASE_FILENAME, model_pickle_filename = MODEL_PICKLE_FILENAME):
    '''
    Return an istance of the model created. If the model pickle file is not present the model will be trained and the file cretaed. There is also a check if the 
    training datataset is available. If not the processing of the data will be performed and saved into the database to allow to train the model

    Args:
        categories_filename (str): categories filename. Default value CATEGORIES_FILENAME
        messages_filename (str): messages filename. Default value MESSAGES_FILENAME
        database_filename (str): database filename. Default value DATABASE_FILENAME
        model_pickle_filename (str): pickle filename. Default value MODEL_PICKLE_FILENAME
    
    Returns:
        model (pipeline.Pipeline): model loaded 
    '''
    print('Check if model present...\n    Model: {}'.format(model_pickle_filename))
    if os.path.isfile(model_pickle_filename) == False:
        print('Not present. Training the model...\nCheck if data are in database...')
        if os.path.isfile(DATABASE_FILENAME) == False:
            data.process_data.process(messages_filename = messages_filename, categories_filename = categories_filename, database_filename = database_filename)

        classifier.train_classifier.train(database_filename = database_filename, model_pickle_filename = model_pickle_filename)
    else:
        print('Ok')

    print('Loading model...\n    Model: {}'.format(model_pickle_filename))
    model = classifier.train_classifier.load_model(model_pickle_filename)
    category_names = get_category_names(database_filename)
    return model


def parse_input_arguments():
    '''
    Parse the command line arguments

    Returns:
        message (str): message to be classified
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline")
    parser.add_argument('--message', type = str, default = DEFAULT_TEST_MESSAGE, help = 'Message to classify')
    args = parser.parse_args()
    #print(args)
    return args.message


if __name__ == '__main__':
    message = parse_input_arguments()
    model = load_pipeline()
    category_predicted = model.predict([message])[0]
    print('Message to classify: {}\nCategories:'.format(message))
    print(get_predicted_category_names(category_predicted))
