# Disaster Response Pipeline Project

## Introduction

This project is part of The [Udacity](https://eu.udacity.com/) Data Scientist Nanodegree Program which is composed by:
* Term 1
    * Supervised Learning
    * Deep Learning
    * Unsupervised Learning
* Term 2
    * Write A Data Science Blog Post
    * Disaster Response Pipelines
    * Recommendation Engines

The goal of this project is to apply the data engineering skills learned in the course to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. The project is divided in three sections:
* **Data Processing**: build an ETL (Extract, Transform, and Load) Pipeline to extract data from the given dataset, clean the data, and then store it in a SQLite database
* **Machine Learning Pipeline**: split the data into a training set and a test set. Then, create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that predicts a message classifications for the 36 categories (multi-output classification)
* **Web development** develop a web application to show classify messages in real time

## Software and Libraries

This project uses Python 3.7.2 and the following libraries:
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [nltk](https://www.nltk.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [sqlalchemy](https://www.sqlalchemy.org/)
* [dash](https://plot.ly/dash/)

## Data

The dataset is provided by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/) is basically composed by:
* **disaster_categories.csv**: Categories of the messages
* **disaster_messages.csv**: Multilingual disaster response messages

## Running the code

From the project folder run `python dash_disaster_response_pipeline.py` to start the dash application. The default url to connect to it is http://127.0.0.1:8050/.

<pre>
|-- disaster_response_pipeline_project
    |-- disaster_response_pipeline
    |   |-- data
    |   |   |-- ETL Pipeline Preparation.ipynb
    |   |   |-- process_data.py    
    |   |   |-- disaster_categories.csv 
    |   |   |-- disaster_messages.csv     
    |   |
    |   |-- classifier
    |   |   |-- ML Pipeline Preparation.ipynb
    |   |   |-- train_classifier.py
    |   |   |-- trained_classifier.pkl
    |   |
    |   |-- disaster_response_pipeline.py
    |   |-- dash_disaster_response_pipeline.py
    |
    |-- db.sqlite3
    |-- db_dump.txt
    |-- README.md
</pre>

If the application does not find the **trained_classifier.pkl** pickle file to load the model it will check also if the database **db.sqlite3** present and if not process the data and finaly train the model (save it in **classifier/trained_classifier.pkl**) to get the application ready to classify messages in real time.

![Flowchart](images/flowchart.png)

Flowchart made using [draw.io](https://about.draw.io/)

All the modules provide the help funcionality provided by [argparse](https://docs.python.org/3/library/argparse.html) module.

By default the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for the best parameters for the model is disable because of the long time required to perform it. To perform it run **train_classifier.py** with the option `--grid_search_cv`

To run the the preparation [Jupyter Notebook](http://ipython.org/notebook.html) run the command `jupyter notebook ETL_Pipeline_Preparation.ipynb` or `jupyter notebook ML_Pipeline_Preparation.ipynb` in the folder were the file is located.    

Using sqlite3 command shell is possible to extract a dump of the database if needed:

`sqlite3 db.sqlite3`

`sqlite3> .output db_dump.txt`

`sqlite3> .dump`

`sqlite3> .quit`

## Results


## Licensing and Acknowledgements

Thank you [Figure Eight](https://www.figure-eight.com/) for the datasets and more information about the licensing of the data can be find [here](https://www.figure-eight.com/datasets/).
