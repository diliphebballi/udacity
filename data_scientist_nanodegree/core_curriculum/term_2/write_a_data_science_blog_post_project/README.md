# Write A Data Science Blog Post Project

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

The goal of this project is to put in practice the technical skills teached during the program but manly to focus on the ability to effectively communicate the results of the analysis. In particular the **CRISP-DM Process** (Cross Industry Process for Data Mining) which is composed by the following steps:
1. Business Understanding
2. Data Understanding
3. Prepare Data
4. Data Modeling
5. Evaluate the Results
6. Deploy

## Software and Libraries
This project uses Python 3.7.2 and the following libraries:
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [nltk](https://www.nltk.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [Matplotlib](http://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [TextBlob](https://textblob.readthedocs.io/en/dev/)
* WordCloud

## Data
The dataset is provided by [Airbnb](http://insideairbnb.com/get-the-data.html) and is basically composed by:
* **listings.csv**: Detailed Listings data for Milan
* **calendar.csv**: Detailed Calendar Data for listings in Milan
* **reviews.csv**: Detailed Review Data for listings in Milan
* **summary_listings.csv**: Summary information and metrics for listings in Milan (good for visualisations).
* **summary_reviews.csv**: Summary Review data and Listing ID (to facilitate time based analytics and visualisations linked to a listing).
* **neighbourhoods.csv**: Neighbourhood list for geo filter. Sourced from city or open source GIS files.
* **neighbourhoods.geojson**: GeoJSON file of neighbourhoods of the city.

To map the neighbourhood in the dataset to the real neighbourhood I have created the file **quartieri.txt** which contains the real neighbourhood of Milan.

## Running the code

The code is provided in the [Jupyter Notebook](http://ipython.org/notebook.html) _neighborhoods_sentiment_analysis.ipynb_
If you donwload it simply run the command `jupyter notebook neighborhoods_sentiment_analysis.ipynb` in the folder were the file is located.

## Results

Results are better explained in this [blog post](https://medium.com/@simone.rigoni01/do-you-want-to-move-to-milan-neighborhoods-sentiment-analysis-using-airbnb-data-72db72ebc070?sk=52ad4741d36bb8a3005b2ad13832d622)

## Licensing and Acknowledgements

Thank you [Airbnb](https://www.airbnb.com/) for the datasets and more information about the licensing of the data can be find [here](http://insideairbnb.com/about.html).
