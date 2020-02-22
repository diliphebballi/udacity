# Image Classifier Project

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

The goal of this project is to train an image classifier to recognize different species of flowers

## Software and Libraries
This project uses Python 3.7.2 and the following libraries:
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [scikit-learn](http://scikit-learn.org/stable/)
* [Matplotlib](http://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
...

## Data
The dataset is provided by [Udacity](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and is composed by:
* **flowers/test**: .jpg images for test
* **flowers/train** .jpg images to train the classifier
* **flowers/valid** .jpg images for validation
* **cat_to_name.json**: dictionary mapping the integer encoded categories to the actual names of the flowers

This set contains images of flowers belonging to 102 different categories. The images were acquired by searching the web and taking pictures. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories

More information in [Automated Flower Classification over a Large Number of Classes by M. Nilsback, A. Zisserman] (http://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08)

## Running the code

The code is provided in a [Jupyter Notebook](http://ipython.org/notebook.html) then converted in a command line application:
* _part_1/image_classifier_project.ipynb_: Load data, Building and training the classifier an testing
* _part_2/train.py_: train a new network on a dataset and save the model as a checkpoint
* _part_2/predict.py_: uses a trained network to predict the class for an input image

If you donwload it simply run the command `jupyter notebook image_classifier_project.ipynb` in the folder were the file is located.

## Results

Results are better explained in this [blog post](https://medium.com/@simone.rigoni01/)

## Licensing and Acknowledgements

Thank you Udacity for the datasets and more information about the licensing of the data can be find [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
