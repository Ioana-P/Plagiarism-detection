# Plagiarism Project, Machine Learning Deployment

This repository contains code and associated files for deploying a plagiarism detector using AWS SageMaker. Significant parts of this code were provided by Udacity via their Machine Learning Engineer Nanodegree. Final performance of the most recent model on test data is shown in confusion matrix below:

![confusion_matrix_RF](https://github.com/Ioana-P/Plagiarism-detection/blob/master/final_conf_matrix.jpg)

## Project Overview

The aim of this project is to build a plagiarism detector that examines a text file and performs binary classification; labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

This project will be broken down into three main notebooks:

File navigation:
**1_Data_Exploration: Data Exploration**
* Loaded in the corpus of plagiarism text data.
* Explored the existing data features and the data distribution.

**2_Plagiarism_Feature_Engineering: Feature Engineering**

* Cleaned and pre-processed the text data.
* Defined features for comparing the similarity of an answer text and a source text, and extracted similarity features.
* Selected "good" features, by analyzing the correlations between different features.
* Created train/test split `.csv` files that hold the relevant features and class labels for train/test data points.

**3_Training_a_Model: Train and Deploy Your Model in SageMaker**

* Upload your train/test feature data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it using SageMaker.
* Evaluate your deployed classifier.

Other Files:
helpers.py - additional cleaning, preprocessing and feature engineering functions utilized in the notebooks
problem_unittests.py - unit tests provided by Udacity

data/
    - `*.txt` all text files containing the actual student submission / wikipedia answer
    - `file_information.csv` table containing metadata on all txt files 
    
    
