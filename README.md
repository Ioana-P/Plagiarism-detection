# Plagiarism Project, Machine Learning Deployment

This repository contains code and associated files for deploying a plagiarism detector using AWS SageMaker. Significant parts of this code were provided by Udacity via their Machine Learning Engineer Nanodegree. Final performance of the most recent Sagemaker model (which was a Sagemaker SKLearn RandomForest) on test data is shown in confusion matrix below. Our final model had an accuracy of 96% and a ROC AUC of 0.95. 

![confusion_matrix_RF](https://github.com/Ioana-P/Plagiarism-detection/blob/master/final_conf_matrix_2.png)

For the multilabel classification task the goal was to be able to predict whether an answer was:
- **cut** and paste
- **light**ly modified from cut and paste
- **heav**ily modified from cut and paste
- **non**-plagiarised

The results are visible in the matrix below. Our final accuracy on test data was 71% using a Random Forest model as well. 

![confusion_matrix_RF_multilabel](https://github.com/Ioana-P/Plagiarism-detection/blob/master/multilabel_conf_matrix.png)


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

**3a_AWS_Training_a_Model: Train and Deploy Your Model in SageMaker**

* Uploading train/test feature data to S3.
* Defined a binary classification model and a training script.
* Trained our model and deployed it using SageMaker.
* Evaluated our deployed classifier.

**3b_Training_local_Model: Train and Deploy Your Model in SageMaker**

* Trained and optimised several sklearn models on binary classification
* Trained and optimised several sklearn models on multiclass classification
* Visualised best model performances on test data

Other Files:
helpers.py - additional cleaning, preprocessing and feature engineering functions utilized in the notebooks
problem_unittests.py - unit tests provided by Udacity

data/
    - `*.txt` all text files containing the actual student submission / wikipedia answer
    - `file_information.csv` table containing metadata on all txt files 
    
References:
Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [Download]