# Mood Prediction Using Time Series Data

## Overview
This project involves the application of data mining techniques to predict mood based on data collected from smartphone sensors. The objective is to utilize both classification and regression models to predict users' moods from their smartphone usage patterns. 
The advanced dataset used in this project originates from the domain of mental health applications. These applications gather various types of sensory data about user behavior and mood ratings. The dataset includes time-stamped entries of multiple variables, such as mood scores, activity levels, and app usage durations.

### Files in the Dataset
- `dataset_mood_smartphone.csv`: Initial raw dataset including user IDs, timestamps, variables, and corresponding values.
- `dataset_outliers_removed.csv`: Dataset after removal of outliers and erroneous entries.
- `cleaned_dataset.csv`: Dataset used for modeling, after extensive cleaning and preprocessing.

## Feature Engineering
Feature engineering was performed to enhance the dataset's predictive capabilities. This involved generating new features, such as rolling averages and time-based aggregations, to capture trends and patterns over time. The goal was to transform the time series data into a format suitable for machine learning models.

## Models
### Random Forest Classifier
The Random Forest Classifier was employed to classify mood based on the engineered features. It involved tuning various hyperparameters to optimize model performance, primarily focusing on accuracy and F1-score.

### Random Forest Regressor
The Random Forest Regressor was used for predicting numerical mood scores. Similar to the classifier, this model underwent extensive hyperparameter tuning, focusing on minimizing prediction errors.

### LSTM Models
#### LSTM Regressor
This model utilized a Long Short-Term Memory (LSTM) network to handle the sequential nature of the data. It was designed to predict mood scores as a continuous output, optimizing for both mean squared error and mean absolute error.

#### LSTM Sequence-to-Label Model
The sequence-to-label LSTM model was crafted to predict mood from sequences of daily observations. This model was particularly aimed at utilizing the temporal dependencies inherent in the dataset.

## Python Files and Notebooks
- `data_cleaning.py`: Scripts for initial data cleaning.
- `data_exploration.py`: Scripts for exploratory data analysis, including visualization.
- `feature_engineering.py`: Scripts for generating and selecting features suitable for modeling.
- `random_forest_classifier.py` and `random_forest_regressor.py`: Implementation of Random Forest models.
- `LSTM_Regressor.ipynb` and `LSTM_s2label_model.ipynb`: Jupyter notebooks detailing the development and tuning of LSTM models.

## Running the Notebooks
To execute the notebooks, ensure Python 3.8+ is installed along with libraries like numpy, pandas, matplotlib, scikit-learn, and TensorFlow. It is recommended to use a virtual environment:
The notebooks for LSTM and the Random Forest Scripts imports feature_engineering.py. 
