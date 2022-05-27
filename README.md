# Summary of Active Learning

This code uses a unique machine learning method known as active learning. Active learning consists of strategies which minimize the number of samples that need to be labeled in order to optimize accuracy. This method is most useful when there is a cost in time or money to label samples. Some of the active learning strategies include the following. These strategies can be combined or used separately. 

1. Uncertainty - choose the next sample that the model is least certain about
2. Density - choose the sample that is most representative of unlabeled samples
3. Query by Committee - choose the sample that has the largest disagreement between a committee of models

# How to Test the Performance of an Active Learning Strategy 
In order to test the performance of an active learning strategy, a simulation using labeled samples is performed. Five random samples are added to the training set, while the other samples are placed in the test set. Then, a sample is removed from the test set and added to the training set. This sample is chosen based on the active learning stategy being used. The simulation proceeds until a specified number of samples are in the training set. In this case, I chose 50 samples. Each time a new sample is added to the training set, the model is tested for its accuracy in predicting the test set labels. Additionally, 5-cross fold validation is performed on the training set. 

# Code Overview

The scripts can be summarized as follows:

- uncertainty.py - uses the uncertainty method. In this case K-nearest Neighbors method was used for the model and the next sample had the highest variance of the K-nearest neighbors.
- cv.py - performs cross-validation of data
- classification.py - executes the program
