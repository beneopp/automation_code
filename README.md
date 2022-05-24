# Summary of Active Learning

This code uses a unique machine learning method known as active learning. Active learning consists of strategies designed to produce an accurate model while minimizing the number of samples that need to be labeled in order to optimize accuraccy. This method can be used when there is a cost in terms of time or money to label samples for a machine learning model. Some of the active learning strategies include the following below. These strategies can be combined or used separately. 

1. Uncertainty - the next sample is the sample the model is least certain about
2. Density - choose the sample that is most representative of unlabeled samples
3. Query by Committee - choose the sample that has the largest disagreement between a committee of models

# How to Test the Performance of an Active Learning Strategy 
In order to test the performance of an active learning strategy, a simulation with data is performed where all samples are labeled. Five random samples are added to the training set. The other samples are treated as unlabeled data in the test set. Next, an unlabeled sample is added to the training set using the chosen active learning stategy. At each round of adding a new sample to the training set, the accuraccy both the unlabeled data and of the labeled data with cross-validation are determined. The accuracy is plotted as a function of the number of labeled samples in the model.

# Code Overview

The scripts can be summarized as follows:

- uncertainty.py - uses the uncertainty method. In this case K-nearest Neighbors method was used for the model and the next sample had the highest variance of the k-nearest neighbors.
- cv.py - performs cross-validation of data
- classification.py - executes program
