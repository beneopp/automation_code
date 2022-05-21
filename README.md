# Automation Code

This code uses a unique machine learning method known as active learning. This method can be used when there is a cost in terms of time or money to label
samples for a machine learning model. This method is designed to produce an accurate model while minimizing the number of samples that are needed to label
in order to achieve this accuraccy. There are three methods that can be combined if it is helpful:

1. Uncertainty - the next sample is the sample the model is least certain about
2. Density - choose the sample that is most representative of unlabeled samples
3. Query by Committee - choose the sample that has the largest disagreement between a committee of models
