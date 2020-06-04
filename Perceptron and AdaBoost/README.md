##Dataset
The linearly-separable-dataset is an artificial linearly separable dataset with the domain X = R2, and the label set Y = {0,1}.

The second is a Breast Cancer Prediction dataset. Diagnosis of breast cancer is performed when an abnormal lump is found (from self-examination or x-ray) or a tiny speck of calcium is seen (on an x-ray). In this dataset, there are 569 observations, with each observation consisting of 5 features (mean radius, mean texture, mean perimeter, mean area, mean smoothness). The last column is the diagnosis, where 0 indicates that the finding was benign, and 1 indicates that it was malignant.

##Perceptron
Implement perceptron from scratch and perform
empirical risk minimization
10-fold cross-validation

##AdaBoost
Implement AdaBoost from scratch. For weak learners, he hypothesis class of decision stumps on the five feature axes (in breast cancer dataset) is used. Perform
empirical risk minimization
10-fold cross-validation


##Run
To run the perceptron algorithm in erm mode, execute \
`python perceptron.py -d path to data -m erm -e 0 -b 1.0`

To run perceptron algorithm in kfold mode, execute \
`python perceptron.py -d path to data -m kfold -k 10 -e 0 -b 1.0`

To run the adaboost algorithm in erm mode, execute\
`python adaboost.py -d path to data -m erm -t 10`

To run adaboost algorithm in kfold mode with validation error v/s erm plot, execute python ad-
`aboost.py -d path to data -m kfold -t 10 -k 10 -p`

For more details refer [report](https://github.com/ajaygk95/Machine-Learning/blob/master/Perceptron%20and%20AdaBoost/CSE512-HW1-Ajay_Gopal_Krishna%20-%20Report_README.pdf)
