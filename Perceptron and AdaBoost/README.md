## Perceptron
Implement perceptron from scratch and perform
empirical risk minimization
10-fold cross-validation

## AdaBoost
Implement AdaBoost from scratch. For weak learners, he hypothesis class of decision stumps on the five feature axes (in breast cancer dataset) is used. Perform
empirical risk minimization
10-fold cross-validation


## Run
To run the perceptron algorithm in erm mode, execute \
`python perceptron.py -d path to data -m erm -e 0 -b 1.0`

To run perceptron algorithm in kfold mode, execute \
`python perceptron.py -d path to data -m kfold -k 10 -e 0 -b 1.0`

To run the adaboost algorithm in erm mode, execute\
`python adaboost.py -d path to data -m erm -t 10`

To run adaboost algorithm in kfold mode with validation error v/s erm plot, execute python ad-
`aboost.py -d path to data -m kfold -t 10 -k 10 -p`

For more details refer [report](https://github.com/ajaygk95/Machine-Learning/blob/master/Perceptron%20and%20AdaBoost/CSE512-HW1-Ajay_Gopal_Krishna%20-%20Report_README.pdf)
