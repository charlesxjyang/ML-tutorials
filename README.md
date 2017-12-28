# ML-tutorials
This repository houses previous models I've constructed for various projects in one location. This is a functional version of various sklearn, keras, and tf models for supervised regression and classifications problems. Designed to be immediately usable, the only non-default formal parameter inputs to the functions are X,y which should be in the form of Pandas DataFrame and Pandas Series respectively( should also work for numpy array), where X contains the features as columns and y is a single target variable.

# Types of Models included
There are 4 main models developed in this repository: linear regression(ordinary least squares(OLS), lasso(l1 regularizer), and ridge(l2 regularizer)), Support Vector Machines(SVM), Adaboost Ensemble Method, and Artificial Neural Networks(ANN). These 4 models cover a wide breadth of the developed Machine Learning model frameworks and should provide robust results for properly cleaned datasets for supervised regression/classification. 

# Documentation/Resources

## Linear Regression

OLS

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

Lasso

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

Ridge

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

Difference between Lasso and Ridge

(code-heavy)

https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/

(math-heavy)

https://blog.alexlenail.me/what-is-the-difference-between-ridge-regression-the-lasso-and-elasticnet-ec19c71c9028

## Support Vector Machines
Regression

http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

Classification

http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

## Adaboost
Regression

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

Classification

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Random Forests

https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

https://www.kdnuggets.com/2017/10/random-forests-explained.html

https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674

Adaboost

https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
## Artificial Neural Networks
