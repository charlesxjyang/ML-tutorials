from preprocessing import *

'''See READ_ME.md to understand what format the function inputs X,y should be'''
###---Helper Functions---###
def error_report(y_test,y_pred):
    '''This Function prints out the relevant error metrics for each model
    Can be modified to print out various other metrics
    Can also be modified to return the values as well'''
    from sklearn.metrics import mean_squared_error, r2_score
    print("Mean squared error: {0:.02f}".format(mean_squared_error(y_test, y_pred)))
    print('Variance score aka r^2: {0:.02f}'.format(r2_score(y_test, y_pred)))

###---Linear Models---###
def lin_reg_workflow(X,y,split=0.2):
    '''Ordinary Least Squares Multivariate Linear Regression'''
    from sklearn import linear_model
    X_train,X_test,y_train,y_test = processing(X,y,split=split)
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_test)
    #coefficients of linear regression model
    coefficients = regr.coef_
    error_report(y_test,y_pred)
    return coefficients

def lassoCV_reg_workflow(X,y,split=0.2):
    '''Uses l1 regularizer e.g. performs feature selection and regularization
    This function also does automated cross-validation for alpha,
    where alpha is the parameter that controls the sparsity of non-zero variables'''
    import numpy as np
    from sklearn.linear_model import LassoCV
    X_train,X_test,y_train,y_test = processing(X,y,split=split)
    alphas = np.logspace(-4, 1, 10) #10**start, 10**end,num_samples,
    lasso_cv = LassoCV(max_iter=10**6,alphas=alphas)
    lasso_cv.fit(X_train,y_train)
    y_pred = lasso_cv.predict(X_test)
    coefficients = lasso_cv.coef_
    error_report(y_test,y_pred)    
    return coefficients
    
def ridgeCV_reg_workflow(X,y,split=0.2):
    '''Uses l2 regularizer e.g. performs feature selection and regularization
    This function also does automated cross-validation for alpha,
    but the ridge does not force parameters to have zero values'''
    import numpy as np
    from sklearn.linear_model import RidgeCV
    X_train,X_test,y_train,y_test = processing(X,y,split=split)
    alphas = np.logspace(-4, -0.5, 10) #10**start, 10**end,num_samples,
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(X_train,y_train)
    y_pred = ridge_cv.predict(X_test)
    coef = ridge_cv.coef_
    error_report(y_test,y_pred)    
    return coef

###---Decision Trees---###
def Adaboost_reg(X,y,num_ada_estimators=500,num_RF_estimators=500,max_depth=3,loss='square',split=0.2):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    X_train,X_test,y_train,y_test = processing(X,y,split=split)
    model = AdaBoostRegressor(RandomForestRegressor(n_estimators=num_RF_estimators,max_depth=max_depth),n_estimators=num_ada_estimatorss)    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    coefficients = model.estimators_
    error_report(y_test,y_pred)    
    return coefficients


def SVR(X,y,kernel='rbf',deg=3, gamma='auto', tol=0.0001, C=1.0, eps=0.1,split=0.2):
    from sklearn.svm import SVR
    X_train,X_test,y_train,y_test = processing(X,y,split=split)    
    #initialize model
    svr = SVR(kernel=kernel,degree=deg,gamma=gamma,tol=tol,C=C,epsilon=eps)
    #fit model
    svr.fit(X_train,y_train)
    #predictions
    y_pred = svr.predict(X_test)
    error_report(y_test,y_pred)
    return y_test,y_pred,X_train,X_test    

def keras_ann(X,y):
    from keras.models import Sequential
    from keras.layers import Dense
    X_train,X_test,y_train,y_test = processing(X,y)
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='sigmoid'))
    model.add(Dense(30, kernel_initializer='uniform', activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['mean_squared_error'])
    # Fit the model
    model.fit(X_train, y_train, epochs=300, verbose=2)
    # calculate predictions
    y_pred = model.predict(X_test)
    print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
    print('Variance score aka r^2: %.2f' % r2_score(y_test, y_pred))
    r2 = r2_score(y_test,y_pred)
    MSE = mean_squared_error(y_test,y_pred)
    return MSE, r2
