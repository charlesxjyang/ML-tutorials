from preprocessing import *

'''See READ_ME.md to understand what format the function inputs X,y should be'''

def logistic_reg_cv(X,y,iter_number=500,num_folds=4,split=0.2):
    from sklearn.linear_model import LogisticRegressionCV
    X_train,X_test,y_train,y_test = processing(X,y,split=split)
    C = np.logspace(-4,4,20)
    #saga, is for l1 penalty, newton-cg, lbfgs and sag for l2
    model = LogisticRegressionCV(Cs = C, cv=num_folds ,penalty='l1',solver='liblinear',max_iter=iter_number)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    error_report(y_test,y_pred)
    coef = model.coef_
    return coef

