def normalization(X_train, X_test):
    '''Normalizes the training set to have a mean of 0 and std of 1
    and then applies the learned transformation to X_test
    In this way, we avoid information leakage
    This function returns the transformed X_train,X_test, and the scaler object
    should we choose to reverse transform the final predicted results'''
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train) 
    #mean = scaler.mean_
    #std = scaler.var_
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, scaler

def test_train_split(X,y,split=0.20):
    '''Splits X,y into X_train,X_test,y_train,y_test randomly
    assumes that data is exchangeable'''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train, X_test, y_train, y_test

def processing(X,y,split=0.2):
    '''This function combines both normalization and test_train_split'''
    X_train, X_test, y_train, y_test = test_train_split(X,y,split=split)
    X_train, X_test, scaler = normalization(X_train,X_test)
    #convert panda df to np array
    y_train,y_test = y_train.values,y_test.values
    return X_train,X_test,y_train,y_test