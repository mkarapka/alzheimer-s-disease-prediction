from sklearn.preprocessing import StandardScaler


def data_preprocessing(X_train, X_test):
    scaler = StandardScaler()
    X_train_scld = scaler.fit_transform(X_train)
    X_test_scld = scaler.transform(X_test)
    return X_train_scld, X_test_scld
