from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def LabelEncoderTrain(objectlist_train, loan_train):

    for feature in objectlist_train:
        loan_train[feature] = le.fit_transform(loan_train[feature].astype(str))

    print (loan_train.info())

def LabelEncoderTest(objectlist_test, loan_test):
    for feature in objectlist_test:
        loan_test[feature] = le.fit_transform(loan_test[feature].astype(str))
    print (loan_test.info())