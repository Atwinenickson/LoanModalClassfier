# classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import pickle as pi


def Decision(train_x, train_y, test_x, test_y):
    df_model = DecisionTreeClassifier()
    df_model.fit(train_x, train_y)
    predict_y = df_model.predict(test_x)
    print(classification_report(test_y, predict_y))
    print("Accuracy:", accuracy_score(predict_y, test_y))

def Randomc(train_x, train_y, test_x, test_y):
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(train_x, train_y)
    predict_y_2 = rf_model.predict(test_x)
    print(classification_report(test_y, predict_y_2))
    print("Accuracy:", accuracy_score(predict_y_2, test_y))

def Logic(train_x, train_y, test_x, test_y):
    lr_model = LogisticRegression(solver='lbfgs', multi_class='auto')
    lr_model.fit(train_x, train_y)
    predict_y_3 = lr_model.predict(test_x)
    print(classification_report(test_y, predict_y_3))
    print("Accuracy:", accuracy_score(predict_y_3, test_y))
    #Save the model
    filename = 'final_model.sav'
    pi.dump(lr_model, open(filename, 'wb'))