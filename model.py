# Data analysis and visualization
import pandas as pd
import numpy as np
from math import pi
import seaborn as sns


# metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt



from pltseaborn import plt_seaborn, plt_scatter, plt_bar
from label_encoder import LabelEncoderTrain, LabelEncoderTest
from classifiers import Decision,Randomc, Logic


#load training and testing data

file_train ='input/train_u6lujuX_CVtuZ9i.csv' 
file_test = 'input/test_Y3wMUE5_7gLdaTN.csv'
loan_train = pd.read_csv(file_train)
loan_test = pd.read_csv(file_test)

print(loan_train.head())

loan_train_cc = loan_train.copy()

loan_train.columns

loan_test.columns

loan_train.dtypes

loan_train.describe()

len(loan_train)

len(loan_test)

# we have missing data
loan_test.isna().values.any()


# Let's visualize the missing data in the TEST data
plt_seaborn(loan_train)



loan_train.isna().sum()


# We'll do a forward fill here, so, we get only 1 or 0 to fill the missing data

loan_train['Credit_History'].fillna(method='ffill', inplace=True)
loan_train['Credit_History'].isna().values.any()


# We'll fill this column using the median of the values

median_loan = loan_train['Loan_Amount_Term'].median()
loan_train['Loan_Amount_Term'].fillna((median_loan), inplace=True)
loan_train['Loan_Amount_Term'].isna().values.any()


# We'll fill this column using the median of the values

median_loan_amount = loan_train['LoanAmount'].median()
loan_train['LoanAmount'].fillna((median_loan_amount), inplace=True)
loan_train['LoanAmount'].isna().values.any()


# Count the values to know which occurs most frequently
loan_train['Self_Employed'].value_counts()


#Fill with mode
loan_train['Self_Employed'].fillna('No', inplace=True)
loan_train['Self_Employed'].isna().values.any()



# fill with mode
loan_train['Dependents'].fillna(0, inplace=True)
loan_train['Dependents'].isna().values.any()




loan_train['Married'].mode()


# fill with mode
loan_train['Married'].fillna('Yes', inplace=True)
loan_train['Married'].isna().values.any()


loan_train['Gender'].mode()


# fill with mode
loan_train['Gender'].fillna('Male', inplace=True)
loan_train['Gender'].isna().values.any()


# Let's run a quick check
loan_train.isna().sum()


# A preview of missing data in the testing set

loan_test.isna().sum()


# fill in credit history
loan_test['Credit_History'].fillna(method='ffill', inplace=True)

# fill in loan amount term
median_loan_test = loan_test['Loan_Amount_Term'].median()
loan_test['Loan_Amount_Term'].fillna((median_loan_test), inplace=True)

# fill in loan amount
median_loan_amount_test = loan_test['LoanAmount'].median()
loan_test['LoanAmount'].fillna((median_loan_amount_test), inplace=True)

# fill in self employed
loan_test['Self_Employed'].fillna('No', inplace=True)

# fill in dependents
loan_test['Dependents'].fillna(0, inplace=True)

# fill in gender
loan_test['Gender'].fillna('Male', inplace=True)


loan_test.isna().values.any()


# Let's run a final check

loan_test.isna().sum()


loan_train.duplicated().values.any()


loan_test.duplicated().values.any()

# Let's preview the data again

loan_train.head()



# Bar charts to get a high level view of categorical data
# plt_bar(loan_train)


# Here, I pass all categorical columns into a list

categorical_columns = loan_train_cc.select_dtypes('object').columns.to_list()


# Then, I filter the list to remove Loan_ID column which is not relevant to the analysis
categorical_columns[1:]


# This code loops through the list, and creates a chart for each

for i in categorical_columns[1:]: 
    plt.figure(figsize=(15,10))
    plt.subplot(3,2,1)
    sns.countplot(x=i ,hue='Loan_Status', data=loan_train_cc, palette='ocean')
    plt.xlabel(i, fontsize=14)


#Plot4- Scatterplot
plt_scatter(loan_train)



# Let's plot correlation overview of the variables.

fig, ax = plt.subplots(figsize=(9, 7))
correlations = loan_train.corr()
  
# plotting correlation heatmap
dataplot = sns.heatmap(correlations, cmap="YlGnBu", annot=True)
  
# displaying heatmap
plt.show()

# Let's take another preview of the data
loan_train.head()



#first identify all categorical columns & pass into a variable
objectlist_train = loan_train.select_dtypes(include = "object").columns


# Then Label Encoding for object to numeric conversion

LabelEncoderTrain(objectlist_train, loan_train)

# Now, repeat the same process to encode the test data

objectlist_test = loan_test.select_dtypes(include='object').columns

LabelEncoderTest(objectlist_test, loan_test)



# Now let's rerun correlation, with other numeric variables now added

fig, ax = plt.subplots(figsize=(10, 8))
correlations_ML = loan_train.iloc[:,1:].corr() # filer out the Loan_ID column as it is not relevant
sns.heatmap(correlations_ML, cmap="YlGnBu", annot=True)
plt.show()



x = loan_train.iloc[:,1:].drop('Loan_Status', axis=1) # drop loan_status column because that is what we are predicting
y = loan_train['Loan_Status']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=0)


# DecisionTreeClassifier
Decision(train_x, train_y, test_x, test_y)

# RandomForestClassifier
Randomc(train_x, train_y, test_x, test_y)

# LogisticRegression
Logic(train_x, train_y, test_x, test_y)


# There's a positive relationship between applicant income & loan amount.
# There's also a positive relationship between credit history and loan status.
# On average, men got more loans. Being married & educated (graduate) were also factors that resulted in loan approvals.
# For our ML model, at 84% accuracy, the Logistic Regression model is the most suitable to make this prediction.