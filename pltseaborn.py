import seaborn as sns
import matplotlib.pyplot as plt

def plt_seaborn(loan_train):
    plt.figure(figsize=(10,6))
    sns.displot(
    data=loan_train.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25)

    plt.show()

def plt_scatter(loan_train):
    fig, ax = plt.subplots(2,2, figsize=(14,12))
    sns.scatterplot(data=loan_train,x="ApplicantIncome", y="LoanAmount",s=70, hue="Loan_Status", palette='ocean',ax=ax[0,0])
    sns.histplot(loan_train, x=loan_train['LoanAmount'], bins=10, ax=ax[0,1])
    sns.scatterplot(data=loan_train,x='CoapplicantIncome', y='LoanAmount',s=70, hue='Loan_Status',palette='ocean', ax=ax[1,0])
    sns.scatterplot(data=loan_train,x='Loan_Amount_Term', y='LoanAmount', s=70, hue='Loan_Status',palette='ocean', ax=ax[1,1])

    plt.show()

def plt_bar(loan_train):
    fig, ax = plt.subplots(3, 2, figsize=(16, 18))

    loan_train.groupby(['Gender'])[['Gender']].count().plot.bar(
        color=plt.cm.Paired(np.arange(len(loan_train))), ax=ax[0,0])
    loan_train.groupby(['Married'])[['Married']].count().plot.bar(
        color=plt.cm.Paired(np.arange(len(loan_train))), ax=ax[0,1])
    loan_train.groupby(['Education'])[['Education']].count().plot.bar(
        color=plt.cm.Paired(np.arange(len(loan_train))), ax=ax[1,0])
    loan_train.groupby(['Self_Employed'])[['Self_Employed']].count().plot.bar(
        color=plt.cm.Paired(np.arange(len(loan_train))), ax=ax[1,1])

    loan_train.groupby(['Loan_Status'])[['Loan_Status']].count().plot.bar(
        color=plt.cm.Paired(np.arange(len(loan_train))),ax=ax[2,0])
    loan_train.groupby(['Property_Area'])[['Loan_Status']].count().plot.bar(
        color=plt.cm.Paired(np.arange(len(loan_train))),ax=ax[2,1])

    plt.show()