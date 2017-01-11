import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

ti_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
full = ti_df.append(test_df, ignore_index=True)


# Drop unecessary columns

# Cabin has too many null values to be useful
# extracting title from name may be useful
full = full.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)


# Embarked
embark_dummies = pd.get_dummies(full['Embarked'])
embark_dummies.drop(['S'], axis=1, inplace=True) # S has a lot of null values
full.drop(['Embarked'], axis=1, inplace=True)
full = full.join(embark_dummies)


# Fare

# fill missing fare value, maybe use better method than median
full['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# convert from float to int
full['Fare'] = full['Fare'].astype(int)


# Age

# generate random values for ages
# can do better than random between mean +- std
rand = np.random.randint(full.Age.mean() - full.Age.std(),
        full.Age.mean() + full.Age.std(), size=full.Age.isnull().sum())

full['Age'][np.isnan(full['Age'])] = rand

# convert from float to int
full['Age'] = full['Age'].astype(int)


# Family

# seems that people with families are more likely to survive than those
# that came alone
full['Family'] = full['Parch'] + full['SibSp']
full['Family'].loc[full['Family'] > 0] = 1
full['Family'].loc[full['Family'] == 0] = 0

# drop Parch and SibSp
full.drop(['Parch', 'SibSp'], axis=1, inplace=True)


# Sex

# children seem to have a high survival rate
def _get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

full['Person'] = full[['Age','Sex']].apply(_get_person, axis=1)

# no need for sex column since creating person column
full.drop(['Sex'], axis=1, inplace=True)

# convert to dummy variables
person_dummies = pd.get_dummies(full['Person'])
person_dummies.columns = ['Child','Female','Male']
# drop male, it has the lowest avg of survived passengers
person_dummies.drop(['Male'], axis=1, inplace=True)
full = full.join(person_dummies)
full.drop(['Person'], axis=1, inplace=True)


# Pclass

# convert to dummy variables
pclass_dummies = pd.get_dummies(full['Pclass'])
pclass_dummies.columns = ['Class1', 'Class2', 'Class3']
# drop 3rd class, it has lowest avg of survived passengers
pclass_dummies.drop(['Class3'], axis=1, inplace=True)
full = full.join(pclass_dummies)
full.drop(['Pclass'], axis=1, inplace=True)


# Define training and testing sets

x_train = full[:891].drop(['Survived'], axis=1)
y_train = full[:891]['Survived']
x_test = full[891:].drop(['Survived'], axis=1)


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_logreg = logreg.predict(x_test).astype(int)
print('logreg score: ' + str(logreg.score(x_train, y_train)))


# Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
y_pred_svm = svc.predict(x_test).astype(int)
print('svm score: ' + str(svc.score(x_train, y_train)))


# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test).astype(int)
print('knn score: ' + str(knn.score(x_train, y_train)))


# Guassian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_gnb = gnb.predict(x_test).astype(int)
print('gnb score: ' + str(gnb.score(x_train, y_train)))


# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test).astype(int)
print('rf score: ' + str(rf.score(x_train, y_train)))


# Get correlation coefficients
coeff_df = pd.DataFrame(x_train.columns)
coeff_df.columns = ['Features']
coeff_df['Coefficient Estimate'] = pd.Series(logreg.coef_[0])
print(coeff_df)


# Generate kaggle submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_pred_rf})

