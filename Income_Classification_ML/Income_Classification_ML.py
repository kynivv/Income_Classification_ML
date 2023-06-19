import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# Data Import
df = pd.read_csv('income_evaluation.csv')
df = df.drop(' capital-gain', axis=1)
df = df.drop(' capital-loss', axis=1)


# EDA
#print(df)
#print(df.dtypes)
#print(df.isnull().sum())


# Data Transformation
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
#print(df.dtypes)
#print(df)


# Train Test Split
features = df.drop(' income', axis=1)
target = df[' income']
#print(features, target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.22, random_state=22)
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Model Training
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

models = [ExtraTreesClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), SVC(), LinearSVC(), KNeighborsClassifier(), DummyClassifier(), BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=100)]

for m in models:
    m.fit(X_train, Y_train)
    pred_train = m.predict(X_train)
    print(f'Train Accuracy of {m} is : {accuracy_score(Y_train, pred_train)}')
    
    pred_test = m.predict(X_test)
    print(f'Test Accuracy of {m} is : {accuracy_score(Y_test, pred_test)}')
    print('|')
    print('|')
    print('|')


