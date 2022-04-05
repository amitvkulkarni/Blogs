#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------- Building models in unstructured way -------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv("hcvdata.csv")
df.drop('Unnamed: 0', inplace = True, axis =1)

cat_dict = {'0=Blood Donor':0, '0s=suspect Blood Donor':0, '1=Hepatitis':1,
'2=Fibrosis':2, '3=Cirrhosis':3}

df['Category'].map(cat_dict)
df['Category'] = df['Category'].map(cat_dict)

# Change Gender
df['Sex'] = df['Sex'].map({'m':1,'f':0})

df = df[['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST',
'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT','Category']]


# Fill Missing Values with 0
df.fillna(0,inplace=True)

# Features & Labels
X_features = df.drop('Category',axis=1)
y_labels = df['Category']

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X_features, y_labels,test_size=0.3,random_state=42)

lr = LogisticRegression()
lr.fit(X_train,y_train)

# Prediction Accuracy
print("Accuracy:", lr.score(X_test,y_test))

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------- Building models with ML pipelines ---------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("hcvdata.csv")

# Features & Labels
X_features = df.drop('Category',axis=1)
y_labels = df['Category']

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X_features, y_labels,test_size=0.3,random_state=42)

numeric_features = df.select_dtypes(
    include=['int64', 'float64']).drop(['Unnamed: 0'], axis =1).columns
categorical_features = df.select_dtypes(
    include=['object']).drop(['Category'], axis=1).columns


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value= 2)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", rf.score(X_test,y_test))


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Building multiple models with same pipeline
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
classifiers = [
    KNeighborsClassifier(3),
    RandomForestClassifier(),
    LogisticRegression(),
    
    ]

for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))






