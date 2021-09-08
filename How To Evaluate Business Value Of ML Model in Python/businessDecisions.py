import wget
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import kds



url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
wget.download(url) 
zf = zipfile.ZipFile('bank-additional.zip')
df = pd.read_csv(zf.open('bank-additional/bank-additional-full.csv'), sep=';')

df = df[['y', 'duration', 'campaign', 'pdays', 'previous', 'euribor3m']]
# rename target class value 'yes' for better interpretation
df.y[df.y == 'yes'] = 'term deposit'

# converting the target variable to categorial and also encoding it
df.y = pd.Categorical(df.y)
df['y'] = df.y.cat.codes


# define target vector y
y = df.y
# define feature matrix X
X = df.drop('y', axis = 1)

# Create the necessary datasets to build models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2021)

results = pd.DataFrame(columns=['Logistic','Logistic_Rank','RF', 'RF_Rank'])

# Logistic regression model
clf_glm = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg').fit(X_train, y_train)
prob_glm = clf_glm.predict_proba(X_test)
max_prob_glm = round(pd.DataFrame(np.amax(prob_glm, axis=1), columns = ['prob_glm']),2)
max_prob_glm['Decile_rank_glm'] = pd.cut(max_prob_glm['prob_glm'], 10, labels = np.arange(10,0, -1))


# Random forest model
clf_rf = RandomForestClassifier().fit(X_train, y_train)
prob_rf = clf_rf.predict_proba(X_test)
max_prob_rf = pd.DataFrame(np.amax(prob_rf, axis=1), columns = ['prob_rf'])
max_prob_rf['Decile_rank_rf'] = pd.cut(max_prob_rf['prob_rf'], 10, labels = np.arange(10,0, -1))




results['Logistic'] = max_prob_glm['prob_glm']
results['Logistic_Rank'] = max_prob_glm['Decile_rank_glm']

results['RF'] = max_prob_rf['prob_rf']
results['RF_Rank'] = max_prob_rf['Decile_rank_rf']
results['Final_Rank'] = results['Logistic_Rank'] + results['RF_Rank']
results.sort_values(by='Final_Rank', ascending=False)
results['Final_Decile'] = pd.cut(results['Final_Rank'],10, labels = False)

# results.to_csv('reco.csv')


kds.metrics.plot_cumulative_gain(y_test.to_numpy(), prob_glm[:,1])
kds.metrics.report(y_test, prob_glm[:,1])


kds.metrics.plot_cumulative_gain(y_test.to_numpy(), prob_rf[:,1])
kds.metrics.report(y_test, prob_rf[:,1])



