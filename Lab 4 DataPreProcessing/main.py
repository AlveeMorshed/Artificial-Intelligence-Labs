# importing necessary libraries
import pandas as pd
import numpy as np

"""#Removing Null values / Handling Missing data"""

mushEd = pd.read_csv('mushroom edibility classification dataset.csv')
print(mushEd.head(3))
print(mushEd.shape)
print(mushEd.isnull().sum())

from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(mushEd[['cap-shape']])
impute.fit(mushEd[['cap-color']])

mushEd['cap-shape'] = impute.transform(mushEd[['cap-shape']])
mushEd['cap-color'] = impute.transform(mushEd[['cap-color']])

'''Visualizing current Correlation between labels'''
mushEd_df = pd.DataFrame(mushEd, columns=mushEd.keys())
mushEd_corr = mushEd_df.corr()
print(mushEd_corr)
import seaborn as sns
sns.heatmap(mushEd_corr, cmap='YlGnBu')

"""dropping columns"""
print(mushEd['veil-type'].unique())
print(mushEd['veil-color'].unique())
print(mushEd['ring-number'].unique())
mushEd = mushEd.drop(['Unnamed: 0','veil-type','veil-color','ring-number'], axis = 1)
print(mushEd.shape)

'''Visualizing current Correlation between labels after dropping'''
mushEd_df = pd.DataFrame(mushEd, columns=mushEd.keys())
mushEd_corr = mushEd_df.corr()
print(mushEd_corr)
sns.heatmap(mushEd_corr, cmap='YlGnBu')

'''Encoding categorical features'''
from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the binary encoding to the "bruises" column
print(mushEd['bruises'].unique())
mushEd['bruises_enc'] = enc.fit_transform(mushEd['bruises'])
mushEd_df = pd.DataFrame(mushEd, columns=mushEd.keys())

# Compare the two columns
print(mushEd[['bruises', 'bruises_enc']].head())
mushEd_corr = mushEd_df.corr()
print(mushEd_corr)
sns.heatmap(mushEd_corr, cmap='YlGnBu')
mushEd_df = mushEd.drop(['bruises'], axis = 1)

'''Splitting dataset for scaling'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mushEd_df.iloc[:,1:],  mushEd_df['class'], random_state=1,  stratify = mushEd_df['class'])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

'''Calculating percentage per target label'''
n_edible= np.sum(y_train=='poisonous')
n_poisonous= np.sum(y_train=='edible')
total= n_edible + n_poisonous
print("No. of poisonous mushrooms is",n_poisonous)
print("No. of edible mushrooms is",n_edible)
print("% of posionous mushrooms is",(n_poisonous/total)*100)
print("% of edible mushrooms is",(n_edible/total)*100)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

knn.fit(X_train, y_train)

print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))

# preprocessing using zero mean and unit variance scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data
knn.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("KNN test accuracy: {:.2f}".format(knn.score(X_test_scaled, y_test)))