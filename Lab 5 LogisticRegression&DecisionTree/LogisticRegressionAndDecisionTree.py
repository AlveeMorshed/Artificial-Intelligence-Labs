# importing necessary libraries
import pandas as pd
import numpy as np

"""Removing Null values / Handling Missing data"""
# Please change the filepath if it shows file not found, 
# used IDE: Spyder, the code and dataset are kept in the same directory
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

'''Visualizing current Correlation between labels after feature encoding'''
mushEd_corr = mushEd_df.corr()
print(mushEd_corr)
sns.heatmap(mushEd_corr, cmap='YlGnBu')
mushEd_df = mushEd.drop(['bruises'], axis = 1)

'''Splitting dataset in 8:2 ratio for scaling'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mushEd_df.iloc[:,1:],  mushEd_df['class'], test_size=0.2, random_state=42,  stratify = mushEd_df['class'])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

'''Calculating percentage per target label'''
n_edible= np.sum(y_train=='edible')
n_poisonous= np.sum(y_train=='poisonous')
total= n_edible + n_poisonous
print("No. of poisonous mushrooms is",n_poisonous)
print("No. of edible mushrooms is",n_edible)
print("% of posionous mushrooms is",(n_poisonous/total)*100)
print("% of edible mushrooms is",(n_edible/total)*100)

''' Feature scaling using Standard Scaler '''
# preprocessing using zero mean and unit variance scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''Classification using Support Vector Machine'''

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

print("Training accuracy of the model is {:.2f}".format(svc.score(X_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(X_test, y_test)))

predictions = svc.predict(X_test)
print(predictions)


from sklearn.metrics import confusion_matrix
mat=confusion_matrix(predictions, y_test)
print(mat)

from seaborn import heatmap
heatmap(mat , cmap="Pastel1_r", xticklabels=['class_0' ,'class_1' ,'class_2'], yticklabels=['class_0' ,'class_1', 'class_2'], annot=True)


'''Classification using Decision Tree'''

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(X_test, y_test)))


predictions = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(predictions, y_test)
print(mat)

from seaborn import heatmap
heatmap(mat , cmap="Pastel1_r", xticklabels=['class_0' ,'class_1' ,'class_2'], yticklabels=['class_0' ,'class_1', 'class_2'], annot=True)


'''Classification using Decision Tree'''

from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(X_test, y_test)))

predictions = nnc.predict(X_test)
print(predictions)

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(predictions, y_test)
print(mat)

from seaborn import heatmap
heatmap(mat , cmap="Pastel1_r", xticklabels=['class_0' ,'class_1' ,'class_2'], yticklabels=['class_0' ,'class_1', 'class_2'], annot=True)

               
'''Accuracy comparison'''
 




              
fig = plt.figure()
axes = fig.add_axes([0,0,0.7,1.6])
frequency = [accuracy_score(y_test, predictions), score]
points = ['Logistic Regression', 'Decision Tree']
data = frequency
axes.bar(points, frequency)
axes.set_title('Accuracy Comparison')
axes.set_xlabel('Classification Methods')
axes.set_ylabel('Frequency')
width = 0.4
plt.show()