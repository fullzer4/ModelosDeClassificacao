import sys
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.tree as tree

print(r"""        ___________    .__  .__                         _____  
 ___.__.\_   _____/_ __|  | |  | ________ ___________  /  |  | 
<   |  | |    __)|  |  \  | |  | \___   // __ \_  __ \/   |  |_
 \___  | |     \ |  |  /  |_|  |__/    /\  ___/|  | \/    ^   /
 / ____| \___  / |____/|____/____/_____ \\___  >__|  \____   | 
 \/          \/                        \/    \/           |__| """)
print("\n Practices in IBM Course in Coursera")

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

y = my_data["Drug"]
y[0:5]

# Definindo treino e teste

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X test set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # mostar os parametros padraõs

drugTree.fit(X_trainset,y_trainset)

# Prevendo

predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

# Precisão do modelo

import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

tree.plot_tree(drugTree)
plt.show()