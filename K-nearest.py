import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier

# imports

print(r"""        ___________    .__  .__                         _____  
 ___.__.\_   _____/_ __|  | |  | ________ ___________  /  |  | 
<   |  | |    __)|  |  \  | |  | \___   // __ \_  __ \/   |  |_
 \___  | |     \ |  |  /  |_|  |__/    /\  ___/|  | \/    ^   /
 / ____| \___  / |____/|____/____/_____ \\___  >__|  \____   | 
 \/          \/                        \/    \/           |__| """)
print("\n Practices in IBM Course in Coursera")

# assinatura

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

# lendo nossos dados

print("\n", df) # vendo tabela
print("\n", df['custcat'].value_counts()) # ver quanto tem de cada classe em nosso modelo

# preparando nossos dados em um numpy array

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

# vendo nossos valores de classificacao

y = df['custcat'].values
y[0:5]

# treino e teste

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# train our model

k = 9
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

# Predict

yhat = neigh.predict(X_test)
yhat[0:5]

# precisâo

print("")
print("===========================")
print("          teste 1          ")
print("===========================")
print("")

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

print("")
print("===========================")
print("          teste 2          ")
print("===========================")
print("")

k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))

print("")
print("===========================")
print("    vendo melhor modelo    ")
print("===========================")
print("")

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)