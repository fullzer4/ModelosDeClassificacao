import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt 

# imports

churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")

# pegando csv

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

# definindo e preparando para o sklearn

print(churn_df.head(), "\n linhas e colunas: ", churn_df.shape)
# visualizando nossas definicoes e o numero de colunas e linhas

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

y = np.asarray(churn_df['churn'])
y [0:5]

#definindo X e Y

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# normalizando nosso dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('\n Train set:', X_train.shape,  y_train.shape)
print ('\n Test set:', X_test.shape,  y_test.shape)

# definindo o nosso treino e test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

# modelando nosso algoritimo

yhat = LR.predict(X_test)
yhat

# prevendo usando test set

yhat_prob = LR.predict_proba(X_test)
yhat_prob

#

from sklearn.metrics import jaccard_score
print("\n Avaliando usando jaccard: ", jaccard_score(y_test, yhat,pos_label=0))

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
confusion_matrix(y_test, yhat, labels=[1,0])

cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])

np.set_printoptions(precision=2)

print("\n", classification_report(y_test, yhat))
# ver a precisao do nosso modelo

from sklearn.metrics import log_loss
print("\n LogLoss: ", log_loss(y_test, yhat_prob))

# vendo log loss

LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("\n LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))

# vendo a LogLoss usando mas com o solver e o reg diferente