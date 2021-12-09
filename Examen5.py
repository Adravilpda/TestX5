# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
conway.py 
A simple Python/matplotlib implementation of Conway's Game of Life.
Author: Mahesh Venkitachalam
"""
import pandas as pn
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
iris=datasets.load_iris()
#X = iris.data
#y = iris.target

A=pn.read_csv('cann.txt',skiprows=0,usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
V1=np.array(A)

B=pn.read_csv('cann.txt.',skiprows=0,usecols=[0])
V2=np.array(B)

X=V1
y=V2
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.tree import DecisionTreeClassifier

cv=StratifiedShuffleSplit(n_splits=10,test_size=0.25,random_state=0)
iteraciones=cv.split(X,y)

clasificador=MLPClassifier(solver='lbfgs')

from sklearn.metrics import confusion_matrix
for indice, (train_indice,test_indice) in enumerate(iteraciones):
    print("---------")
    #print(indice)
    #print(train_indice)
    #print(y[train_indice])
    #print()
    clasificador.fit(X[train_indice],y[train_indice])
    y_obtenido=clasificador.predict(X[test_indice])
    print(y[test_indice])
    print(y_obtenido)
    print(confusion_matrix(y[test_indice], y_obtenido))
