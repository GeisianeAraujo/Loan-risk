# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:54:06 2018

@author: GeisyPC
"""

import pandas as pd

base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

# transformar atributos categóricos em discretos (numerico)                 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
# importar naive bayes           
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
# gera a tabela de probabilidade
classificador.fit(previsores, classe)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)