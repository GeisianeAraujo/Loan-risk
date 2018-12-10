# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:57:45 2018

@author: GeisyPC
"""

import pandas as pd

base = pd.read_csv('risco-credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
                 
from sklearn.tree import DecisionTreeClassifier, export
# entropy é a função para ganho de informação 
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)
# mostra a importancia de cada um dos atributos
print(classificador.feature_importances_)
# vizualização da arvore de decisão (salvando o "arvore.dot" para ser aberto no graphviz)
export.export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = ['alto', 'moderado', 'baixo'],
                       filled = True,
                       leaves_parallel=True)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(resultado)
print(classificador.classes_)