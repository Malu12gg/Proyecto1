#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:27:44 2023

@author: valeriarrondon
"""

import plotly.express as px
import pandas as pd
import itertools
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output

from pgmpy.sampling import BayesianModelSampling

na_values = ["?"]

df = pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 1/Proyecto/datos.txt",
                 sep = ",", na_values = na_values , header = None)

# Renombrar columnas
df.columns
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])
df.head()


df= df.dropna()
df.isnull().sum()


model = BayesianNetwork ([( "age" , "fbs" ),
                          ( "age" , "chol" ),
                          ( "age", "trestbps" ),
                          ( "age" , "thalach" ),
                          ( "sex", "chol" ),
                          ( "cp" , "thal" ),
                          ( "cp" , "slope" ),
                          ( "cp" , "num" ),
                          ( "cp", "restecg" ),
                          ( "trestbps" , "cp" ),
                          ( "chol" , "trestbps" ),
                          ( "fbs" , "trestbps" ),
                          ( "restecg", "slope" ),
                          ( "restecg" , "num" ),
                          ( "thalach" , "slope" ),
                          ( "exang" , "cp" ),
                          ( "exang" , "num" ),
                          ( "oldpeak" , "num" ),
                          ( "slope" , "oldpeak" ),
                          ( "num" , "thal" ),
                          ( "num" , "ca" )])


from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import LabelEncoder as le

# EDAD
df["age"] = pd.cut(df['age'], bins=4, labels=['29-39', '40-49', '50-59', '60-79'])

# PRESIÓN SANGUÍNEA EN REPOSO
intervalos2 = [0, 80, 120, 129, 139, 179, 600]
categorias2 = ['hipotensión', 'normal', 'elevada', 'hiptertensión nivel 1', 'hiptertensión nivel 2', 'crisis hipertensión']
df['trestbps'] = pd.cut(df['trestbps'], bins=intervalos2, labels=categorias2)

# OLDPEAK
intervalos3 = [0, 1.4, 2.5, 7]
categorias3 = ['baja', 'normal', 'terrible']
df['oldpeak'] = pd.cut(df['oldpeak'], bins=intervalos3, labels=categorias3)

# COLESTEROL
intervalos = [0, 200, 239, 600]
categorias = ['saludable', 'riesgoso', 'peligroso']
df['chol'] = pd.cut(df['chol'], bins=intervalos, labels=categorias)

# THLACH
intervalo_edad = [29,39,49,59,79]
intervalo_frec = [0, 90, ]

categoria_edad = ['29-39', '40-49', '50-59', '60-79']
categoria_frec = ['inadecuado', 'normal', 'buena', 'excelente']

rangos = {(29,39):{'inadeucado': (84,300),
                   'normal': (72,84),
                   'buena': (64,71),
                   'excelente': (60,62)},
          
          (40-49):{'inadeucado': (90,300),
                   'normal': (74,89),
                   'buena': (66,73),
                   'excelente': (64,66)},
          
                    
          (50-59):{'inadeucado': (90,300),
                   'normal': (76,89),
                   'buena': (68,75),
                   'excelente': (66,67)},
          
          (60-79):{'inadeucado': (90,300),
                   'normal': (76,89),
                   'buena': (68,75),
                   'excelente': (66,67)}
          }

df['thalach'] = df.apply(lambda row: 'inadeucado' if row['thalach'] < rangos[row['age']]['inadeucado'][0] 
                                                      or row['thalach'] > rangos[row['age']]['inadeucado'][1] 
                                                      else ('normal' if row['thalach'] < rangos[row['age']]['normal'][1] 
                                                                      else ('buena' if row['thalach'] < rangos[row['age']]['buena'][1] 
                                                                                      else 'excelente')), axis=1)


print(df)
 

emv = MaximumLikelihoodEstimator( model = model , data = df )
# Estimar para nodos sin padres
# Estimar para nodo age
cpdem_age = emv.estimate_cpd( node ="age")
print( cpdem_age )
# Estimar para nodo sex
cpdem_sex = emv.estimate_cpd( node ="sex")
print( cpdem_sex )
# Estimar para nodo exang
cpdem_exang = emv.estimate_cpd( node ="exang")
print( cpdem_exang )
model.fit(data=df , estimator = MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(model.get_cpds(i) )

    #estimación por metodo de Bayer
from pgmpy.estimators import BayesianEstimator

eby = BayesianEstimator ( model = model , data = df ) 
cpdby_num = eby.estimate_cpd(node="num", prior_type="dirichlet", pseudo_counts= [[100000], [100000],[100000],[100000]])
print(cpdby_num)

    
    
    
    
    
    
    