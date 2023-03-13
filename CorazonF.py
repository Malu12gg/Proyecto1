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

df=pd.read_csv("C:/Users/ricky\Downloads/base/base/processed.cleveland.data.csv", sep = ",")

# Renombrar columnas
df.columns
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])
df.head()


df= df.dropna()



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

df["age"] = df["age"].apply(int)
for i in df["age"].index:

    if 29 <= df["age"][i]<= 39:
        df["age"][i]= "29-39"
    elif 40 <= df["age"][i]<= 49:
        df["age"][i]= "40-49"
    elif 50 <= df["age"][i]<= 59: 
        df["age"][i]= "50-59"
    elif 60 <= df["age"][i]: 
        df["age"][i]= "60+" 
       

for i in df["trestbps"].index:

    if 180<= df["trestbps"][i]:
        df["trestbps"][i]= "Hipertensión crisis"
    elif 140 <= df["trestbps"][i]< 180:
        df["trestbps"][i]= "Hipertensión nivel 2"
    elif 130 <= df["trestbps"][i]<= 139: 
        df["trestbps"][i]= "Hipertensión nivel 1"
    elif 120<= df["trestbps"][i]<= 129: 
        df["trestbps"][i]= "elevada"
    elif 80 <= df["trestbps"][i]<= 120: 
        df["trestbps"][i]= "normal"
   
for i in df["oldpeak"].index:

    if 2.55<= df["oldpeak"][i]:
        df["oldpeak"][i]= "Terrible"
    elif 1.5 <= df["oldpeak"][i]< 2.55:
        df["oldpeak"][i]= "Normal"
    elif  df["oldpeak"][i]<= 1.5: 
        df["oldpeak"][i]= "Low"
   
for i in df["chol"].index:

    if 230<= df["chol"][i] and df["sex"][i] ==0:
        df["chol"][i]= "Nivel peligroso"
    elif 240<= df["chol"][i] and df["sex"][i] ==1:
        df["chol"][i]= "Nivel peligroso"
    elif 190<= df["chol"][i]< 229 and df["sex"][i] ==0:
        df["chol"][i]= "Nivel riesgoso"
    elif 200<= df["chol"][i]< 239 and df["sex"][i] ==1:
        df["chol"][i]= "Nivel riesgoso"  
    elif 200> df["chol"][i]< 229 and df["sex"][i] ==1:
        df["chol"][i]= "Nivel saludable"
    elif 190> df["chol"][i] and df["sex"][i] ==0:
        df["chol"][i]= "Nivel saludable"
   
for i in df["thalach"].index:

     if df["age"][i]=="29-39": 
        if 84< df["thalach"][i]:
         df["thalach"][i]= "Inadecuada"
     
        elif 72<= df["thalach"][i] <=84:
            df["thalach"][i] = "normal"
        
        elif 64<= df["thalach"][i]<=71:
            df["thalach"][i]= "Buena"    
        elif 60<= df["thalach"][i]< 62:
            df["thalach"][i]= "excelente"

     elif df["age"][i]=="40-49" and 90< df["thalach"][i]:
         df["thalach"][i]= "Inadecuada"
     elif df["age"][i]=="40-49" and 74<= df["thalach"][i]<=89:
        df["thalach"][i]= "normal"
     elif df["age"][i]=="40-49" and 66<= df["thalach"][i]<=73:
         df["thalach"][i]= "Buena"  
     elif df["age"][i]=="40-49" and 64<= df["thalach"][i]< 66:
       df["thalach"][i]= "excelente" 

     elif  df["age"][i]=="60+" and 90< df["thalach"][i] or  df["age"][i]=="50-59"and 90< df["thalach"][i]: 
        df["thalach"][i]= "Inadecuada"
    
     elif  df["age"][i]=="60+" and 76<= df["thalach"][i]<=89 or  df["age"][i]== "50-59" and 76<= df["thalach"][i]<=89:
        df["thalach"][i]=  "normal"

     elif  df["age"][i]=="60+" and 68<= df["thalach"][i]<=75 or  df["age"][i]== "50-59" and 68<= df["thalach"][i]<=75:
        df["thalach"][i]=  "Buena" 
    
     elif  df["age"][i]=="60+" and 66<= df["thalach"][i]<=67 or  df["age"][i]== "50-59" and 66<= df["thalach"][i]<=67:
        df["thalach"][i]="excelente" 
    

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
