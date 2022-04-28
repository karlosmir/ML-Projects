
"""
ALgoritmo de XGBOOST con DATASET de Banca
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# PATH
path = "C:/Users/USUARIO/Desktop/CursoML/Data/"
data_client= pd.read_csv(path + "Churn_Modelling.csv")
#print(data_client.info())  # Son 10.000 datos
#print(data_client.head())

# Preprocesado de las columnas  del DF que nos interesa
X = data_client.iloc[:, 3:-1].values
y = data_client.iloc[:, -1].values

# Codificamos la variables String a variables categoricas con el enconder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Lo mismo, Codificamos la variables String a variables categoricas con el enconder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Dividimos los datos usando la funcion train_test_split para Train Y test del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Llamamos a la funcion XGBOOST y creamos un objeto. Ajustamos el modelo y lo entrenamos
XGBOOST = XGBClassifier()
XGBOOST.fit(X_train, y_train)

# Con el modelo entrenado, hacemos una prediccion de los datos del testing
y_pred = XGBOOST.predict(X_test)

# Hacemos la matriz de confusion para evaluar los datos, se observa en ella que
# tenemos 1497 personas correctas que se quedan en el banco y 209 que se van del banco
# , también hay falsos positivos y falsos negativos pero con una desviacion menor
cm = confusion_matrix(y_test, y_pred)

# Aplicamos el algoritmo de K-fold cross de validacion para medir la precision de nuestro algoritmo
# teniendo asi una mejor medida (validacion cruzada)
accuracies = cross_val_score(estimator = XGBOOST, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))