"""
RED NEURONAL ARTIFICIAL
DataSet Bancarios realista de 10.000 clientes
Contacta por Tasa de abandono de clientes
Predecir si un cliente es capaz de dejar el banco
Problema de clasificacion
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# PATH
path = "C:/Users/USUARIO/Desktop/CursoML/Data/"
data_client= pd.read_csv(path + "Churn_Modelling.csv")
#print(data_position.info())  # Son 10.000 datos
#print(data_client.head())

# Preprocesado de Datos
X = data_client.iloc[:, 3:-1].values
Y = data_client.iloc[:, -1].values

# Transformamos las columna Gender
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Pasar los datos State de string a variables Categoricas siendo Numericos
# MAS NUEVO COLUMTRANSFORMER para columna geography
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Dividimos los datos usando la funcion train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=(0))

# Escalado de variables, para agilizar el resultado 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Construir la red neuronal artificial con keras del modelo secuencial
RNA = tf.keras.models.Sequential()

# Regla de nodos, la capa oculta se pone la media entre la capa de entrada y salida
# 1º Capa de inputs
RNA.add(tf.keras.layers.Dense(units=6, activation='relu'))
# 2º Capa oculta de 6 nodos
RNA.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Ultima capa de salidad, con funcion sigmoide
RNA.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compilamos y ajustamos el modelo
RNA.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
RNA.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Prediccion de los resultados
y_pred = RNA.predict(X_test)
y_pred = (y_pred > 0.5) # Cuota intermedia del 50% que podria abandonar
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Elaborar una matriz de confunsion
cm = confusion_matrix(y_test, y_pred)
print(cm)















