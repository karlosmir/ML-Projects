"""
Ejemplo basico de una ecuacion lineal basica con un dataset de valores aleatorios
"""
#LIBRERIAS
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(101)
import warnings
warnings.filterwarnings('ignore')

# Y = MX + B + noise ( Ecuacion lineal basica)
m = 2
b = 3
x = np.linspace(0,50,100)

noise = np.random.normal(loc=0, scale=4, size=len(x)) #normal... Normal / Gaussian distribution

# Y = MX + B + noise ( Ecuacion lineal basica)
y = 2*x + b + noise

# GRAFICA
fig, ax = plt.subplots()
ax.plot(y)
ax.set(xlabel='x', ylabel='y', title='y = 2x + b + noise')
plt.show();

# Modelo Secuencial
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(1,))) # Entrada 1
model.add(tf.keras.layers.Dense(4, activation='relu')) # 4 nodos en la capa
model.add(tf.keras.layers.Dense(4, activation='relu')) # 4 nodos en la capa
model.add(tf.keras.layers.Dense(1, activation='linear')) # Salida 1
model.summary() # Resumen de nuestro modelo

# Compilacion de nuestro modelo
# Adam: método de descenso de gradiente estocástico que se basa en la estimación 
# adaptativa de momentos de primer y segundo orden
# MeanSquaredError: Calcula la media de los cuadrados de errores entre etiquetas y predicciones.
model.compile(optimizer='adam',
              loss='MeanSquaredError',
              metrics=['accuracy'])
 
# Ajustamos el modelo
model.fit(x,y, epochs=200)
loss = model.history.history['loss']
epochs = range(len(loss))

# GRAFICAS
fig, ax = plt.subplots()
ax.plot(epochs, loss)
ax.set(xlabel='epochs', ylabel='Loss', title='Model')
plt.show();

x_predict = np.linspace(0,50,100)
y_predict = model.predict(x_predict)

plt.plot(x,y,'.')
plt.plot(x_predict, y_predict, 'r')
























