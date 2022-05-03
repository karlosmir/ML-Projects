""" 
VALIDACION CRUZADA
k-Fold Cross Validation se utiliza para proporcionar una evaluacion relevante de
la eficicacia del modelo
"""
# LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

#PATH
path = "C:/Users/USUARIO/Desktop/CursoML/Data/"

# DATASET de un anuncio publicidad en una red social para saber si un cliente compraria o no un producto
dataset = pd.read_csv(path + 'Social_Network_Ads.csv')

# Preprocesado de columnas
X = dataset.iloc[:, [2, 3]].values # Datos indpendendientes, edad y salario
y = dataset.iloc[:, -1].values # Dato dependiente compra realizada
#print(data_client.info())  # Son 10.000 datos
#print(data_client.head())

# Escalado de los datos
sc = StandardScaler()
X = sc.fit_transform(X)

# Divimos los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Creamos un SuperVector de soporte C y ajustamos el modelo con los datos de entrenamiento 
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de los resultados
y_pred = classifier.predict(X_test)

# Matriz de confunsion para evaluar los datos
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n\n")

# Aplicamos k-Fold Cross Validation, para que todos los elementos se utilicen para entrenamiento
# y testing y asi tener otra medida visual de la dispersion del sesgo y varianza
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("\n\n")

# Visualizacion del Train
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()

# Visualizacion del TEST
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()