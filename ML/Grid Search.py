""" 
GRID SEARCH
Se utiliza para encontrar los parametros optimos y el mejor modelo.
Nos dice si es mejor un modelo lineal o un modelo como kernel (tecnicas no lineales)
Optamos por coger el archivo de kfold anterior para comparar.
"""
# LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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
print('Matriz de confunsion')
print(cm)
print("\n")

# Aplicamos k-Fold Cross Validation, para que todos los elementos se utilicen para entrenamiento
# y testing y asi tener otra medida visual de la dispersion del sesgo y varianza
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('KFOLD VALIDATION')
print("Precision: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("\n")

# Aplicamos Grid Search para encontrar el mejor modelo y los mejores parametros
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

# Entrenamos el modelo ajustandolo con los datos
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('GRID SEARCH')
print("Best Precision: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
print("\n")

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