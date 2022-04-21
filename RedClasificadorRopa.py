#### RED NEURONAL QUE ENTRENA UN MODELO PARA CLASIFICAR IMAGENES
### BIBLIOTECA fashion_MNIST
#### OBJETIVO QUE EL MODELO SEA CAPAZ DE RECONOCER LAS IMAGENES CARGADAS

### BIBLIOTECAS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Datasets cargados desde KERAS fashion_MNIST
"""
zalando = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = zalando.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

### ANALIZAMOS EL TAMAÑO DE LAS FOTOS POR SI ES NECESARIO REDIMENSIONAR AL MISMO TAMAÑO PERO NO, 28X28px
print(train_images.shape)
print(test_images.shape)
 
### OBSERVAMOS 1 FIGURATRAIN DEL DATASET
plt.figure()
plt.imshow(train_images[24])
plt.colorbar()
plt.grid(False)
plt.show()

# MAPEADO DE TODAS LAS ETIQUETAS
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

### PREPROCESADO
train_images = train_images / 255.0
test_images = test_images / 255.0

### MODELO SECUENCIAL
model = tf.keras.Sequential()

########## CAPA01 ######### 
# Transforma el formato de las imagenes de un arreglo bi-dimensional 
# (de 28 por 28 pixeles) a un arreglo uni dimensional (de 28*28 pixeles = 784 pixeles).
# Esta capa es una capa no apilada de filas de pixeles en la misma
# imagen y alineandolo. Esta capa no tiene parametros que aprender; solo 
#reformatea el set de datos.
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#CAPA02
## Capa "densamente" (completamente) conectada con 128 nodos
#La modificación de los parámetros predeterminados le permite utilizar umbrales 
#distintos de cero, cambiar el valor máximo de la activación y utilizar un 
#múltiplo distinto de cero de la entrada para valores por debajo del umbral.
model.add(tf.keras.layers.Dense(128, activation='relu'))
#CAPA03
## Capa ultima  "densamente" (completamente) conectada con 10 nodos softmax
## devolviendo 10 salidas, devolviendo cada nodo una probabilidad de que la imagen
## clasificada pertenezca a una de las 10 clases
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Resumen de capas
model.summary()
# Optimizador ADAM, Monitoriza la "exactitud",  "crossentropy "esta función de 
# pérdida de entropía cruzada se usa cuando haya dos o más clases de etiquetas
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# AJUSTE
model.fit(train_images, train_labels, epochs=10)

# EVALUAMOS EL MODELO CON NUESTRO DATASETS TEST
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# VARIABLES PREDICCIONES DEL MODELO
predictions = model.predict(test_images)

print(predictions[0])
# MEJOR PREDICCIONES CONFIANZA LA ETIQUETA TIP O9 DE ANKLE BOOT
print(np.argmax(predictions[0]))


# GRAFICAS CON LA PREDICCION REALIZADA
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# Graficas
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()









