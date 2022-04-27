"""
RED NEURONAL CONVOLUCIONAL, 
Dataset con fotos de Humanos y Caballos
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator # Genera las imagenes

# Preprocesado
# Rescala las imagenes del Train
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Rescala las imagenes del test
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creando el DF Training SET
training_set = train_datagen.flow_from_directory('C:/Users/USUARIO/Desktop/CursoML/Data/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creando el DF Test SET
test_set = test_datagen.flow_from_directory('C:/Users/USUARIO/Desktop/CursoML/Data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 10,
                                            class_mode = 'binary')

# Creamos la red RNC, Convolucion --> Pooling --> Flattenin --> Full Connect
RNC = tf.keras.models.Sequential()
# 1º Capa Convolucion2D
RNC.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
# 2º Capa - Pooling, Simplificamos los problemas y reduce las operaciones
RNC.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
# 3º Capa de Convolucion y Pooling
RNC.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
RNC.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
# 4º Capa - Flattening, adapta la estructura de forma vertical en una columna
RNC.add(tf.keras.layers.Flatten())
# Full Connection, añadimos la red neuronal totalmentne conectada
RNC.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Capa de Salida
RNC.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Funcion sigmoide

# Compilamos el modelos con el optimizador Adam y entropia cruzada binaria
RNC.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenamos el modelo
RNC.fit_generator(training_set,
                  steps_per_epoch = 40,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 20)