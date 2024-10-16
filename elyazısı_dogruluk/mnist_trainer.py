import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import json

# MNIST veri setini yükle ve ön işle
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Basit bir sinir ağı modeli tanımla
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
model.fit(x_train, y_train, epochs=10)

# Modeli değerlendir
loss, accuracy = model.evaluate(x_test, y_test)

# Modeli kaydet
model.save('mnist_model.h5')

# Doğruluk oranını kaydet
with open('accuracy.json', 'w') as f:
    json.dump({'accuracy': accuracy}, f)
