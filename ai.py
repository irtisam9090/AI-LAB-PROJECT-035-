import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  
    Dense(128, activation='relu'),    
    Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, 
          epochs=5, 
          batch_size=32, 
          validation_data=(test_images, test_labels))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

predictions = model.predict(test_images[:5])
print("Predicted labels:", np.argmax(predictions, axis=1))
print("True labels:", np.argmax(test_labels[:5], axis=1))

import matplotlib.pyplot as plt
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {np.argmax(test_labels[i])}, Pred: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
