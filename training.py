from neural_network_architecture import CNN_architecture
from data_preprocessing import x_y_images
from scripts import create_dataset
import numpy as np
import tensorflow as tf

checkpoint_path = "training/cp.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

model = CNN_architecture.model

model.compile(
    optimizer='adam',              # Optimizer
    loss='binary_crossentropy',  # Loss function
    metrics=['accuracy']           # Metric to track
)

x, y = create_dataset.generate()

x_numpy_array = np.array(x)
y_numpy_array = np.array(y)

def train_test_split(array):
    upper_limit = int(len(array)*0.8)
    return array[:upper_limit], array[upper_limit:]

x_train, x_test = train_test_split(x_numpy_array)
y_train, y_test = train_test_split(y_numpy_array)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    callbacks=[checkpoint_callback]
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
